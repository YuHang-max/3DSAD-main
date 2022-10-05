# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
# VOTENET AND DETR MODEL CODEBASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')  # use c++ codes; setup install needed
sys.path.append(MODEL_DIR)
# print('\n'.join(sys.path))

class VoteDetr(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', vote_stage=1,
                 config_backbone=None, config_transformer=None, quality_channel=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.quality_channel = quality_channel
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.vote_stage = vote_stage
        assert vote_stage in [1, 2, 3]

        # Backbone point feature learning
        if config_backbone is None:
            print('DETR - BACKBONE: using default pointnet backbone! (votenet)')
            config_backbone = {'name': 'votenet_backbone'}
        if config_backbone['name'] == 'votenet_backbone':
            print('detr-backbone: using votenet-backbone (num_pc=1024)')
            from backbone_module import Pointnet2Backbone
            from voting_module import VotingModule
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
            # Hough voting
            self.with_voting = True
            self.vgen1 = VotingModule(self.vote_factor, 256)
        elif config_backbone['name'] == 'pointnet_cpp':
            print('detr-backbone: using votenet-backbone (pointnet cpp; num_pc=1024)')
            from backbone_module import Pointnet2Backbone
            from voting_module import VotingModule
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
            self.with_voting = False
        elif config_backbone['name'] == 'pointnet':
            print('detr-backbone: using pointnet', '<<< Warning! it is not votenet backbone!')
            POINT2_DIR = os.path.join(ROOT_DIR, 'pointnet2_python')
            sys.path.append(POINT2_DIR)
            from dump_helper import dump_results
            from pointnet_cls import get_model
            self.backbone_net = get_model(config_backbone)
            self.with_voting = False
        else:
            raise NotImplementedError(config_backbone)

        # Vote aggregation and detection
        print(self.sampling, '<< sampling')
        if self.sampling in ('vote_fps', 'seed_fps', 'random', 'vote_dist_fps'):
            from proposal_votenet import ProposalModule
            self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                       mean_size_arr, num_proposal, sampling, config_transformer=config_transformer, quality_channel=quality_channel)
        elif self.sampling in ('vote_fps_bbox', 'seed_fps_bbox', 'random_bbox'):
            from proposal_votenet_bbox import ProposalModule
            self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                       mean_size_arr, num_proposal, sampling, config_transformer=config_transformer)
        elif self.sampling in ('no_vote'):
            from proposal_pointnet import ProposalModule
            self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                       mean_size_arr, num_proposal, config_transformer=config_transformer)
        elif self.sampling in ('bbox_directly'):
            from proposal_pointnet_bbox import ProposalModule
            self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                       mean_size_arr, num_proposal, config_transformer=config_transformer)
        else:
            raise NotImplementedError(sampling)


    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        # import ipdb; ipdb.set_trace()
        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        if self.with_voting:
            xyz = end_points['fp2_xyz']
            features = end_points['fp2_features']
            end_points['seed_inds'] = end_points['fp2_inds']
            end_points['seed_xyz'] = xyz
            end_points['seed_features'] = features

            if self.vote_stage >= 1:
                xyz, features = self.vgen1(xyz, features)  # just fc and vote it
                end_points['vote_xyz_stage_1'] = xyz

            end_points['vote_stage'] = self.vote_stage
            end_points['vote_xyz'] = xyz
            end_points['vote_features'] = features

            seed_xyz = end_points['seed_xyz']  # initial
            xyz = end_points['vote_xyz']
        else:
            xyz = end_points['fp2_xyz']
            features = end_points['fp2_features']
            end_points['seed_inds'] = end_points['fp2_inds']
            end_points['seed_xyz'] = xyz
            end_points['seed_features'] = features
            seed_xyz = xyz

        # print(seed_xyz.shape, features.shape, '<<<   detr codes; features dim')
        end_points['point_clouds'] = inputs['point_clouds']  # for pc normalization
        if self.sampling in ('vote_fps', 'seed_fps', 'vote_fps_bbox', 'seed_fps_bbox', 'vote_dist_fps'):
            end_points = self.pnet(seed_xyz, xyz, features, end_points)  # for feature
        elif self.sampling in ('no_vote'):
            end_points = self.pnet(seed_xyz, features, end_points)  # for feature
        elif self.sampling in ('bbox_directly'):
            end_points = self.pnet(seed_xyz, features, end_points)  # for feature
        else:
            raise NotImplementedError(self.sampling)
        del end_points['point_clouds']

        return end_points
