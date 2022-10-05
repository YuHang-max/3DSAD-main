# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from focal_loss import FocalLoss
from gfocal_loss import QualityFocalLoss
from box_utils_3d import iou_loss, box_c2p
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss, pc_nn_distance

FAR_THRESHOLD = 0.6
FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_loss(end_points, vote_xyz):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
        vote_xyz: vote xyz in cascade vote stage k
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    # vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    if 'no_vote_loss_weight' in end_points.keys():
        seed_xyz = end_points['seed_xyz']
        seed_dist = seed_xyz - vote_xyz
        no_vote_loss_weight = end_points['no_vote_loss_weight']
        seed_dist = torch.mean(huber_loss(seed_dist, delta=1.0), dim=-1)
        novote_mask = -seed_gt_votes_mask+1
        no_vote_loss = torch.sum(seed_dist*novote_mask.float()) / (torch.sum(novote_mask.float())+1e-6)
        # print(no_vote_loss, novote_mask, seed_gt_votes_mask, 'voteweight', flush=True)
        end_points['no_vote_loss'] = no_vote_loss
    return vote_loss

def compute_cascade_vote_loss(end_points):
    vote_stage = end_points['vote_stage']
    assert vote_stage <= 1
    vote_loss = torch.zeros(1).cuda()
    if vote_stage >= 1:  # just 1-stage loss
        end_points['vote_loss_stage_1'] = compute_vote_loss(end_points, end_points['vote_xyz_stage_1'])
        vote_loss += end_points['vote_loss_stage_1']

    return vote_loss

def compute_objectness_loss(end_points, config):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise

    objectness_label_type = 'default'
    if 'objectness_label_config' in end_points.keys():
        objectness_label_config = end_points['objectness_label_config']
        # print('change objectness_label', objectness_label_config, flush=True)
        objectness_label_type = objectness_label_config['name']
        del end_points['objectness_label_config']
    else:
        # use NMS; NEAR_THRESHOLD=FAR_THRESHOLD=0.3; 
        pass
    if objectness_label_type in ['default']:
        dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2
        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_mask = torch.zeros((B,K)).cuda()
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1
        # Compute objectness loss
        # print(OBJECTNESS_CLS_WEIGHTS, '<<<', flush=True)
        objectness_scores = end_points['objectness_scores']
    elif objectness_label_type == 'ATSS':
        box_label_mask = end_points['box_label_mask']
        k = objectness_label_config['k']
        max_dist = objectness_label_config['max_dist']
        nn_dist = pc_nn_distance(aggregated_vote_xyz, gt_center)
        # print(nn_dist.shape, '<< nn_dist shape FFF', flush=True)
        dist_k_min, ind = torch.topk(nn_dist, k, dim=1, largest=False, sorted=False) 
        dist_mean, dist_std = dist_k_min.mean(dim=1), dist_k_min.std(dim=1)
        dist_thres = dist_mean + dist_std
        dist_thres[dist_thres>max_dist] = max_dist
        dist_thres = dist_thres * box_label_mask
        nn_dist_label = (nn_dist <= dist_thres.unsqueeze(1)).float()
        nn_dist_label = nn_dist_label * box_label_mask.unsqueeze(1)
        dist_label = nn_dist_label.sum(-1)
        
        normalized_thres = dist_thres
        normalized_thres[dist_thres<0.1] = 0.1  # norm_min;
        normalized_thres = normalized_thres * box_label_mask
        normalized_dist = nn_dist / (normalized_thres.unsqueeze(1)+1e-6)
        min_norm_dist, ind1 = normalized_dist.min(-1)
        # print(dist_mean.shape, dist_std.shape, dist_thres, '<< dist', flush=True)
        # print(normalized_dist, nn_dist_label, ind1, normalized_thres, '<<<', flush=True)
        # Compute objectness loss
        objectness_scores = end_points['objectness_scores']
        objectness_label = (dist_label >= 1).long()
        # ambiguous = (dist_label >= 2).sum()
        # print(ambiguous, objectness_label.sum(), '<< ambiguous, objectness shape', flush=True)
        objectness_mask = torch.ones((B,K)).cuda()

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1
    if 'objectness_label_loss_type' in end_points.keys():
        obj_loss_type_dict = end_points['objectness_label_loss_type']
        obj_loss_type = obj_loss_type_dict['name']
        del end_points['objectness_label_loss_type']
    else:
        obj_loss_type_dict = {}
        obj_loss_type = 'CrossEntropy'

    # Calculate IoU
    if 'quality_weight' in end_points.keys() or obj_loss_type == 'QualityFocalLoss':
        # Calculate IOU: Just Center & Size
        with torch.no_grad():
            mean_size_arr = config.mean_size_arr
            pred_center = end_points['center']
            gt_center = end_points['center_label'][:,:, 0:3]
            size_class_pred, size_class_gt = end_points['size_scores'], end_points['size_class_label']  # anchor
            size_residual_pred, size_residual_gt = end_points['size_residuals'], end_points['size_residual_label']  # not-normalized
            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).type_as(size_class_pred).unsqueeze(0).repeat(B,1,1)  # B, (num_size_cluster, 3)

            ## # Compute size loss
            gt_pc_center = torch.gather(gt_center, 1, object_assignment[:, :, None].repeat(1, 1, 3))
            gt_class_size = torch.gather(mean_size_arr_expanded, 1, size_class_gt[:, :, None].repeat(1, 1, 3))
            gt_size = gt_class_size + size_residual_gt
            gt_pc_size = torch.gather(gt_size, 1, object_assignment[:, :, None].repeat(1, 1, 3))

            _, pred_size_inds = torch.max(size_class_pred, dim=-1)
            # print(pred_size_inds, size_class_gt, size_class_pred.shape)
            pred_class_size = torch.gather(mean_size_arr_expanded, 1, pred_size_inds[:, :, None].repeat(1, 1, 3))
            pred_residual_size = torch.gather(size_residual_pred, 2, pred_size_inds[:, :, None, None].repeat(1, 1, 1, 3)).squeeze(2)
            pred_size = pred_class_size + pred_residual_size

            pred_size = torch.relu(pred_size).detach()
            bbox_gt = box_c2p(torch.cat([gt_pc_center, gt_pc_size], dim=-1)).view(-1, 6)
            bbox_pred = box_c2p(torch.cat([pred_center, pred_size], dim=-1)).view(-1, 6)
            iou = iou_loss(bbox_gt, bbox_pred, type='iou').reshape(B, -1)
    if 'quality_weight' in end_points.keys():
        quality_pred = end_points['quality_weight']
        quality_pred = quality_pred.sigmoid()
        # print(quality_pred, iou, '<< pred quality value', flush=True)
        end_points['obj_prob_weight'] = quality_pred.detach().cpu().numpy() * 2
        quality_score_loss = torch.abs(quality_pred - iou).mean()
        # print(quality_score_loss, quality_pred, iou, flush=True)
        end_points['quality_score_loss'] = quality_score_loss
    criterion = None
    # print(obj_loss_type, flush=True)
    if obj_loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    elif obj_loss_type == 'FocalLoss':
        criterion = FocalLoss(alpha=obj_loss_type_dict.get('alpha', 0.25), reduction='none')
        B, N = objectness_scores.shape[:2]
        # objectness_loss = criterion(objectness_scores.reshape(B*N, -1), objectness_label.reshape(B*N))
        objectness_loss = criterion(objectness_scores, objectness_label).reshape(B, N)
    elif obj_loss_type == 'QualityFocalLoss':
        criterion = QualityFocalLoss(reduction='none', loss_weight=10)  # it is too small
        B, N = objectness_scores.shape[:2]
        objectness_loss = criterion(objectness_scores.reshape(B*N,-1), objectness_label.reshape(B*N), iou.reshape(B*N))
        objectness_loss = objectness_loss.mean(-1)
        objectness_loss = objectness_loss.reshape(B, N)
        # print(objectness_loss, '<< obj loss shape', flush=True)
        # pass
    else:
        raise NotImplementedError(obj_loss_type)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    # dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    _, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    # USE GT_CENTER TO CALCULATE CENTER
    gt_center_label = torch.gather(gt_center, dim=1, index=object_assignment.unsqueeze(-1).repeat(1, 1, 3))
    dist1 = torch.norm(gt_center_label - pred_center, p=2, dim=2)
    dist2 = torch.sqrt(dist2 + 1e-8)
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    # print(end_points['heading_residuals_normalized'].shape, heading_label_one_hot.shape, heading_residual_normalized_label.shape, '<< heading label normalize shape from loss helper', flush=True)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """
    # Box loss and sem cls loss
    if config.loss_weight is not None:
        # print('using loss weight from config', config.loss_weight, flush=True)
        loss_weight_dict = config.loss_weight.loss_weight.copy()
        if 'objectness_label_assign' in config.loss_weight:
            objectness_label_config = config.loss_weight.objectness_label_assign
            end_points['objectness_label_config'] = objectness_label_config
        if 'objectness_label_weight' in config.loss_weight:
            global OBJECTNESS_CLS_WEIGHTS
            obj_assign_weight = config.loss_weight.objectness_label_weight
            OBJECTNESS_CLS_WEIGHTS = obj_assign_weight
            # print(obj_assign_weight, flush=True)
        if 'objectness_label_loss_type' in config.loss_weight:
            obj_loss_type = config.loss_weight.objectness_label_loss_type
            end_points['objectness_label_loss_type'] = obj_loss_type
    else:
        loss_weight_dict = {
            'center_loss': 1,
            'heading_class_loss': 0.1,
            'heading_residual_loss': 1,
            'size_class_loss': 0.1,
            'size_residual_loss': 1,

            'vote_loss': 1,
            'objectness_loss': 0.5,
            'box_loss': 1,
            'sem_cls_loss': 0.1,

            'all_weight': 10,
        }
        # print('Using Defalut loss_weight', loss_weight_dict, '<< loss_weight dict', flush=True)

    if 'no_vote_loss' in loss_weight_dict.keys():
        end_points['no_vote_loss_weight'] = loss_weight_dict['no_vote_loss']
    # Vote loss
    vote_loss = compute_cascade_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss
    if 'no_vote_loss_weight' in end_points.keys():
        vote_loss = vote_loss + end_points['no_vote_loss'] * end_points['no_vote_loss_weight']
        del end_points['no_vote_loss_weight']
    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points, config)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    # box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    box_loss = center_loss * loss_weight_dict.get('center_loss', 1) \
             + heading_cls_loss * loss_weight_dict.get('heading_class_loss', 0.1)\
             + heading_reg_loss * loss_weight_dict.get('heading_residual_loss', 1) \
             + size_cls_loss * loss_weight_dict.get('size_class_loss', 0.1) \
             + size_reg_loss * loss_weight_dict.get('size_residual_loss', 1)
    end_points['box_loss'] = box_loss

    # Final loss function
    # loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss = vote_loss * loss_weight_dict.get('vote_loss', 1) + \
           objectness_loss * loss_weight_dict.get('objectness_loss', 0.5) + \
           box_loss * loss_weight_dict.get('box_loss', 1) + \
           sem_cls_loss* loss_weight_dict.get('sem_cls_loss', 0.1)

    if 'quality_score_loss' in end_points.keys():
        assert 'quality_score_loss' in loss_weight_dict.keys(), 'loss should have weight'
    if 'quality_score_loss' in loss_weight_dict.keys():
        quality_score_loss = end_points['quality_score_loss']
        loss += quality_score_loss * loss_weight_dict['quality_score_loss']
    loss *= loss_weight_dict.get('all_weight', 10)
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
