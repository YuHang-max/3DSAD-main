import torch
import torch.nn as nn
from core.model.task_basemodel.taskmodel.seg_model import seg_module
from core.model.Pointnet.part_module.pointnet_utils import PointNetMSG, PointNetFeature, PointNetPropagation, MLP_List
from core.model.PointnetYanx27 import provider
from core.model.task_error.ShapeNetError import ShapeNetError


class PointnetPlusPartSegv2(seg_module):
    def __init__(self, config):
        self.params = []
        self.task_type = config.get('task_type', 'No Impl')
        if self.task_type == 'ShapeNet':
            self.num_output = 50
            self.num_label = 16  # before fc (later use)
        else:
            raise NotImplementedError('task type %s' % self.task_type)

        normal_channel = config.get('normal_channel', True)
        self.config = config
        super(PointnetPlusPartSegv2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetMSG(512, [0.2], [32], 3 + in_channel + self.num_label, [[64, 64, 128]], [128])  # should input xyz...
        self.sa2 = PointNetMSG(128, [0.4], [64], 128, [[128, 128, 256]], [256])
        self.fc1 = PointNetFeature(256, [256, 512, 1024], [1024])  # in; mlp; fc
        self.fp3 = PointNetPropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetPropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetPropagation(in_channel=128 + 3 + in_channel + self.num_label, mlp=[128, 128])
        self.mlp1 = MLP_List(128, [128, self.num_output], dropout=[0.5],
                             FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set']
        cls_label = input['cls']  # pointnet cls use

        B, N, _ = xyz.shape
        norm = xyz  # channel = 3 + (rgb)
        xyz = xyz[:,:,:3]

        cls_label = nn.functional.one_hot(cls_label, self.num_label).type_as(xyz)
        cls_label_one_hot = cls_label.view(B, 1, self.num_label).repeat(1, N, 1)
        l0_feature = torch.cat([cls_label_one_hot, norm], dim=2)

        l1_xyz, l1_points = self.sa1(xyz, l0_feature)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_feature = self.fc1(l2_xyz, l2_points)
        l3_feature = l3_feature.view(l3_feature.shape[0], 1, l3_feature.shape[1])
        l3_xyz = torch.zeros((l2_xyz.shape[0], 1, 3)).type_as(xyz)
        l2_feature = self.fp3(l2_xyz, l3_xyz, l2_points, l3_feature)  # same as concat
        l1_feature = self.fp2(l1_xyz, l2_xyz, l1_points, l2_feature)
        final_feature = self.fp1(xyz, l1_xyz, l0_feature, l1_feature)  # 个人认为dropout no use
        x = self.mlp1(final_feature.permute(0, 2, 1)).permute(0, 2, 1)
        return {'value': x}

    def _before_forward(self, input):
        # print(input['point_set'].shape, 'before forward; TODO CHECK IT')
        if self.mode == 'train':
            points = input['point_set'].cpu().data.numpy()
            # points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points)
            if input['point_set'].is_cuda:
                points = points.cuda()
            input['point_set'] = points
            # print('before forwrad')
        return input

    def calculate_error(self, input, output):
        output = super(PointnetPlusPartSegv2, self).calculate_error(input, output)
        if self.task_type == 'ShapeNet':
            output = ShapeNetError(input, output)
        return output


if __name__ == "__main__":
    import sys
    import os
    from easydict import EasyDict

    config = {
        'num_output': 83,
        'normal_channel': False
    }
    config = EasyDict(config)
    print(os.getcwd())
    net = PointnetPlusPartSegv2(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
