import torch.nn as nn
from core.model.task_basemodel.taskmodel.cls_model import cls_module
from core.model.Pointnet.part_module.pointnet_utils import PointNetMSG, PointNetFeature


class PointnetPlus(cls_module):
    def __init__(self, config):
        self.params = []
        normal_channel = config.get('normal_channel', True)
        num_output = config.get('num_output', 100)
        self.config = config
        super(PointnetPlus, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetMSG(512, [0.2], [32], in_channel,
                               [[64, 64, 128]],
                               [128])
        self.sa2 = PointNetMSG(128, [0.4], [64], 128,
                               [[128, 128, 256]],
                               [256])
        self.fc1 = PointNetFeature(256, [256, 512, 1024], [512, 256, num_output], [0.4, 0.4])
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set']

        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:]
        else:
            norm = None
        xyz = xyz[:, :, :3]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        x = self.fc1(l2_xyz, l2_points)
        return {'value': x}


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
    net = PointnetPlus(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
