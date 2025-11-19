import torch
import torch.nn as nn
from models.pc_encoder.PointNet2.PointNet2_utils import PointNetSetAbstractionMsg


class PointNet2Encoder(nn.Module):
    def __init__(self, args):
        super(PointNet2Encoder, self).__init__()
        # npoint: 采样点的数量
        # radius: 用于球查询的半径列表
        # nsample: 每个半径下采样的邻近点数量
        # in_channel: 输入特征的通道数
        # mlp: 局部PointNet的MLP层定义
        # self.sa1 = PointNetSetAbstractionMsg(10000, [0.05, 0.1], [16, 32], args.input_dim, [[16, 16, 32], [32, 32, 64]])
        # self.sa2 = PointNetSetAbstractionMsg(3000, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        # self.sa3 = PointNetSetAbstractionMsg(1000, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        # self.sa4 = PointNetSetAbstractionMsg(100, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 256], [256, 256, 256]])

        self.sa1 = PointNetSetAbstractionMsg(2048, [0.05, 0.1], [16, 32], args.input_dim, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(100, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 256], [256, 256, 256]])

    def forward(self, pc):
        l0_points = pc
        l0_xyz = pc[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        return l4_points


class MultiScalePointNet2Encoder(nn.Module):
    def __init__(self, args):
        super(MultiScalePointNet2Encoder, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.05, 0.1], [16, 32], args.input_dim, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 256], [256, 256, 256]])

    def forward(self, pc):

        scale_features = []
        l0_points = pc
        l0_xyz = pc[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # 存储转置后的特征 (B, D, N) -> (B, N, D)
        scale_features.append(l1_points.transpose(1, 2).contiguous())

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        scale_features.append(l2_points.transpose(1, 2).contiguous())

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        scale_features.append(l3_points.transpose(1, 2).contiguous())

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        scale_features.append(l4_points.transpose(1, 2).contiguous())

        return scale_features
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # PointNet++ parameters
    parser.add_argument('--input_dim', type=int, default=6)
    args = parser.parse_args()

    model = PointNet2Encoder(args).cuda()
    x = torch.randn(1, 6, 30000).cuda()
    fx = model(x)
    print("fx:", fx.shape)  # (1, 512, 100)
