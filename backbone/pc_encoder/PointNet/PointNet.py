import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batch_size, 1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        iden = iden.to(x.device)  # 将 iden 移动到 x 所在的设备
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, args):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(args.input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = args.feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(100)
        self.fc = nn.Linear(1024, args.d_pcd)

    def forward(self, x):
        B, D, N = x.size()  # (batch_size, 6, N)
        xyz = x[:, :3, :]
        trans = self.stn(xyz)  # (batch_size, 3, 3)

        x = x.transpose(2, 1)  # (batch_size, N, 6)
        xyz = x[:, :, :3]
        feature = x[:, :, 3:]
        xyz = torch.bmm(xyz, trans)  # (batch_size, N, 3)
        x = torch.cat([xyz, feature], dim=2)  # (batch_size, N, 6)

        x = x.transpose(2, 1)  # (batch_size, 6, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, N)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (batch_size, 1024, N)

        x = self.adaptive_pool(x)  # (batch_size, 1024, 100)
        x = x.transpose(2, 1)
        x = self.fc(x)           # (batch_size, 100, 512)

        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        # x = self.fc(x)

        return x


