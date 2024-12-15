import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class MyGraphConvolution(nn.Module):
    def __init__(self, channels_num=64, graph_convolution_kernel=5, is_channel_attention=True):
        super(MyGraphConvolution, self).__init__()
        # 导入邻接矩阵
          # solve the adjacency matrix (N*N, eg. 64*64)
        edges = np.load(f'../gnn-eeg-master/utils/edges{channels_num}.npy')
        self.to(device)
        self.adj = None
        self.e_vales = None
        self.e_vets = None
        self.channels_num = channels_num
        self.graph_kernel = None
        self.is_channel_attention = is_channel_attention
        # plt.matshow(self.graph_kernel[0,0])
        self.graph_convolution_kernel = graph_convolution_kernel
        # 添加 注意力 机制
        self.graph_channel_attention = MyChannelAttention(cha=self.channels_num) if is_channel_attention else []
        adjacency = np.zeros((self.channels_num, self.channels_num))
        for x, y in edges:
            adjacency[x][y] = 1
            adjacency[y][x] = 1
        adjacency = np.sign(adjacency + np.eye(self.channels_num))
        # 度矩阵计算
        adjacency = np.sum(adjacency, axis=0) * np.eye(self.channels_num) - adjacency
        # 计算特征值与特征向量
        e_vales, e_vets = np.linalg.eig(adjacency)
        # 计算模型需要的参数
        self.e_vales = torch.tensor(e_vales, dtype=torch.float32)
        self.e_vets = torch.tensor(e_vets, dtype=torch.float32)
        # 计算 图卷积 的卷积核
        graph_kernel = torch.nn.Parameter(torch.FloatTensor(self.graph_convolution_kernel, 1, self.channels_num), requires_grad=True)
        nn.init.xavier_normal(graph_kernel, gain=1)
        # graph_kernel.data.fill_(1)
        self.graph_kernel = graph_kernel * torch.eye(self.channels_num)
        self.graph_kernel = torch.matmul(torch.matmul(self.e_vets, self.graph_kernel), torch.transpose(torch.tensor(self.e_vets), 1, 0))
        self.graph_kernel = torch.unsqueeze(self.graph_kernel, 0)
    def forward(self, x):
    #adj为卷积核
        adj = self.graph_kernel.to(device)
    # 通道注意力网络
        if self.is_channel_attention:
            cha_attention = self.graph_channel_attention(x)
            adj = cha_attention * adj
            cha = cha_attention
        else:
            cha = torch.ones(64)
    # 卷积过程
        x = torch.matmul(adj, x)
        x = torch.nn.ReLU()(x)
        return x, cha

    # @staticmethod
    # def compute_output_shape(input_shape):
    #     return input_shape

class MyChannelAttention(nn.Module):
    def __init__(self, cha):
        super(MyChannelAttention, self).__init__()
        self.channel_attention = torch.nn.Sequential(
            torch.nn.Linear(cha, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, cha),
        )
        self.to(device)


    def forward(self, inputs):
        inputs = torch.transpose(inputs, 2, 1)
        inputs = inputs.mean(dim=-1)
        inputs = inputs.mean(dim=-1)
        cha_attention = self.channel_attention(inputs)
        cha_attention = torch.mean(cha_attention, dim=0)
        return cha_attention


class TGCN(nn.Module):
    def __init__(self, num_features=128, kernel=5, channel_num=64):
        super(TGCN, self).__init__()
        self.num_features = num_features
        self.graph_convolution_kernel = kernel
        self.channel_num = channel_num
        self.conv1 = MyGraphConvolution(graph_convolution_kernel=self.graph_convolution_kernel, is_channel_attention=False, channels_num=channel_num)
        # self.conv2 = GATConv(8 * 4, 16, heads=1)
        self.BN1 = nn.BatchNorm2d(1)
        self.BN2 = nn.BatchNorm2d(self.graph_convolution_kernel)
        # self.fc = torch.nn.Linear(2 * 16, 1)
        self.fc = torch.nn.Sequential(nn.Dropout(0.3), nn.Linear(self.channel_num*self.graph_convolution_kernel, 8), nn.Tanh(),
                                      nn.Dropout(0.3), nn.Linear(8, 2),  nn.Softmax())
        self.av = nn.AvgPool2d((1, num_features))
        self.F = nn.Flatten()
        self.to(device)
        self.BN0 = nn.BatchNorm1d(channel_num)
        self.Tconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 8))
        # self.BN1 = nn.BatchNorm2d(1)
        # self.BN = nn.BatchNorm2d(1)
        # self.BN2 = nn.BatchNorm2d(self.graph_convolution_kernel)
    def forward(self, data):
        x = data
        x = torch.transpose(x, 2, 1)
        x = torch.unsqueeze(x, dim=1)
        x = self.BN1(x)
        x = self.Tconv(x)
        x, cha = self.conv1(x)
        x = self.BN2(x)
        x = torch.mean(x, dim=3)
        x = self.F(x)
        x = self.fc(x)
        return x, cha