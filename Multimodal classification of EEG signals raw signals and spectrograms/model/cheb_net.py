import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
import torch.nn.functional as F
import torch_geometric as tfg
from torch_geometric.nn import GCNConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool, global_max_pool
from model.sag_pool_ import SelfAttentionPooling2, SelfAttentionPooling1, SelfAttentionPooling3
import scipy.sparse as sp


class MyGraphConvolution(nn.Module):
    def __init__(self, channels_num=64, graph_convolution_kernel=7, is_channel_attention=False):
        super(MyGraphConvolution, self).__init__()
        self.adj = None
        self.e_vales = None
        self.e_vets = None
        self.channels_num = channels_num
        self.graph_kernel = None
        self.is_channel_attention = is_channel_attention
        # plt.matshow(self.graph_kernel[0,0])
        self.graph_convolution_kernel = graph_convolution_kernel
        # # 添加 注意力 机制
        self.graph_channel_attention = MyChannelAttention(cha=self.channels_num) if is_channel_attention else []
        # self.a = torch.FloatTensor(self.graph_convolution_kernel, 1, self.channels_num)
        # self.graph_kernel = torch.nn.Parameter(self.a, requires_grad=True)
        self.graph_kernel = torch.nn.Parameter(torch.empty(self.graph_convolution_kernel, 1, self.channels_num), requires_grad=True)
        nn.init.xavier_normal(self.graph_kernel, gain=1)

    def kernel(self, adjacency):
        D = torch.sum(adjacency, dim=0)
        adjacency = D - adjacency
        # 计算特征值与特征向量
        e_vales, e_vets = torch.linalg.eig(adjacency)
        # 计算模型需要的参数
        e_vales = torch.tensor(e_vales, dtype=torch.float32)
        e_vets = torch.tensor(e_vets, dtype=torch.float32)
        # 计算 图卷积 的卷积核
        I = torch.eye(self.channels_num).to(device)
        graph_kernel = self.graph_kernel * I
        graph_kernel = torch.matmul(torch.matmul(e_vets, graph_kernel), torch.transpose(torch.tensor(e_vets),
                                                                                             1, 0))
        graph_kernel = torch.unsqueeze(graph_kernel, 0)
        return graph_kernel


    def forward(self, x, adjacency):
        #adj为卷积核
        adj = self.kernel(adjacency)
        x = torch.matmul(adj, x)
        x = torch.nn.ReLU()(x)
        return x, []

class MyChannelAttention(nn.Module):
    def __init__(self, cha=64):
        super(MyChannelAttention, self).__init__()
        self.channel_num = cha
        self.channel_attention = torch.nn.Sequential(
            torch.nn.Linear(self.channel_num, 4),
            torch.nn.Tanh(),
            torch.nn.ReLU(),
            torch.nn.Linear(4, self.channel_num),
        )


    def forward(self, inputs):
        inputs = inputs.mean(dim=-1)
        # inputs = inputs.mean(dim=-1)
        cha_attention = self.channel_attention(inputs)
        cha_attention = torch.mean(cha_attention, dim=0)
        return cha_attention


class chebnet(nn.Module):
    def __init__(self, num_features=128, leaky_rate=0.2, channel_num=64, kernel=7):
        super(chebnet, self).__init__()
        self.alpha = leaky_rate
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.num_features = num_features
        self.graph_convolution_kernel = kernel
        self.channel = channel_num
        self.ratio = 0
        self.is_channel_attention = True
        self.conv2 = MyGraphConvolution(graph_convolution_kernel=self.graph_convolution_kernel, channels_num=self.channel)
        self.weight_key = nn.Parameter(torch.zeros(size=(self.num_features-3, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.num_features-3, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=(1,4))
        self.BN1 = nn.BatchNorm2d(1)
        self.BN2 = nn.BatchNorm2d(self.graph_convolution_kernel)
        self.BN0 = nn.BatchNorm1d(channel_num)
        self.BN3 = nn.BatchNorm1d(num_features)
        self.fc = torch.nn.Sequential(nn.Dropout(0.1), nn.Linear(self.channel*self.graph_convolution_kernel, 8), nn.Tanh(),
                                      nn.Dropout(0.1), nn.Linear(8, 2),  nn.Softmax())

        # self.av = nn.AdaptiveAvgPool2d((64, 1))
        # self.av = nn.AvgPool2d((1, self.num_features))

        self.pool = nn.AvgPool1d(2)
        self.F = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.pool1 = SelfAttentionPooling3(channels_num=self.channel, keep_ratio=self.ratio, num_features=self.num_features)
        self.pool2 = SelfAttentionPooling2(channels_num=self.channel, keep_ratio=self.ratio, num_features=self.num_features)
    def latent_correlation_layer(self, x):
        # input:batch*64*128
        input = x.permute(0, 2, 1).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        adj = 0.5 * (attention + attention.T)
        return adj

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        d_k = torch.tensor(query.size(-1))
        data = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    # def forward(self, data):
    #     x, edge_index, batch, batchsize = data.x, data.edge_index, data.batch, len(data.y)
    #     x: [batch*64, 128]
    #     # x = self.MyChannelAttention(x)
    #     x = torch.reshape(x, (batchsize, self.channel, self.num_features))
    #     x = self.BN0(x)
    #     x = self.pool1(x)
    #     adjacency = self.latent_correlation_layer(x)
    #     x = torch.unsqueeze(x, dim=1)
    #     x = self.BN1(x)
    #     x, _ = self.conv2(x, adjacency)
    #     x = self.BN2(x)
    #     # x = self.SAGPool(x)
    #     x2 = torch.mean(x, dim=3)
    #     out = self.F(x2)
    #     out = self.fc(out)
    #     return out,[]

    def forward(self, data):
        x, edge_index, batch, batchsize = data.x, data.edge_index, data.batch, len(data.y)
        x: [batch*64, 128]
        # x = self.MyChannelAttention(x)

        x = torch.reshape(x, (batchsize, self.channel, self.num_features))
        x = self.BN0(x)
        x = torch.unsqueeze(x,dim=1)
        x = self.conv1(x)
        # x, att = self.pool1(x)
        # x = self.pool2(x)
        x = torch.squeeze(x)
        adjacency = self.latent_correlation_layer(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.BN1(x)
        x, _ = self.conv2(x, adjacency)
        x = self.BN2(x)
        # x = self.SAGPool(x)
        x2 = torch.mean(x, dim=3)
        out = self.F(x2)
        out = self.fc(out)
        return out, adjacency