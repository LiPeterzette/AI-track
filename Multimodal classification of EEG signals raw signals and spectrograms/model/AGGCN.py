import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
import torch.nn.functional as F

class MyGraphConvolution(nn.Module):
    def __init__(self, channels_num=64, graph_convolution_kernel=5, is_channel_attention=True):
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
        D = torch.diag(D)
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
    def __init__(self, cha):
        super(MyChannelAttention, self).__init__()
        self.channel_attention = torch.nn.Sequential(
            torch.nn.Linear(cha, 4),
            torch.nn.Tanh(),
            torch.nn.ReLU(),
            torch.nn.Linear(4, cha),
        )


    def forward(self, inputs):
        inputs = torch.transpose(inputs, 2, 1)
        inputs = inputs.mean(dim=-1)
        inputs = inputs.mean(dim=-1)
        cha_attention = self.channel_attention(inputs)
        cha_attention = torch.mean(cha_attention, dim=0)
        return cha_attention

class latent_correlation_layer(nn.Module):
    def __init__(self, num_features):
        super(latent_correlation_layer, self).__init__()
        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=0.4)
        self.weight_key = nn.Parameter(torch.zeros(size=(num_features, 4)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(num_features, 4)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        # input:batch*64*128
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        d_k = torch.tensor(query.size(-1))
        data = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def forward(self, x):
        input = x.permute(0, 2, 1).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        attention = 0.5 * (attention + attention.T)
        input = input.permute(0, 2, 1).contiguous()
        return attention, input

class AGGCN(nn.Module):
    def __init__(self, num_features=128, leaky_rate=0.2, kernel=5, channel_num=None):
        super(AGGCN, self).__init__()
        self.num_features = num_features
        self.graph_convolution_kernel = kernel
        self.channel_num = channel_num
        self.BN0 = nn.BatchNorm1d(channel_num)
        self.conv1 = MyGraphConvolution(graph_convolution_kernel=self.graph_convolution_kernel,  is_channel_attention=False, channels_num=self.channel_num)
        # self.conv2 = MyGraphConvolution(graph_convolution_kernel=self.graph_convolution_kernel,
        #                                 is_channel_attention=False, channels_num=self.channel_num)
        # self.conv2 = GATConv(8 * 4, 16, heads=1)
        self.relu = nn.LeakyReLU(0.2)
        self.Tconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 8))
        self.BN1 = nn.BatchNorm2d(1)
        self.BN = nn.BatchNorm2d(1)
        self.BN2 = nn.BatchNorm2d(self.graph_convolution_kernel)
        self.fc = torch.nn.Sequential(nn.Dropout(0.1), nn.Linear(self.channel_num*self.graph_convolution_kernel, 8), nn.Tanh(),
                                      nn.Dropout(0.1), nn.Linear(8, 2),  nn.Softmax())
        # self.av = nn.AdaptiveAvgPool2d((64, 1))
        self.av = nn.AvgPool2d((1, num_features))
        self.F = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.latent_correlation_layer1 = latent_correlation_layer(num_features=self.num_features)
        # self.latent_correlation_layer2 = latent_correlation_layer(num_features=self.num_features-7)

    def forward(self, data):
        x = data
        x = torch.transpose(x,2,1)
        # x = torch.unsqueeze(x,dim=1)
        # # x:[batch,64,128]
        # x = self.BN(x)
        # x = self.Tconv(x)
        # x = torch.squeeze(x,dim=1)
        x = self.BN0(x)
        adjacency, _ = self.latent_correlation_layer1(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.BN1(x)
        x, cha_attention = self.conv1(x, adjacency)
        x = self.BN2(x)
        x = torch.mean(x, dim=3)
        x = self.F(x)
        x = self.fc(x)
        return x, adjacency

