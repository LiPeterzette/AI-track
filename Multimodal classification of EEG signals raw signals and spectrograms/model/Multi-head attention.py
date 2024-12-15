import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_feature, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_feature = d_feature
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(p=dropout)
        self.q_linear = nn.Linear(d_feature, n_head * d_k)
        self.v_linear = nn.Linear(d_feature, n_head * d_v)
        self.k_linear = nn.Linear(d_feature, n_head * d_k)
        self.out = nn.Linear(n_head * d_v, d_feature)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # Linear projections
        k = self.k_linear(k).view(bs, -1, self.n_head, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_head, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_head, self.d_v)

        # Transpose to (batch, n_head, feature_length, d_k or d_v)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = torch.softmax(scores, dim=-2)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)

        # Transpose to (batch, feature_length, n_head, d_v) and concat
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_head * self.d_v)

        # Linear projection
        output = self.out(output)

        return output



class MultiHeadModel(nn.Module):
    def __init__(self):
        super(MultiHeadModel, self).__init__()
        self.attention = MultiHeadAttention(n_head=8, d_feature=16, d_k=8, d_v=8)
        self.fc = nn.Linear(16, 9)

    def forward(self, x):
        # x shape: (batch, feature, channel)
        x = self.attention(x, x, x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
