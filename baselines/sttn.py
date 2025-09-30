import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import BaseModel


class STTN(BaseModel):
    '''
    Reference code: https://github.com/xumingxingsjtu/STTN
    '''

    # NOTE: we fix the following hyperparameters
    MLP_EXPANSION_FACTOR = 2

    def __init__(self, num_blocks, hidden_dim, transition_matrix, dropout, **args):
        super(STTN, self).__init__(**args)
        self.t_modules = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=(1, 1))

        self.transition_matrix = transition_matrix
        self.num_blocks = num_blocks
        for b in range(num_blocks):
            self.t_modules.append(
                TemporalTransformer(dim=hidden_dim,
                                    depth=1, heads=4,
                                    mlp_dim=hidden_dim * self.MLP_EXPANSION_FACTOR,
                                    time_num=self.seq_len,
                                    dropout=dropout,
                                    window_size=self.seq_len,
                                    ))

            self.s_modules.append(
                SpatialTransformer(dim=hidden_dim,
                                   depth=1, heads=4,
                                   mlp_dim=hidden_dim * self.MLP_EXPANSION_FACTOR,
                                   node_num=self.node_num,
                                   dropout=dropout,
                                   stage=b,
                                   ))

            self.bn.append(nn.BatchNorm2d(hidden_dim))

        # NOTE: the purpose of these neural num_blocks is unclear, so they are removed
        # self.end_conv_1 = nn.Conv2d(in_channels=hidden_dim,
        #                             out_channels=end_channels,
        #                             kernel_size=(1, 1),
        #                             bias=True)

        # self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
        #                             out_channels=self.output_dim * self.horizon,
        #                             kernel_size=(1, 1),
        #                             bias=True)


    def forward(self, x, label=None):
        # NOTE: x.shape = [batch, time, space, features]

        x = x.transpose(1, 3)
        x = self.start_conv(x)
        for i in range(self.num_blocks):
            residual = x
            x = self.s_modules[i](x, self.transition_matrix)
            x = self.t_modules[i](x)
            x = self.bn[i](x) + residual

        # NOTE: the next code snippet employs the removed neural blocks
        # x = x[..., -1:]
        # out = F.relu(self.end_conv_1(x))
        # out = self.end_conv_2(out)

        x = x.transpose(1, 3)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, time_num, dropout, window_size):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, time_num, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  causal=True,
                                  stage=i),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, window_size=1, dropout=0., causal=True, stage=0, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.stage = stage

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size))

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            self.mask = self.mask.to(x.device)
            attn = attn.masked_fill_(self.mask == 0, float("-inf")).softmax(dim=-1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, node_num, dropout, stage=0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, node_num, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim, heads=heads,
                                 dropout=dropout,
                                 stage=stage),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                GCN(dim, dim, dropout),
            ]))

    def forward(self, x, adj):
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)
        x = x + self.pos_embedding
        for attn, ff, gcn in self.layers:
            residual = x.reshape(b, t, n, c)
            x = attn(x, adj) + x
            x = ff(x) + x

            x = gcn(residual.permute(0, 3, 2, 1), adj).permute(
                0, 3, 2, 1).reshape(b*t, n, c) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., stage=0, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.stage = stage 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self,x):
        return self.mlp(x)

    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, transition_matrix):
        out = [x]

        # NOTE: we use only one adjacency matrix
        # for a in support:
        #     x1 = self.nconv(x, a)
        #     out.append(x1)
        #     for k in range(2, self.order + 1):
        #         x2 = self.nconv(x1, a)
        #         out.append(x2)
        #         x1 = x2

        x1 = self.nconv(x, transition_matrix)
        out.append(x1)
        for _ in range(2, self.order + 1):
            x2 = self.nconv(x1, transition_matrix)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
