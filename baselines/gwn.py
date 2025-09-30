import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import BaseModel


class GWN(BaseModel):
    '''
    Reference code: https://github.com/nnzhan/Graph-WaveNet
    '''

    # NOTE: we fix the following hyperparameters
    NUM_LAYERS = 2

    # NOTE: we use one common dimension for all num_layers
    def __init__(self, num_blocks, hidden_dim, kernel_size, transition_matrix, dropout, **args):
        super(GWN, self).__init__(**args)

        self.transition_matrix = transition_matrix

        # NOTE: we use only one adjacency matrix
        # self.supports_len = len(supports)

        # NOTE: we do not use learnable adjacency matrix
        # self.adp_adj = adp_adj
        # if adp_adj:
        #     self.nodevec1 = nn.Parameter(torch.randn(self.node_num, 10), requires_grad=True)
        #     self.nodevec2 = nn.Parameter(torch.randn(10, self.node_num), requires_grad=True)
        #     self.supports_len += 1

        self.dropout = dropout
        self.num_blocks = num_blocks

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=(1, 1))

        receptive_field = 1
        for _ in range(num_blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(self.NUM_LAYERS):
                self.filter_convs.append(nn.Conv2d(in_channels=hidden_dim,
                                                   out_channels=hidden_dim,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=hidden_dim,
                                                 out_channels=hidden_dim,
                                                 kernel_size=(1, kernel_size),
                                                 dilation=new_dilation))

                self.skip_convs.append(nn.Conv2d(in_channels=hidden_dim,
                                                 out_channels=hidden_dim,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(hidden_dim))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(GCN(hidden_dim, hidden_dim, self.dropout))
        self.receptive_field = receptive_field

        # NOTE: the purpose of these neural blocks is unclear, so they are removed
        # self.end_conv_1 = nn.Conv2d(in_channels=hidden_dim,
        #                           out_channels=hidden_dim,
        #                           kernel_size=(1, 1),
        #                           bias=True)

        # self.end_conv_2 = nn.Conv2d(in_channels=hidden_dim,
        #                             out_channels=self.output_dim * self.horizon,
        #                             kernel_size=(1, 1),
        #                             bias=True)

    def forward(self, x, label=None):
        # NOTE: x.shape = [batch, time, space, features]

        x = x.transpose(1, 3)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x,(self.receptive_field - in_len, 0, 0, 0))

        # NOTE: we do not use learnable adjacency matrix
        # if self.adp_adj:
        #     adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        #     new_supports = self.supports + [adp]
        # else:
        #     new_supports = self.supports

        x = self.start_conv(x)
        skip = 0

        for i in range(self.num_blocks * self.NUM_LAYERS):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:         
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # x = self.gconv[i](x, new_supports)
            x = self.gconv[i](x, self.transition_matrix)
            
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # NOTE: the next code snippet employs the removed neural blocks
        # x = F.relu(skip)
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)

        x = x.transpose(1, 3)
        return x


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

    def forward(self, x):
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
