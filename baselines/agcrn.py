import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import BaseModel


class AGCRN(BaseModel):
    '''
    Reference code: https://github.com/LeiBAI/AGCRN
    '''
    # NOTE: we fix the following hyperparameters
    EMBEDDING_DIM = 8

    def __init__(self, num_layers, rnn_unit, cheb_k, **args):
        super(AGCRN, self).__init__(**args)
        self.node_embed = nn.Parameter(torch.randn(self.node_num, self.EMBEDDING_DIM), requires_grad=True)
        self.encoder = AVWDCRNN(self.input_dim, rnn_unit, cheb_k, self.EMBEDDING_DIM, num_layers)

        # NOTE: we use more expressive and reasonable custom pooling
        # self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, rnn_unit), bias=True)

    def forward(self, x, label=None):
        # NOTE: x.shape = [batch, time, space, features]

        bs, _, node_num, _ = x.shape
        init_state = self.encoder.init_hidden(bs, node_num)
        x, _ = self.encoder(x, init_state, self.node_embed)
        # x = x[:, -1:, :, :]
        # x = self.end_conv(x)
        return x


class AVWDCRNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_layers):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embed):
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embed)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, node_num):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, node_num))
        return torch.stack(init_states, dim=0)


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embed):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embed))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embed))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embed):
        node_num = node_embed.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embed, node_embed.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embed, self.weights_pool)
        bias = torch.matmul(node_embed, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv
