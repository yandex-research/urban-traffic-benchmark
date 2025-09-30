from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F
import tsl.nn as stnn
import dgl

from dgl import ops
from plr_embeddings import PLREmbeddings
from utils import _check_dim_and_num_heads_consistency

from baselines import (
    AGCRN, ASTGCN, GWN, STGCN, STGODE, STTN,
    calculate_scaled_laplacian_matrix,
    calculate_transition_matrix,
    calculate_cheb_polynomials,
)
from torch_geometric.utils import to_dense_adj


class ResidualModulesWrapper(nn.Module):
    def __init__(self, modules):
        super().__init__()

        if isinstance(modules, nn.Module):
            modules = [modules]

        for module in modules:
            module.takes_graph_as_input = ('graph' in signature(module.forward).parameters)

        self.wrapped_modules = nn.ModuleList(modules)

    def forward(self, graph, x):
        x_res = x
        for module in self.wrapped_modules:
            if module.takes_graph_as_input:
                x_res = module(graph, x_res)
            else:
                x_res = module(x_res)

        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, num_inputs=1, dropout=0):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=dim * num_inputs, out_features=dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GraphMeanAggregationModule(nn.Module):
    def __init__(self, sep=True, **kwargs):
        super().__init__()
        self.sep = sep

    def forward(self, graph, x):
        x_aggregated = [x] if self.sep else []
        for cur_edge_type in graph.etypes:
            cur_graph = dgl.edge_type_subgraph(graph, [cur_edge_type])
            cur_x_aggregated = ops.copy_u_mean(cur_graph, x)
            x_aggregated.append(cur_x_aggregated)

        x_aggregated = torch.cat(x_aggregated, axis=-1)

        return x_aggregated


class GraphMaxAggregationModule(nn.Module):
    def __init__(self, sep=True, **kwargs):
        super().__init__()
        self.sep = sep

    def forward(self, graph, x):
        x_aggregated = [x] if self.sep else []
        for cur_edge_type in graph.etypes:
            cur_graph = dgl.edge_type_subgraph(graph, [cur_edge_type])
            cur_x_aggregated = ops.copy_u_max(cur_graph, x)
            x_aggregated.append(cur_x_aggregated)

        x_aggregated = torch.cat(x_aggregated, axis=-1)
        x_aggregated[x_aggregated.isinf()] = 0

        return x_aggregated


class GraphGCNAggregationModule(nn.Module):
    def __init__(self, sep=True, **kwargs):
        super().__init__()
        self.sep = sep

    def forward(self, graph, x):
        x_aggregated = [x] if self.sep else []
        for cur_edge_type in graph.etypes:
            cur_graph = dgl.edge_type_subgraph(graph, [cur_edge_type])

            cur_in_degrees = cur_graph.in_degrees().float()
            cur_out_degrees = cur_graph.out_degrees().float()
            cur_degree_edge_products = ops.u_mul_v(cur_graph, cur_out_degrees, cur_in_degrees)
            cur_degree_edge_products[cur_degree_edge_products == 0] = 1
            cur_norm_coefs = 1 / cur_degree_edge_products.sqrt()

            cur_x_aggregated = ops.u_mul_e_sum(cur_graph, x, cur_norm_coefs)
            x_aggregated.append(cur_x_aggregated)

        x_aggregated = torch.cat(x_aggregated, axis=-1)

        return x_aggregated


class GraphAttnGATAggregationModule(nn.Module):
    def __init__(self, dim, num_heads, num_edge_types=1, sep=True, **kwargs):
        super().__init__()
        self.sep = sep

        _check_dim_and_num_heads_consistency(dim=dim, num_heads=num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_edge_types = num_edge_types

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads * num_edge_types)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads * num_edge_types, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph, x):
        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)

        attn_scores_u = attn_scores_u.split(split_size=[self.num_heads for _ in range(self.num_edge_types)], dim=-1)
        attn_scores_v = attn_scores_v.split(split_size=[self.num_heads for _ in range(self.num_edge_types)], dim=-1)

        x_aggregated = [x] if self.sep else []
        x = x.reshape(-1, self.head_dim, self.num_heads)
        for edge_type, cur_attn_scores_u, cur_attn_scores_v in zip(graph.etypes, attn_scores_u, attn_scores_v):
            cur_graph = dgl.edge_type_subgraph(graph, [edge_type])

            cur_attn_scores = ops.u_add_v(cur_graph, cur_attn_scores_u, cur_attn_scores_v)
            cur_attn_scores = self.attn_act(cur_attn_scores)
            cur_attn_probs = ops.edge_softmax(cur_graph, cur_attn_scores)

            cur_x_aggregated = ops.u_mul_e_sum(cur_graph, x, cur_attn_probs)
            cur_x_aggregated = cur_x_aggregated.reshape(-1, self.dim)

            x_aggregated.append(cur_x_aggregated)

        x_aggregated = torch.cat(x_aggregated, axis=-1)

        return x_aggregated


class GraphAttnTrfAggregationModule(nn.Module):
    def __init__(self, dim, num_heads, num_edge_types=1, dropout=0, sep=True, **kwargs):
        super().__init__()
        self.sep = sep

        _check_dim_and_num_heads_consistency(dim=dim, num_heads=num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_edge_types = num_edge_types
        self.attn_scores_multiplier = 1 / torch.tensor(self.head_dim).sqrt()

        self.attn_qkv_linear = nn.Linear(in_features=dim, out_features=dim * num_edge_types * 3)

        self.output_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=dim, out_features=dim) for _ in range(num_edge_types)]
        )
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_edge_types)])

    def forward(self, graph, x):
        qkvs = self.attn_qkv_linear(x)
        qkvs = qkvs.reshape(-1, self.num_heads * self.num_edge_types, self.head_dim * 3)
        queries, keys, values = qkvs.split(split_size=(self.head_dim, self.head_dim, self.head_dim), dim=-1)

        queries = queries.split(split_size=[self.num_heads for _ in range(self.num_edge_types)], dim=-2)
        keys = keys.split(split_size=[self.num_heads for _ in range(self.num_edge_types)], dim=-2)
        values = values.split(split_size=[self.num_heads for _ in range(self.num_edge_types)], dim=-2)

        x_aggregated = [x] if self.sep else []
        for edge_type, cur_queries, cur_keys, cur_values, output_linear, dropout in zip(
                graph.etypes, queries, keys, values, self.output_linear_layers, self.dropout_layers
        ):
            cur_graph = dgl.edge_type_subgraph(graph, [edge_type])

            cur_attn_scores = ops.u_dot_v(cur_graph, cur_keys, cur_queries) * self.attn_scores_multiplier
            cur_attn_probs = ops.edge_softmax(cur_graph, cur_attn_scores)

            cur_x_aggregated = ops.u_mul_e_sum(cur_graph, cur_values, cur_attn_probs)
            cur_x_aggregated = cur_x_aggregated.reshape(-1, self.dim)

            cur_x_aggregated = output_linear(cur_x_aggregated)
            cur_x_aggregated = dropout(cur_x_aggregated)

            x_aggregated.append(cur_x_aggregated)

        x_aggregated = torch.cat(x_aggregated, axis=-1)

        return x_aggregated


class RNNSequenceEncoderModule(nn.Module):
    rnn_types = {
        'LSTM': nn.LSTM,
        'GRU': nn.GRU
    }

    def __init__(self, rnn_type_name, num_layers, dim, dropout=0, **kwargs):
        super().__init__()

        RNN = self.rnn_types[rnn_type_name]
        self.rnn = RNN(input_size=dim, hidden_size=dim, num_layers=num_layers, dropout=dropout,
                       batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)

        return x


class CausalTCNBlock(nn.Module):
    def __init__(self, dim, kernel_size, dilation=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                padding=(kernel_size - 1) * dilation, dilation=dilation)
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]
        x = self.dropout(x)
        x = self.activation(x)
        return x


class TCNSequenceEncoderModule(nn.Module):
    '''See here for details: https://discuss.pytorch.org/t/causal-convolution/3456.'''
    
    def __init__(self, num_layers, dim, kernel_size, dilation=1, dropout=0.0, **kwargs):
        super().__init__()

        self.blocks = nn.Sequential(*[
            CausalTCNBlock(dim=dim, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        for block in self.blocks:
            x = block(x)
        
        return x.permute(0, 2, 1)


class TransformerSequenceEncoderModule(nn.Module):
    def __init__(self, num_layers, dim, num_heads, seq_len, bidir_attn=False, dropout=0, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim=dim, num_heads=num_heads)

        self.bidir_attn = bidir_attn
        self.attn_mask = None if bidir_attn else nn.Transformer.generate_square_subsequent_mask(seq_len)

        self.positional_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=dim)

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim, dropout=dropout,
                                       activation='gelu', norm_first=True, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        pos_embs = self.positional_embeddings.weight[None, ...]
        x = x + pos_embs

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, src_mask=self.attn_mask, is_causal=not self.bidir_attn)

        return x


class FeaturesPreparatorForDeepModels(nn.Module):
    def __init__(self, features_dim, use_learnable_node_embeddings, num_nodes, learnable_node_embeddings_dim,
                 initialize_learnable_node_embeddings_with_deepwalk, deepwalk_node_embeddings,
                 use_plr_for_numerical_features, numerical_features_mask, plr_numerical_features_frequencies_dim,
                 plr_numerical_features_frequencies_scale, plr_numerical_features_embedding_dim,
                 plr_numerical_features_shared_linear, plr_numerical_features_shared_frequencies,
                 use_plr_for_past_targets, past_targets_mask, plr_past_targets_frequencies_dim,
                 plr_past_targets_frequencies_scale, plr_past_targets_embedding_dim,
                 plr_past_targets_shared_linear, plr_past_targets_shared_frequencies):
        super().__init__()

        output_dim = features_dim

        self.use_learnable_node_embeddings = use_learnable_node_embeddings
        if use_learnable_node_embeddings:
            output_dim += learnable_node_embeddings_dim
            if initialize_learnable_node_embeddings_with_deepwalk:
                if learnable_node_embeddings_dim != deepwalk_node_embeddings.shape[1]:
                    raise ValueError(f'initialize_learnable_node_embeddings_with_deepwalk argument is True, but the '
                                     f'value of learnable_node_embeddings_dim argument does not match the dimension of '
                                     f'the precomputed DeepWalk node embeddings: '
                                     f'{learnable_node_embeddings_dim} != {deepwalk_node_embeddings.shape[1]}.')

                self.node_embeddings = nn.Embedding(num_embeddings=num_nodes,
                                                    embedding_dim=learnable_node_embeddings_dim,
                                                    _weight=deepwalk_node_embeddings)
            else:
                self.node_embeddings = nn.Embedding(num_embeddings=num_nodes,
                                                    embedding_dim=learnable_node_embeddings_dim)

        self.use_plr_for_numerical_features = use_plr_for_numerical_features
        if use_plr_for_numerical_features:
            numerical_features_dim = numerical_features_mask.sum()
            output_dim = output_dim - numerical_features_dim + \
                         (numerical_features_dim * plr_numerical_features_embedding_dim)
            self.register_buffer('numerical_features_mask', numerical_features_mask)
            self.plr_embeddings_numerical_features = PLREmbeddings(
                features_dim=numerical_features_dim,
                frequencies_dim=plr_numerical_features_frequencies_dim,
                frequencies_scale=plr_numerical_features_frequencies_scale,
                embedding_dim=plr_numerical_features_embedding_dim,
                shared_linear=plr_numerical_features_shared_linear,
                shared_frequencies=plr_numerical_features_shared_frequencies
            )

        self.use_plr_for_past_targtes = use_plr_for_past_targets
        if use_plr_for_past_targets:
            past_targets_dim = past_targets_mask.sum()
            output_dim = output_dim - past_targets_dim + past_targets_dim * plr_past_targets_embedding_dim
            self.register_buffer('past_targets_mask', past_targets_mask)
            self.plr_embeddings_past_targets = PLREmbeddings(
                features_dim=past_targets_dim,
                frequencies_dim=plr_past_targets_frequencies_dim,
                frequencies_scale=plr_past_targets_frequencies_scale,
                embedding_dim=plr_past_targets_embedding_dim,
                shared_linear=plr_past_targets_shared_linear,
                shared_frequencies=plr_past_targets_shared_frequencies
            )

        self.output_dim = output_dim

    def forward(self, x):
        if self.use_plr_for_numerical_features:
            x_numerical = x[..., self.numerical_features_mask]
            x_numerical_embedded = self.plr_embeddings_numerical_features(x_numerical).flatten(start_dim=-2)

        if self.use_plr_for_past_targtes:
            x_targets = x[..., self.past_targets_mask]
            x_targets_embedded = self.plr_embeddings_past_targets(x_targets).flatten(start_dim=-2)

        if self.use_plr_for_numerical_features or self.use_plr_for_past_targtes:
            if not self.use_plr_for_numerical_features:
                x = [x_targets_embedded, x[..., ~self.past_targets_mask]]
            elif not self.use_plr_for_past_targtes:
                x = [x_numerical_embedded, x[..., ~self.numerical_features_mask]]
            else:
                x = [
                    x_targets_embedded,
                    x_numerical_embedded,
                    x[..., ~(self.past_targets_mask | self.numerical_features_mask)]
                ]

            x = torch.cat(x, axis=-1)

        if self.use_learnable_node_embeddings:
            graph_batch_size = x.shape[0] // self.node_embeddings.weight.shape[0]
            node_embs = self.node_embeddings.weight.repeat(graph_batch_size, 1)

            if x.dim() == 3:
                # SequenceInputModel is used, so sequence dimension needs to be added.
                seq_len = x.shape[1]
                node_embs = node_embs.unsqueeze(1).expand(-1, seq_len, -1)

            x = torch.cat([x, node_embs], axis=-1)

        return x


NEIGHBORHOOD_AGGREGATION_MODULES = {
    'MeanAggr': GraphMeanAggregationModule,
    'MaxAggr': GraphMaxAggregationModule,
    'GCNAggr': GraphGCNAggregationModule,
    'AttnGATAggr': GraphAttnGATAggregationModule,
    'AttnTrfAggr': GraphAttnTrfAggregationModule
}

SEQUENCE_ENCODER_MODULES = {
    'RNN': RNNSequenceEncoderModule,
    'TCN': TCNSequenceEncoderModule,
    'Transformer': TransformerSequenceEncoderModule
}

NORMALIZATION_MODULES = {
    'none': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


### torch spatiotemporal baselines


class DCRNNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim, normalization_name, **kwargs):
        super().__init__()
        self.encoder = stnn.encoders.DCRNN(input_size=hidden_dim,
                                           hidden_size=hidden_dim,
                                           n_layers=num_spatiotemporal_blocks)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)
    
    def forward(self, x, edge_index):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x, _ = self.encoder(x, edge_index, None)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class EGCNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim, normalization_name, **kwargs):
        super().__init__()
        self.encoder = stnn.encoders.EvolveGCN(input_size=hidden_dim,
                                               hidden_size=hidden_dim,
                                               n_layers=num_spatiotemporal_blocks,
                                               norm='mean')  # replaced 'gcn' with 'mean' to correspond to default in tsl

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x, edge_index):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.encoder(x, edge_index, None)
        x = x.squeeze()

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class GWNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim,
                 normalization_name, temporal_kernel_size, temporal_dilation,
                 spatial_kernel_size, dropout, **kwargs):
        super().__init__()

        self.temporal_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        self.skip_connection_blocks = nn.ModuleList()

        dilation_mod = 2
        receptive_field = 1

        for block_idx in range(num_spatiotemporal_blocks):
            d = temporal_dilation ** (block_idx % dilation_mod)
            temporal_block = stnn.blocks.encoders.TemporalConvNet(input_channels=hidden_dim,
                                                                  hidden_channels=hidden_dim,
                                                                  kernel_size=temporal_kernel_size,
                                                                  dropout=dropout,
                                                                  dilation=d,
                                                                  exponential_dilation=False,
                                                                  n_layers=1,
                                                                  causal_padding=False,
                                                                  gated=True)

            spatial_block = stnn.layers.graph_convs.DiffConv(in_channels=hidden_dim,
                                                             out_channels=hidden_dim,
                                                             k=spatial_kernel_size)

            skip_connection_block = nn.Linear(hidden_dim, hidden_dim)

            self.temporal_blocks.append(temporal_block)
            self.spatial_blocks.append(spatial_block)
            self.skip_connection_blocks.append(skip_connection_block)

            receptive_field += d * (temporal_kernel_size - 1)

        self.dropout = nn.Dropout(p=dropout)
        self.receptive_field = receptive_field
        self.num_blocks = num_spatiotemporal_blocks

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)

    def forward(self, x, edge_index):
        x = x.permute(1, 0, 2).unsqueeze(0)

        if self.receptive_field > x.shape[1]:
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.shape[1], 0))

        out = torch.zeros(1, x.shape[1], 1, 1, device=x.device)

        for i in range(self.num_blocks):
            res = x
            x = self.temporal_blocks[i](x)
            # NOTE: skip connection is originally applied here
            # out = self.skip_connection_blocks[i](x) + out[:, -x.shape[1]:]
            x = self.spatial_blocks[i](x, edge_index)
            x = self.dropout(x)
            x = x + res[:, -x.shape[1]:]
            out = self.skip_connection_blocks[i](x) + out[:, -x.shape[1]:]

        x = out
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class GGNAdapter(nn.Module):
    def __init__(self, num_temporal_blocks, num_spatial_blocks,
                 hidden_dim, output_dim, normalization_name, seq_length, **kwargs):
        super().__init__()

        self.input_encoder = nn.Linear(hidden_dim * seq_length, hidden_dim)

        self.temporal_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()

        for _ in range(num_temporal_blocks):
            temporal_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.temporal_blocks.append(temporal_block)
        
        for _ in range(num_spatial_blocks):
            spatial_block = stnn.layers.graph_convs.GatedGraphNetwork(hidden_dim, hidden_dim)
            self.spatial_blocks.append(spatial_block)

        self.num_temporal_blocks = num_temporal_blocks
        self.num_spatial_blocks = num_spatial_blocks

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = x.unsqueeze(0)

        x = torch.flatten(x, start_dim=2)
        x = self.input_encoder(x)

        for temporal_block in self.temporal_blocks:
            x = temporal_block(x)

        for spatial_block in self.spatial_blocks:
            x = spatial_block(x, edge_index)

        x = x.squeeze()

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class GRUGCNAdapter(nn.Module):
    def __init__(self, num_temporal_blocks, num_spatial_blocks,
                 hidden_dim, output_dim, normalization_name, **kwargs):
        super().__init__()

        self.input_encoder = stnn.blocks.encoders.RNN(input_size=hidden_dim,
                                                      hidden_size=hidden_dim,
                                                      n_layers=num_temporal_blocks,
                                                      return_only_last_state=True,
                                                      cell='gru')

        self.spatial_blocks = nn.ModuleList()
        for _ in range(num_spatial_blocks):
            spatial_block = stnn.layers.graph_convs.GraphConv(input_size=hidden_dim,
                                                              output_size=hidden_dim,
                                                              root_weight=False,
                                                              activation='relu')
            
            self.spatial_blocks.append(spatial_block)

        self.skip_connection_block = nn.Linear(hidden_dim, hidden_dim)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = x.permute(1, 0, 2).unsqueeze(0)

        x = self.input_encoder(x)
        out = x

        for layer in self.spatial_blocks:
            out = layer(out, edge_index)

        x = out + self.skip_connection_block(x)
        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        x = x.squeeze()

        return x


### large-st repository baselines


class AGCRNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim, normalization_name,
                 spatial_kernel_size, seq_length, num_nodes_batched, **kwargs):
        super().__init__()

        # NOTE: `num_nodes_batched` must be the number of nodes in batched graph, not the original one
        # NOTE: `input_dim` and `output_dim` are probably not used and required just for consistency

        self.backbone = AGCRN(node_num=num_nodes_batched,
                              input_dim=hidden_dim,
                              output_dim=1,
                              horizon=output_dim,
                              seq_len=seq_length,
                              num_layers=num_spatiotemporal_blocks,
                              rnn_unit=hidden_dim,
                              cheb_k=spatial_kernel_size)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)


        self.init_backbone()

    def init_backbone(self):
        for parameter in self.backbone.parameters():
            nn.init.uniform_(parameter)

    def forward(self, x, *args):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.backbone(x)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class ASTGCNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim, normalization_name,
                 seq_length, num_nodes_batched, edge_index_batched, **kwargs):
        super().__init__()

        adjacency_matrix = to_dense_adj(edge_index_batched, max_num_nodes=num_nodes_batched).cpu().numpy()[0]
        laplacian_matrix = calculate_scaled_laplacian_matrix(adjacency_matrix)
        cheb_polynomials = calculate_cheb_polynomials(laplacian_matrix, order=3)

        cheb_polynomials = [torch.FloatTensor(matrix) for matrix in cheb_polynomials]

        # NOTE: `num_nodes_batched` must be the number of nodes in batched graph, not the original one
        # NOTE: `edge_index_batched` of train batched graph is passed, so train and eval batch sizes must be equal
        # NOTE: `input_dim` and `output_dim` are probably not used and required just for consistency

        self.backbone = ASTGCN(node_num=num_nodes_batched,
                               input_dim=hidden_dim,
                               output_dim=1,
                               horizon=output_dim,
                               seq_len=seq_length,
                               cheb_polynomials=cheb_polynomials,
                               hidden_dim=hidden_dim,
                               num_blocks=num_spatiotemporal_blocks)

        self.init_backbone()

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)

    def init_backbone(self):
        for parameter in self.backbone.parameters():
            nn.init.uniform_(parameter)
    
    def forward(self, x, *args):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.backbone(x)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class GWNv2Adapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim,
                 normalization_name, temporal_kernel_size, seq_length,
                 num_nodes_batched, edge_index_batched, dropout, **kwargs):
        super().__init__()

        adjacency_matrix = to_dense_adj(edge_index_batched, max_num_nodes=num_nodes_batched).cpu().numpy()[0]
        transition_matrix = calculate_transition_matrix(adjacency_matrix)
        transition_matrix = torch.FloatTensor(transition_matrix).to(edge_index_batched.device)

        # NOTE: `num_nodes_batched` must be the number of nodes in batched graph, not the original one
        # NOTE: `edge_index_batched` of train batched graph is passed, so train and eval batch sizes must be equal
        # NOTE: `input_dim` and `output_dim` are probably not used and required just for consistency

        self.backbone = GWN(node_num=num_nodes_batched,
                            input_dim=hidden_dim,
                            output_dim=1,
                            horizon=output_dim,
                            seq_len=seq_length,
                            num_blocks=num_spatiotemporal_blocks,
                            hidden_dim=hidden_dim,
                            kernel_size=temporal_kernel_size,
                            transition_matrix=transition_matrix,
                            dropout=dropout)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)
    
    def forward(self, x, *args):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.backbone(x)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class STGCNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim,
                 normalization_name, temporal_kernel_size, spatial_kernel_size,
                 seq_length, num_nodes_batched, edge_index_batched, dropout, **kwargs):
        super().__init__()

        adjacency_matrix = to_dense_adj(edge_index_batched, max_num_nodes=num_nodes_batched).cpu().numpy()[0]
        laplacian_matrix = calculate_scaled_laplacian_matrix(adjacency_matrix)
        laplacian_matrix = torch.FloatTensor(laplacian_matrix).to(edge_index_batched.device)

        Ko = seq_length - (temporal_kernel_size - 1) * 2 * num_spatiotemporal_blocks

        blocks = []
        blocks.append([hidden_dim])

        for _ in range(num_spatiotemporal_blocks):
            blocks.append([hidden_dim, hidden_dim, hidden_dim])

        if Ko == 0:
            blocks.append([hidden_dim])
        elif Ko > 0:
            blocks.append([hidden_dim, hidden_dim])

        blocks.append([output_dim])

        # NOTE: `num_nodes_batched` must be the number of nodes in batched graph, not the original one
        # NOTE: `edge_index_batched` of train batched graph is passed, so train and eval batch sizes must be equal
        # NOTE: `input_dim` and `output_dim` are probably not used and required just for consistency

        self.backbone = STGCN(node_num=num_nodes_batched,
                              input_dim=hidden_dim,
                              output_dim=1,
                              horizon=output_dim,
                              seq_len=seq_length,
                              laplacian_matrix=laplacian_matrix,
                              blocks=blocks,
                              temporal_kernel_size=temporal_kernel_size,
                              spatial_kernel_size=spatial_kernel_size,
                              dropout=dropout)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)

    def forward(self, x, *args):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.backbone(x)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class STGODEAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim, normalization_name,
                 seq_length, num_nodes_batched, edge_index_batched, dropout, **kwargs):
        super().__init__()

        adjacency_matrix = to_dense_adj(edge_index_batched, max_num_nodes=num_nodes_batched).cpu().numpy()[0]
        transition_matrix = calculate_transition_matrix(adjacency_matrix)
        transition_matrix = torch.FloatTensor(transition_matrix).to(edge_index_batched.device)

        # NOTE: `num_nodes_batched` must be the number of nodes in batched graph, not the original one
        # NOTE: `edge_index_batched` of train batched graph is passed, so train and eval batch sizes must be equal
        # NOTE: `input_dim` and `output_dim` are probably not used and required just for consistency

        self.backbone = STGODE(node_num=num_nodes_batched,
                               input_dim=hidden_dim,
                               output_dim=1,
                               horizon=output_dim,
                               seq_len=seq_length,
                               hidden_dim=hidden_dim,
                               transition_matrix=transition_matrix)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)
    
    def forward(self, x, *args):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.backbone(x)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class STTNAdapter(nn.Module):
    def __init__(self, num_spatiotemporal_blocks, hidden_dim, output_dim, normalization_name,
                 seq_length, num_nodes_batched, edge_index_batched, dropout, **kwargs):
        super().__init__()

        adjacency_matrix = to_dense_adj(edge_index_batched, max_num_nodes=num_nodes_batched).cpu().numpy()[0]
        transition_matrix = calculate_transition_matrix(adjacency_matrix)
        transition_matrix = torch.FloatTensor(transition_matrix).to(edge_index_batched.device)

        # NOTE: `num_nodes_batched` must be the number of nodes in batched graph, not the original one
        # NOTE: `edge_index_batched` of train batched graph is passed, so train and eval batch sizes must be equal
        # NOTE: `input_dim` and `output_dim` are probably not used and required just for consistency

        self.backbone = STTN(node_num=num_nodes_batched,
                             input_dim=hidden_dim,
                             output_dim=1,
                             horizon=output_dim,
                             seq_len=seq_length,
                             num_blocks=num_spatiotemporal_blocks,
                             hidden_dim=hidden_dim,
                             transition_matrix=transition_matrix,
                             dropout=dropout)

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]
        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)
    
    def forward(self, x, *args):
        x = x.permute(1, 0, 2).unsqueeze(0)
        x = self.backbone(x)
        x = x.squeeze().permute(1, 0, 2)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


BASELINE_ADAPTERS = {
    ### torch spatiotemporal
    'DCRNN': DCRNNAdapter,
    'EGCN': EGCNAdapter,
    'GWN': GWNAdapter,
    'GGN': GGNAdapter,
    'GRUGCN': GRUGCNAdapter,

    ### large-st repository
    'AGCRN': AGCRNAdapter,
    'ASTGCN': ASTGCNAdapter,
    'GWNv2': GWNv2Adapter,
    'STGCN': STGCNAdapter,
    'STGODE': STGODEAdapter,
    'STTN': STTNAdapter,
}
