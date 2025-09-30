"""
Ok, so this module looks scary. It has abstract base classes, metaclasses, and all that stuff. But what you really need
to know to use it, is that there are two abstract base classes: SingleInputModel and SequenceInputModel, and each model
used in this project should be a subclass of one of these two abstract base classes. If the model only takes as input
information about one timestamp, that is, a tensor of shape [num_nodes, features_dim] (which might include information
about targets from previous timestamps in the features), than this model shpuld be a subclass of SingleInputModel.
If the model takes as input information about a sequence of timestamps, that is, a tensor of shape
[num_nodes, num_timestamps, features_dim], than this model should be a subclass of SequenceInputModel.
"""

from abc import ABC, ABCMeta

import torch
import torch.nn as nn

from modules import (FeaturesPreparatorForDeepModels, ResidualModulesWrapper, FeedForwardModule,
                     NEIGHBORHOOD_AGGREGATION_MODULES, SEQUENCE_ENCODER_MODULES, NORMALIZATION_MODULES,
                     BASELINE_ADAPTERS)

class ModelRegistry(ABCMeta):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = ABCMeta.__new__(mcs, name, bases, attrs)
        mcs.registry[new_cls.__name__] = new_cls

        return new_cls

    @classmethod
    def get_model_class(mcs, model_class_name):
        model_class = mcs.registry[model_class_name]

        return model_class


class SingleInputModel(nn.Module, ABC, metaclass=ModelRegistry):
    """
    Abstract base class for models that take as input a tensor of shape [num_nodes, features_dim].
    This input tensor contains for each node the features of this node at the current timestamp and the target values
    for this node at the current and previous timestamps. The model uses this data (in the case of a linear model) or
    deep representations of this data (in the case of deep models) to predict target values for this node at the future
    timestamps.

    Each model in this project should be a subclass of either this abstract base class or SequenceInputModel abstract
    base class.
    """
    single_input = True
    sequence_input = False


class SequenceInputModel(nn.Module, ABC, metaclass=ModelRegistry):
    """
    Abstract base class for models that take as input a tensor of shape [num_nodes, num_timestamps, features_dim].
    This input tensor contains for each node a sequence, in whcih each sequence element contains the features of
    this node and the target value of this node at a single timestamp - the timestamp corresponding to this sequence
    element. To predict target values for this node at the future timestamps, the deep representations of this sequence
    need to be converted to a single representation. This can be done by taking the deep representations of the final
    element of this sequence or by taking the mean and max of the deep representations computed over the entire
    sequence or by using all these methods and concatenating their outputs.

    Each model in this project should be a subclass of either this abstract base class or SingleInputModel abstract
    base class.
    """
    single_input = False
    sequence_input = True


class LinearModel(SingleInputModel):
    """A simple graph-agnostic linear model."""
    def __init__(self, features_dim, output_dim, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features=features_dim, out_features=output_dim)

    def forward(self, graph, x):
        return self.linear(x).squeeze(1)


class ResNet(SingleInputModel):
    """A graph-agnostic deep model with skip-connections and normalization."""
    def __init__(self, normalization_name, num_residual_blocks, features_dim, hidden_dim, output_dim, dropout,
                 use_learnable_node_embeddings, num_nodes, learnable_node_embeddings_dim,
                 initialize_learnable_node_embeddings_with_deepwalk, deepwalk_node_embeddings,
                 use_plr_for_numerical_features, numerical_features_mask, plr_numerical_features_frequencies_dim,
                 plr_numerical_features_frequencies_scale, plr_numerical_features_embedding_dim,
                 plr_numerical_features_shared_linear, plr_numerical_features_shared_frequencies,
                 use_plr_for_past_targets, past_targets_mask, plr_past_targets_frequencies_dim,
                 plr_past_targets_frequencies_scale, plr_past_targets_embedding_dim,
                 plr_past_targets_shared_linear, plr_past_targets_shared_frequencies,
                 **kwargs):
        super().__init__()

        NormalizationModule = NORMALIZATION_MODULES[normalization_name]

        self.features_preparator = FeaturesPreparatorForDeepModels(
            features_dim=features_dim,
            use_learnable_node_embeddings=use_learnable_node_embeddings,
            num_nodes=num_nodes,
            learnable_node_embeddings_dim=learnable_node_embeddings_dim,
            initialize_learnable_node_embeddings_with_deepwalk=initialize_learnable_node_embeddings_with_deepwalk,
            deepwalk_node_embeddings=deepwalk_node_embeddings,
            use_plr_for_numerical_features=use_plr_for_numerical_features,
            numerical_features_mask=numerical_features_mask,
            plr_numerical_features_frequencies_dim=plr_numerical_features_frequencies_dim,
            plr_numerical_features_frequencies_scale=plr_numerical_features_frequencies_scale,
            plr_numerical_features_embedding_dim=plr_numerical_features_embedding_dim,
            plr_numerical_features_shared_linear=plr_numerical_features_shared_linear,
            plr_numerical_features_shared_frequencies=plr_numerical_features_shared_frequencies,
            use_plr_for_past_targets=use_plr_for_past_targets,
            past_targets_mask=past_targets_mask,
            plr_past_targets_frequencies_dim=plr_past_targets_frequencies_dim,
            plr_past_targets_frequencies_scale=plr_past_targets_frequencies_scale,
            plr_past_targets_embedding_dim=plr_past_targets_embedding_dim,
            plr_past_targets_shared_linear=plr_past_targets_shared_linear,
            plr_past_targets_shared_frequencies=plr_past_targets_shared_frequencies
        )

        self.input_linear = nn.Linear(in_features=self.features_preparator.output_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_residual_blocks):
            residual_module = ResidualModulesWrapper(
                modules=[
                    NormalizationModule(hidden_dim),
                    FeedForwardModule(dim=hidden_dim, dropout=dropout)
                ]
            )

            self.residual_modules.append(residual_module)

        self.output_normalization = NormalizationModule(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.features_preparator(x)

        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class SingleInputGNN(SingleInputModel):
    """
    Like ResNet, but additionally has a graph neighborhood aggregation (aka message passing) module in each residual
    block. That is, each residual block consists of the following sequence of modules: normalization, graph
    neighborhood aggregation, two-layer MLP.
    """
    def __init__(self, neighborhood_aggregation_name, neighborhood_aggregation_sep, normalization_name, num_edge_types,
                 num_residual_blocks, features_dim, hidden_dim, output_dim, neighborhood_aggr_attn_num_heads, dropout,
                 use_learnable_node_embeddings, num_nodes, learnable_node_embeddings_dim,
                 initialize_learnable_node_embeddings_with_deepwalk, deepwalk_node_embeddings,
                 use_plr_for_numerical_features, numerical_features_mask, plr_numerical_features_frequencies_dim,
                 plr_numerical_features_frequencies_scale, plr_numerical_features_embedding_dim,
                 plr_numerical_features_shared_linear, plr_numerical_features_shared_frequencies,
                 use_plr_for_past_targets, past_targets_mask, plr_past_targets_frequencies_dim,
                 plr_past_targets_frequencies_scale, plr_past_targets_embedding_dim,
                 plr_past_targets_shared_linear, plr_past_targets_shared_frequencies,
                 **kwargs):
        super().__init__()

        NeighborhoodAggregationModule = NEIGHBORHOOD_AGGREGATION_MODULES[neighborhood_aggregation_name]
        NormalizationModule = NORMALIZATION_MODULES[normalization_name]

        self.features_preparator = FeaturesPreparatorForDeepModels(
            features_dim=features_dim,
            use_learnable_node_embeddings=use_learnable_node_embeddings,
            num_nodes=num_nodes,
            learnable_node_embeddings_dim=learnable_node_embeddings_dim,
            initialize_learnable_node_embeddings_with_deepwalk=initialize_learnable_node_embeddings_with_deepwalk,
            deepwalk_node_embeddings=deepwalk_node_embeddings,
            use_plr_for_numerical_features=use_plr_for_numerical_features,
            numerical_features_mask=numerical_features_mask,
            plr_numerical_features_frequencies_dim=plr_numerical_features_frequencies_dim,
            plr_numerical_features_frequencies_scale=plr_numerical_features_frequencies_scale,
            plr_numerical_features_embedding_dim=plr_numerical_features_embedding_dim,
            plr_numerical_features_shared_linear=plr_numerical_features_shared_linear,
            plr_numerical_features_shared_frequencies=plr_numerical_features_shared_frequencies,
            use_plr_for_past_targets=use_plr_for_past_targets,
            past_targets_mask=past_targets_mask,
            plr_past_targets_frequencies_dim=plr_past_targets_frequencies_dim,
            plr_past_targets_frequencies_scale=plr_past_targets_frequencies_scale,
            plr_past_targets_embedding_dim=plr_past_targets_embedding_dim,
            plr_past_targets_shared_linear=plr_past_targets_shared_linear,
            plr_past_targets_shared_frequencies=plr_past_targets_shared_frequencies
        )

        self.input_linear = nn.Linear(in_features=self.features_preparator.output_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_residual_blocks):
            residual_module = ResidualModulesWrapper(
                modules=[
                    NormalizationModule(hidden_dim),
                    NeighborhoodAggregationModule(dim=hidden_dim, num_heads=neighborhood_aggr_attn_num_heads,
                                                  num_edge_types=num_edge_types, dropout=dropout,
                                                  sep=neighborhood_aggregation_sep),
                    FeedForwardModule(dim=hidden_dim, num_inputs=num_edge_types + neighborhood_aggregation_sep,
                                      dropout=dropout)
                ]
            )

            self.residual_modules.append(residual_module)

        self.output_normalization = NormalizationModule(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.features_preparator(x)

        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class SequenceInputGNN(SequenceInputModel):
    """
    Like ResNet, but additionally has a sequence encoder module and a graph neighborhood aggregation (aka message
    passing) module in each residual block. That is, each residual block consists of the following sequence of modules:
    normalization, sequence encoder, graph neighborhood aggregation, two-layer MLP. Also has one sequence encoder
    module before all residual blocks.
    """
    def __init__(self, sequence_encoder_name, neighborhood_aggregation_name, neighborhood_aggregation_sep,
                 normalization_name, num_edge_types, num_residual_blocks, features_dim, hidden_dim, output_dim,
                 neighborhood_aggr_attn_num_heads, seq_encoder_num_layers, seq_encoder_rnn_type_name,
                 temporal_kernel_size, temporal_dilation, seq_encoder_attn_num_heads, seq_encoder_bidir_attn, seq_encoder_seq_len, dropout,
                 use_learnable_node_embeddings, num_nodes, learnable_node_embeddings_dim,
                 initialize_learnable_node_embeddings_with_deepwalk, deepwalk_node_embeddings,
                 use_plr_for_numerical_features, numerical_features_mask, plr_numerical_features_frequencies_dim,
                 plr_numerical_features_frequencies_scale, plr_numerical_features_embedding_dim,
                 plr_numerical_features_shared_linear, plr_numerical_features_shared_frequencies,
                 use_plr_for_past_targets, past_targets_mask, plr_past_targets_frequencies_dim,
                 plr_past_targets_frequencies_scale, plr_past_targets_embedding_dim,
                 plr_past_targets_shared_linear, plr_past_targets_shared_frequencies,
                 **kwargs):
        super().__init__()

        SequenceEncoderModule = SEQUENCE_ENCODER_MODULES[sequence_encoder_name]
        NeighborhoodAggregationModule = NEIGHBORHOOD_AGGREGATION_MODULES[neighborhood_aggregation_name]
        NormalizationModule = NORMALIZATION_MODULES[normalization_name]

        self.features_preparator = FeaturesPreparatorForDeepModels(
            features_dim=features_dim,
            use_learnable_node_embeddings=use_learnable_node_embeddings,
            num_nodes=num_nodes,
            learnable_node_embeddings_dim=learnable_node_embeddings_dim,
            initialize_learnable_node_embeddings_with_deepwalk=initialize_learnable_node_embeddings_with_deepwalk,
            deepwalk_node_embeddings=deepwalk_node_embeddings,
            use_plr_for_numerical_features=use_plr_for_numerical_features,
            numerical_features_mask=numerical_features_mask,
            plr_numerical_features_frequencies_dim=plr_numerical_features_frequencies_dim,
            plr_numerical_features_frequencies_scale=plr_numerical_features_frequencies_scale,
            plr_numerical_features_embedding_dim=plr_numerical_features_embedding_dim,
            plr_numerical_features_shared_linear=plr_numerical_features_shared_linear,
            plr_numerical_features_shared_frequencies=plr_numerical_features_shared_frequencies,
            use_plr_for_past_targets=use_plr_for_past_targets,
            past_targets_mask=past_targets_mask,
            plr_past_targets_frequencies_dim=plr_past_targets_frequencies_dim,
            plr_past_targets_frequencies_scale=plr_past_targets_frequencies_scale,
            plr_past_targets_embedding_dim=plr_past_targets_embedding_dim,
            plr_past_targets_shared_linear=plr_past_targets_shared_linear,
            plr_past_targets_shared_frequencies=plr_past_targets_shared_frequencies
        )

        self.input_linear = nn.Linear(in_features=self.features_preparator.output_dim, out_features=hidden_dim)
        self.input_sequence_encoder = SequenceEncoderModule(rnn_type_name=seq_encoder_rnn_type_name,
                                                            kernel_size=temporal_kernel_size, dilation=temporal_dilation,
                                                            num_layers=seq_encoder_num_layers, dim=hidden_dim,
                                                            num_heads=seq_encoder_attn_num_heads,
                                                            bidir_attn=seq_encoder_bidir_attn,
                                                            seq_len=seq_encoder_seq_len, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_residual_blocks):
            residual_module = ResidualModulesWrapper(
                modules=[
                    NormalizationModule(hidden_dim),
                    SequenceEncoderModule(rnn_type_name=seq_encoder_rnn_type_name, kernel_size=temporal_kernel_size, dilation=temporal_dilation,
                                          num_layers=seq_encoder_num_layers, dim=hidden_dim, num_heads=seq_encoder_attn_num_heads,
                                          bidir_attn=seq_encoder_bidir_attn, seq_len=seq_encoder_seq_len,
                                          dropout=dropout),
                    NeighborhoodAggregationModule(dim=hidden_dim, num_heads=neighborhood_aggr_attn_num_heads,
                                                  num_edge_types=num_edge_types, dropout=dropout,
                                                  sep=neighborhood_aggregation_sep),
                    FeedForwardModule(dim=hidden_dim, num_inputs=num_edge_types + neighborhood_aggregation_sep,
                                      dropout=dropout)
                ]
            )

            self.residual_modules.append(residual_module)

        self.output_normalization = NormalizationModule(hidden_dim * 3)
        self.output_linear = nn.Linear(in_features=hidden_dim * 3, out_features=output_dim)

    def forward(self, graph, x):
        x = self.features_preparator(x)

        x = self.input_linear(x)
        x = self.input_sequence_encoder(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x_final = x[:, -1]
        x_mean = x.mean(axis=1)
        x_max = x.max(axis=1).values
        x = torch.cat([x_final, x_mean, x_max], axis=1)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class BaselineModel(SequenceInputModel):
    def __init__(self, baseline_name, normalization_name, num_spatiotemporal_blocks, num_temporal_blocks, num_spatial_blocks,
                 features_dim, hidden_dim, output_dim, use_learnable_node_embeddings, num_nodes, batch_size, edge_index_batched,
                 learnable_node_embeddings_dim, temporal_kernel_size, temporal_dilation, spatial_kernel_size,
                 seq_encoder_seq_len, dropout, initialize_learnable_node_embeddings_with_deepwalk, deepwalk_node_embeddings,
                 use_plr_for_numerical_features, numerical_features_mask, plr_numerical_features_frequencies_dim,
                 plr_numerical_features_frequencies_scale, plr_numerical_features_embedding_dim,
                 plr_numerical_features_shared_linear, plr_numerical_features_shared_frequencies,
                 use_plr_for_past_targets, past_targets_mask, plr_past_targets_frequencies_dim,
                 plr_past_targets_frequencies_scale, plr_past_targets_embedding_dim,
                 plr_past_targets_shared_linear, plr_past_targets_shared_frequencies,
                 **kwargs):
        super().__init__()

        BaselineAdapter = BASELINE_ADAPTERS[baseline_name]

        self.features_preparator = FeaturesPreparatorForDeepModels(
            features_dim=features_dim,
            use_learnable_node_embeddings=use_learnable_node_embeddings,
            num_nodes=num_nodes,
            learnable_node_embeddings_dim=learnable_node_embeddings_dim,
            initialize_learnable_node_embeddings_with_deepwalk=initialize_learnable_node_embeddings_with_deepwalk,
            deepwalk_node_embeddings=deepwalk_node_embeddings,
            use_plr_for_numerical_features=use_plr_for_numerical_features,
            numerical_features_mask=numerical_features_mask,
            plr_numerical_features_frequencies_dim=plr_numerical_features_frequencies_dim,
            plr_numerical_features_frequencies_scale=plr_numerical_features_frequencies_scale,
            plr_numerical_features_embedding_dim=plr_numerical_features_embedding_dim,
            plr_numerical_features_shared_linear=plr_numerical_features_shared_linear,
            plr_numerical_features_shared_frequencies=plr_numerical_features_shared_frequencies,
            use_plr_for_past_targets=use_plr_for_past_targets,
            past_targets_mask=past_targets_mask,
            plr_past_targets_frequencies_dim=plr_past_targets_frequencies_dim,
            plr_past_targets_frequencies_scale=plr_past_targets_frequencies_scale,
            plr_past_targets_embedding_dim=plr_past_targets_embedding_dim,
            plr_past_targets_shared_linear=plr_past_targets_shared_linear,
            plr_past_targets_shared_frequencies=plr_past_targets_shared_frequencies
        )

        self.input_linear = nn.Linear(in_features=self.features_preparator.output_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.baseline_adapter = BaselineAdapter(num_spatiotemporal_blocks=num_spatiotemporal_blocks,
                                                num_temporal_blocks=num_temporal_blocks,
                                                num_spatial_blocks=num_spatial_blocks,
                                                hidden_dim=hidden_dim,
                                                output_dim=output_dim,
                                                normalization_name=normalization_name,
                                                temporal_kernel_size=temporal_kernel_size,
                                                temporal_dilation=temporal_dilation,
                                                spatial_kernel_size=spatial_kernel_size,
                                                seq_length=seq_encoder_seq_len,
                                                num_nodes_batched=num_nodes * batch_size,
                                                edge_index_batched=edge_index_batched,
                                                dropout=dropout)

    def forward(self, graph, x):
        edge_index = graph  # here we explicitly tell that graph is represented as edge index instead of other objects
                            # but the signature must remain with `graph` argument
        x = self.features_preparator(x)

        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        x = self.baseline_adapter(x, edge_index)

        return x
