import os
import typing as tp
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
import dgl
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from data_transforms import IdentityTransform, StandardScaler, MinMaxScaler, RobustScaler, QuantileTransform
from time_based_features import create_time_based_features
from utils import NirvanaNpzDataWrapper, StateHandler, get_tensor_or_wrap_memmap, read_memmap
from nirvana_utils import copy_out_to_snapshot


class Dataset:
    transforms = {
        'none': IdentityTransform,
        'standard-scaler': StandardScaler,
        'min-max-scaler': MinMaxScaler,
        'robust-scaler': RobustScaler,
        'quantile-transform-normal': partial(QuantileTransform, distribution='normal'),
        'quantile-transform-uniform': partial(QuantileTransform, distribution='uniform')
    }

    def __init__(self, name_or_path, state_handler: StateHandler, prediction_horizon=12, only_predict_at_end_of_horizon=False,
                 provide_sequnce_inputs=False, direct_lookback_num_steps=48,
                 seasonal_lookback_periods=None, seasonal_lookback_num_steps=None,
                 drop_early_train_timestamps='direct',
                 reverse_edges=False, to_undirected=False, use_forward_and_reverse_edges_as_different_edge_types=False,
                 add_self_loops=False, targets_for_loss_transform='none', targets_for_features_transform='none',
                 targets_for_features_nan_imputation_strategy='prev', add_nan_indicators_to_targets_for_features=False,
                 do_not_use_temporal_features=False, do_not_use_spatial_features=False,
                 do_not_use_spatiotemporal_features=False,
                 time_based_features_types=('one-hot', 'sin-cos'), time_based_features_periods=('auto',),
                 use_deepwalk_node_embeddings=False, initialize_learnable_node_embeddings_with_deepwalk=False,
                 numerical_features_transform='none', numerical_features_nan_imputation_strategy='most_frequent',
                 train_batch_size=1, eval_batch_size=None, eval_max_num_predictions_per_step=1_000_000_000,
                 device='cpu', nirvana=False, spatiotemporal_features_local_processed_memmap_name: str | None = None,
                 disable_features_checkpointing: bool = True, use_edge_index=False):

        DATA_ROOT = 'data'
        # torch.set_default_device(device)

        # NOTE this code can crash if the dataset doesn't have specified format
        if name_or_path.endswith('.npz'):
            name = os.path.splitext(os.path.basename(name_or_path))[0].replace('_', '-')
            path = name_or_path
        else:
            name = name_or_path
            path = f'{DATA_ROOT}/{name.replace("-", "_")}.npz'

        print('Preparing data...')
        if nirvana and not os.environ.get('LOCAL'):
            data = NirvanaNpzDataWrapper(root_path=DATA_ROOT)
        else:
            data = np.load(path, allow_pickle=True)

        # GET TIME SPLITS

        # Indices of all timestamps available for a particular split. Note that the number of timestamps at which
        # predictions can be made for a particular split will be different because it also depends on prediction
        # horizon (and, for train split, it also depends on lookback horizon). I.e., we cannot train on the last few
        # timestamps from the train split because the targets that should be predicted at these timestamps actually
        # belong (at least partly) to the val split. Indices of timestamps at which predictions can be made for
        # a particular split will be created later.
        all_train_timestamps = data['train_timestamps']
        all_val_timestamps = data['val_timestamps']
        all_test_timestamps = data['test_timestamps']

        train_slice = slice(all_train_timestamps[0], all_train_timestamps[-1] + 1)

        # PREPARE TARGETS
        (
            targets,
            targets_nan_mask,
            targets_for_loss_transform,
            targets_for_features_transform,
        ) = self._prepare_targets_or_return_from_state(
            checkpoint_dir=state_handler.checkpoint_dir,
            data=data,
            targets_for_loss_transform=targets_for_loss_transform,
            train_slice=train_slice,
            targets_for_features_transform=targets_for_features_transform,
            targets_for_features_nan_imputation_strategy=targets_for_features_nan_imputation_strategy,
            nirvana=nirvana,
            disable_features_checkpointing=disable_features_checkpointing,
        )

        num_timestamps = data['num_timestamps'].item() if 'num_timestamps' in data else targets.shape[0]
        num_nodes = data['num_nodes'].item() if 'num_nodes' in data else targets.shape[1]


        # PREPARE FEATURES
        temporal_features, temporal_feature_names, skip_temporal_features = (
            self._prepare_temporal_features_or_return_from_state(
                checkpoint_dir=state_handler.checkpoint_dir,
                data=data,
                do_not_use_temporal_features=do_not_use_temporal_features,
                num_timestamps=num_timestamps,
                nirvana=nirvana,
                disable_features_checkpointing=disable_features_checkpointing,
            )
        )
        spatial_features, spatial_feature_names, skip_spatial_features = (
            self._prepare_spatial_features_or_return_from_state(
                checkpoint_dir=state_handler.checkpoint_dir,
                data=data,
                do_not_use_spatial_features=do_not_use_spatial_features,
                num_nodes=num_nodes,
                nirvana=nirvana,
                disable_features_checkpointing=disable_features_checkpointing,
            )
        )

        spatiotemporal_features, spatiotemporal_feature_names, skip_spatiotemporal_features = (
            self._prepare_spatiotemporal_features_or_return_from_state(
                checkpoint_dir=state_handler.checkpoint_dir,
                data=data,
                do_not_use_spatiotemporal_features=do_not_use_spatiotemporal_features,
                num_nodes=num_nodes,
                num_timestamps=num_timestamps,
                spatiotemporal_features_local_processed_memmap_name=spatiotemporal_features_local_processed_memmap_name,
                data_root=DATA_ROOT,
                nirvana=nirvana,
                disable_features_checkpointing=disable_features_checkpointing,
            )
        )

        if use_deepwalk_node_embeddings or initialize_learnable_node_embeddings_with_deepwalk:
            if "deepwalk_node_embeddings" not in data:  # data can be wrapper with no .keys() method specified, but __contains__ is defined correctly for both cases and returns equivalent result
                raise ValueError(
                    "DeepWalk node embeddings are not provided for this dataset."
                )

        if use_deepwalk_node_embeddings:
            deepwalk_embeddings = data['deepwalk_node_embeddings'].astype(np.float32)
        else:
            deepwalk_embeddings = np.empty((1, num_nodes, 0), dtype=np.float32)

        if initialize_learnable_node_embeddings_with_deepwalk:
            deepwalk_embeddings_for_initializing_learnable_embeddings = data['deepwalk_node_embeddings']\
                .astype(np.float32)

        numerical_feature_names_set = set(data['num_feature_names'])
        binary_feature_names_set = set(data['bin_feature_names'])
        categorical_feature_names_set = set(data['cat_feature_names'])

        _pool_arguments = [
            [
                'temporal', temporal_features if not skip_temporal_features else np.empty((0, 0, temporal_features.shape[2])),
                temporal_feature_names, numerical_feature_names_set,
                categorical_feature_names_set, numerical_features_transform, numerical_features_nan_imputation_strategy,
                train_slice, skip_temporal_features, state_handler.checkpoint_dir, nirvana, disable_features_checkpointing
            ],
            [
                'spatial', spatial_features if not skip_spatial_features else np.empty((0, 0, spatial_features.shape[2])),
                spatial_feature_names, numerical_feature_names_set,
                categorical_feature_names_set, numerical_features_transform, numerical_features_nan_imputation_strategy,
                train_slice, skip_spatial_features, state_handler.checkpoint_dir, nirvana, disable_features_checkpointing
            ],
            [
                'spatiotemporal', spatiotemporal_features if not skip_spatiotemporal_features else np.empty((0, 0, spatiotemporal_features.shape[2])),
                spatiotemporal_feature_names, numerical_feature_names_set,
                categorical_feature_names_set, numerical_features_transform, numerical_features_nan_imputation_strategy,
                train_slice, skip_spatiotemporal_features, state_handler.checkpoint_dir, nirvana, disable_features_checkpointing
            ]
        ]

        with Pool(processes=3) as preprocessing_pool:
            features_preprocessing_results = preprocessing_pool.starmap(self._transform_feature_group, _pool_arguments)

        features_groups_maybe_returned, feature_names_groups, numerical_features_masks_by_group = zip(*features_preprocessing_results)

        # we need to correctly redefine features, b.c. if features are skipped from preprocessed, `features_groups_maybe_returned` contains
        # empty array for that features type with the last dimension equal to the last dimension of features type
        temporal_features_maybe_returned, spatial_features_maybe_returned, spatiotemporal_features_maybe_returned = features_groups_maybe_returned

        temporal_features = (
            temporal_features_maybe_returned
            if not skip_temporal_features
            else temporal_features
        )

        spatial_features = (
            spatial_features_maybe_returned
            if not skip_spatial_features
            else spatial_features
        )

        spatiotemporal_features = (
            spatiotemporal_features_maybe_returned
            if not skip_spatiotemporal_features
            else spatiotemporal_features
        )

        temporal_feature_names, spatial_feature_names, spatiotemporal_feature_names = feature_names_groups
        numerical_temporal_features_mask, numerical_spatial_features_mask, numerical_spatiotemporal_features_mask = \
            numerical_features_masks_by_group

        # Add time-based features.
        if time_based_features_types and time_based_features_periods:
            unix_timestamps = data['unix_timestamps']

            time_based_features, time_based_feature_names, numerical_time_based_features_mask = \
                create_time_based_features(unix_timestamps=unix_timestamps,
                                           time_based_features_types=time_based_features_types,
                                           time_based_features_periods=time_based_features_periods,
                                           last_train_timestamp_idx=all_train_timestamps[-1])

            temporal_features = np.concatenate([temporal_features, time_based_features], axis=2)
            temporal_feature_names += time_based_feature_names
            numerical_temporal_features_mask = np.concatenate(
                [numerical_temporal_features_mask, numerical_time_based_features_mask], axis=0
            )

        # PREPARE GRAPH

        if sum([reverse_edges, to_undirected, use_forward_and_reverse_edges_as_different_edge_types]) > 1:
            raise ValueError('At most one of the graph edge processing arguments reverse_edges, to_undirected, '
                             'use_forward_and_reverse_edges_as_different_edge_types can be True.')

        edges = torch.from_numpy(data['edges'])

        if use_forward_and_reverse_edges_as_different_edge_types:
            if use_edge_index:
                raise ValueError(
                    'The use of both forward and reverse edges in the graph as different edge types is not supported '
                    'for PyG graphs. Arguments use_forward_and_reverse_edges_as_different_edge_types and pyg cannot be '
                    'both True.'
                )

            graph = dgl.heterograph(
                {
                    ('node', 'forward_edge', 'node'): (edges[:, 0], edges[:, 1]),
                    ('node', 'reverse_edge', 'node'): (edges[:, 1], edges[:, 0])
                },
                num_nodes_dict={'node': num_nodes},
                idtype=torch.int32
            )

        else:
            graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes, idtype=torch.int32)
            if to_undirected:
                graph = dgl.to_bidirected(graph)
            elif reverse_edges:
                graph = dgl.reverse(graph)

        if add_self_loops:
            for cur_edge_type in graph.etypes:
                graph = dgl.add_self_loop(graph, etype=cur_edge_type)

        train_batched_graph = dgl.batch([graph for _ in range(train_batch_size)])
        if eval_batch_size is not None and eval_batch_size != train_batch_size:
            eval_batched_graph = dgl.batch([graph for _ in range(eval_batch_size)])

        if use_edge_index:
            # Convert DGL graphs to edge index (which is simply a pair of torch tensors storing edges).
            graph = torch.stack(graph.edges(), axis=0)
            train_batched_graph = torch.stack(train_batched_graph.edges(), axis=0).long()  # torch_scatter needs int64 instead of int32
            if eval_batch_size is not None and eval_batch_size != train_batch_size:
                eval_batched_graph = torch.stack(eval_batched_graph.edges(), axis=0).long()  # torch_scatter needs int64 instead of int32

        # We do not need the original data anymore.
        del data

        # PREPARE INDEX SHIFTS FROM THE CURRENT TIMESTAMP TO PAST TARGETS THAT WILL BE USED AS FEATURES

        # Check validity of seasonal lookback arguments.
        if (provide_sequnce_inputs and
                (seasonal_lookback_periods is not None or seasonal_lookback_num_steps is not None)):
            raise ValueError(
                'Seasonal lookback is not meant to be used with sequence input, as sequential timestamps are required '
                'for sequence input to a model. Either set sequence input to False and use a single input model or '
                'disable seasonal lookback by setting seasonal_lookback_periods and seasonal_lookback_num_steps '
                'arguments to None.'
            )
        elif seasonal_lookback_periods is not None and seasonal_lookback_num_steps is None:
            raise ValueError(
                'If argument seasonal_lookback_periods is provided, then argument seasonal_lookback_num_steps should '
                'also be provided.'
            )
        elif seasonal_lookback_periods is None and seasonal_lookback_num_steps is not None:
            raise ValueError(
                'If argument seasonal_lookback_num_steps is provided, then argument seasonal_lookback_periods should '
                'also be provided.'
            )
        elif (seasonal_lookback_periods is not None and
              len(seasonal_lookback_periods) != len(seasonal_lookback_num_steps)):
            raise ValueError(
                'Arguments seasonal_lookback_periods and seasonal_lookback_num_steps should be provided with the same '
                'number of values.'
            )

        past_timestamp_shifts_for_features = - np.arange(start=0, stop=direct_lookback_num_steps)

        if seasonal_lookback_periods is not None:
            # For each predicted target at timestamp t we add to features target values from the past at the following
            # timestamps: t - period, t - period * 2, ..., t - period * num_steps.
            all_seasonal_shifts = []
            for period, num_steps in zip(seasonal_lookback_periods, seasonal_lookback_num_steps):
                start_shifts = - period * np.arange(start=1, stop=num_steps + 1)
                for start_shift in start_shifts:
                    if only_predict_at_end_of_horizon:
                        cur_shifts = np.array([start_shift + prediction_horizon])
                    else:
                        cur_shifts = np.arange(start=start_shift + 1, stop=start_shift + prediction_horizon + 1)

                    all_seasonal_shifts.append(cur_shifts)

            all_seasonal_shifts = np.hstack(all_seasonal_shifts)
            all_seasonal_shifts = all_seasonal_shifts.clip(max=0)

            past_timestamp_shifts_for_features = np.hstack([past_timestamp_shifts_for_features, all_seasonal_shifts])
            past_timestamp_shifts_for_features = np.unique(past_timestamp_shifts_for_features)
            past_timestamp_shifts_for_features.sort()
            # A copy is made after reversing the array because otherwise the stride in the array will be negative,
            # and because of this it will not be possible to load it into a torch tensor using torch.from_numpy.
            past_timestamp_shifts_for_features = past_timestamp_shifts_for_features[::-1].copy()

        # PREPARE INDEX SHIFTS FROM THE CURRENT TIMESTAMP TO FUTURE TARGETS THAT WILL BE PREDICTED

        if only_predict_at_end_of_horizon:
            future_timestamp_shifts_for_prediction = np.array([prediction_horizon])
        else:
            future_timestamp_shifts_for_prediction = np.arange(start=1, stop=prediction_horizon + 1)

        # PREPARE INDICES OF TIMESTAMPS AT WHICH PREDICTIONS WILL BE MADE FOR EACH SPLIT

        if provide_sequnce_inputs and drop_early_train_timestamps != 'direct':
            raise ValueError(
                f'If provide_sequnce_inputs argument is True, than only "direct" is a valid value for '
                f'drop_early_train_timestamps argument, but value {drop_early_train_timestamps} was provided instead.'
            )

        if drop_early_train_timestamps == 'all':
            first_train_timestamp = - past_timestamp_shifts_for_features.min()
        elif drop_early_train_timestamps == 'direct':
            first_train_timestamp = direct_lookback_num_steps - 1
        elif drop_early_train_timestamps == 'none':
            first_train_timestamp = 0
        else:
            raise ValueError(f'Unknown value for drop_early_train_timestamps argument: {drop_early_train_timestamps}.')

        first_train_timestamp = max(first_train_timestamp, 0)

        max_prediction_shift = future_timestamp_shifts_for_prediction[-1]

        train_timestamps = all_train_timestamps[first_train_timestamp:-max_prediction_shift]

        val_timestamps = np.concatenate(
            [np.array([all_train_timestamps[-1]]), all_val_timestamps[:-max_prediction_shift]], axis=0
        )

        test_timestamps = np.concatenate(
            [np.array([all_val_timestamps[-1]]), all_test_timestamps[:-max_prediction_shift]], axis=0
        )

        # Remove train timestamps for which all targets are NaNs.
        train_targets_nan_mask = targets_nan_mask[first_train_timestamp + 1:all_train_timestamps[-1] + 1]
        if only_predict_at_end_of_horizon or prediction_horizon == 1:
            # In this case max_prediction_shift is the only prediction shift.
            # Transform train_targets_nan_mask to shape [len(train_timestamps), num_nodes].
            train_targets_nan_mask = train_targets_nan_mask[max_prediction_shift - 1:]
            drop_train_timestamps_mask = train_targets_nan_mask.all(axis=1)
        else:
            # Transform train_targets_nan_mask to shape [len(train_timestamps), num_nodes, prediction_horizon].
            train_targets_nan_mask = torch.from_numpy(train_targets_nan_mask)
            train_targets_nan_mask = train_targets_nan_mask.unfold(dimension=0, size=prediction_horizon, step=1)
            train_targets_nan_mask = train_targets_nan_mask.numpy()
            drop_train_timestamps_mask = train_targets_nan_mask.all(axis=(1, 2))

        train_timestamps = train_timestamps[~drop_train_timestamps_mask]

        # COMPUTE FEATURES DIM AND PREPARE FEATURES MASKS

        # Dataset provides features as a single tensor. In this tensor, the order of features is as follows:
        # past targets, past targets NaN mask, temporal features, spatial features, spatiotemporal features,
        # deepwalk node embeddings.

        past_targets_features_dim = 1 if provide_sequnce_inputs else len(past_timestamp_shifts_for_features)
        past_targets_nan_mask_features_dim = past_targets_features_dim * add_nan_indicators_to_targets_for_features
        features_dim = (past_targets_features_dim + past_targets_nan_mask_features_dim +
                        temporal_features.shape[2] + spatial_features.shape[2] + spatiotemporal_features.shape[2] +
                        deepwalk_embeddings.shape[2])

        past_targets_mask = np.zeros(features_dim, dtype=bool)
        past_targets_mask[:past_targets_features_dim] = True

        numerical_features_mask = np.concatenate([
            np.zeros(past_targets_features_dim + past_targets_nan_mask_features_dim, dtype=bool),
            numerical_temporal_features_mask, numerical_spatial_features_mask, numerical_spatiotemporal_features_mask,
            np.zeros(deepwalk_embeddings.shape[2], dtype=bool)
        ], axis=0)

        # STORE EVERYTHING WE MIGHT NEED

        self.use_edge_index = use_edge_index
        self.name = name
        self.provide_sequence_inputs = provide_sequnce_inputs
        self.device = device

        self.num_timestamps = num_timestamps
        self.num_nodes = num_nodes

        # Indices of all timestamps available for a particular split.
        self.all_train_timestamps = torch.from_numpy(all_train_timestamps)
        self.all_val_timestamps = torch.from_numpy(all_val_timestamps)
        self.all_test_timestamps = torch.from_numpy(all_test_timestamps)

        # Indices of timestamps at which predictions can be made for a particular split.
        self.train_timestamps = torch.from_numpy(train_timestamps)
        self.val_timestamps = torch.from_numpy(val_timestamps)
        self.test_timestamps = torch.from_numpy(test_timestamps)

        self.targets = get_tensor_or_wrap_memmap(targets)
        self.targets_nan_mask = get_tensor_or_wrap_memmap(targets_nan_mask)

        self.add_nan_indicators_to_targets_for_features = add_nan_indicators_to_targets_for_features
        self.targets_for_loss_transform = targets_for_loss_transform.torch().to(device)
        self.targets_for_features_transform = targets_for_features_transform.torch().to(device)

        self.temporal_features = get_tensor_or_wrap_memmap(temporal_features)
        self.temporal_feature_names = temporal_feature_names
        self.spatial_features = get_tensor_or_wrap_memmap(spatial_features)
        self.spatial_feature_names = spatial_feature_names
        self.spatiotemporal_features = get_tensor_or_wrap_memmap(spatiotemporal_features)
        self.spatiotemporal_feature_names = spatiotemporal_feature_names
        self.deepwalk_embeddings = get_tensor_or_wrap_memmap(deepwalk_embeddings)

        # Spatial node features and node embeddings are the same for all timestamps, so we load them on the appropriate
        # device and batch in advance to avoid doing it each training step.
        self.spatial_features_batched_train = self.spatial_features.to(device).repeat(1, train_batch_size, 1)
        self.deepwalk_embeddings_batched_train = self.deepwalk_embeddings.to(device).repeat(1, train_batch_size, 1)
        if eval_batch_size is None or eval_batch_size == train_batch_size:
            self.spatial_features_batched_eval = self.spatial_features_batched_train
            self.deepwalk_embeddings_batched_eval = self.deepwalk_embeddings_batched_train
        else:
            self.spatial_features_batched_eval = self.spatial_features.to(device).repeat(1, eval_batch_size, 1)
            self.deepwalk_embeddings_batched_eval = self.deepwalk_embeddings.to(device).repeat(1, eval_batch_size, 1)

        # Might be used for applying numerical feature embeddings.
        self.numerical_features_mask = torch.from_numpy(numerical_features_mask).to(device)
        self.past_targets_mask = torch.from_numpy(past_targets_mask).to(device)

        if initialize_learnable_node_embeddings_with_deepwalk:
            self.deepwalk_embeddings_for_initializing_learnable_embeddings = torch.from_numpy(
                deepwalk_embeddings_for_initializing_learnable_embeddings
            )
        else:
            self.deepwalk_embeddings_for_initializing_learnable_embeddings = None

        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size if eval_batch_size is None else eval_batch_size

        self.graph = graph
        self.train_batched_graph = train_batched_graph.to(device)
        if eval_batch_size is None or eval_batch_size == train_batch_size:
            self.eval_batched_graph = self.train_batched_graph
        else:
            self.eval_batched_graph = eval_batched_graph.to(device)

        self.future_timestamp_shifts_for_prediction = torch.from_numpy(future_timestamp_shifts_for_prediction)
        self.past_timestamp_shifts_for_features = torch.from_numpy(past_timestamp_shifts_for_features)

        self.targets_dim = 1 if only_predict_at_end_of_horizon else prediction_horizon
        self.past_targets_features_dim = past_targets_features_dim
        self.features_dim = features_dim
        self.seq_len = direct_lookback_num_steps if provide_sequnce_inputs else None

        self.eval_max_num_timestamps_per_step = eval_max_num_predictions_per_step // self.targets_dim // num_nodes
        if not disable_features_checkpointing:
            copy_out_to_snapshot(state_handler.checkpoint_dir, dump=True) # dump all prepared features, transforms and targets

    def get_timestamp_features_as_single_input(self, timestamp):
        past_timestamps = timestamp + self.past_timestamp_shifts_for_features
        negative_mask = (past_timestamps < 0)
        past_timestamps[negative_mask] = 0

        past_targets = self.targets[past_timestamps].to(self.device)
        past_targets_orig_shape = past_targets.shape
        past_targets = self.targets_for_features_transform.transform(past_targets.reshape(-1, 1))\
            .reshape(*past_targets_orig_shape)
        past_targets = past_targets.T

        if self.add_nan_indicators_to_targets_for_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps]
            past_targets_nan_mask[negative_mask] = 1
            past_targets_nan_mask = past_targets_nan_mask.to(self.device).T
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes, 0, device=self.device)

        temporal_features = self.temporal_features[timestamp].to(self.device).expand(self.num_nodes, -1)
        spatiotemporal_features = self.spatiotemporal_features[timestamp].to(self.device)
        spatial_features = self.spatial_features.to(self.device).squeeze(0)
        deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).squeeze(0)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=1)

        return features

    def get_timestamp_features_as_sequence_input(self, timestamp):
        past_timestamps = timestamp + self.past_timestamp_shifts_for_features

        past_targets = self.targets[past_timestamps].to(self.device)
        past_targets_orig_shape = past_targets.shape
        past_targets = self.targets_for_features_transform.transform(past_targets.reshape(-1, 1))\
            .reshape(*past_targets_orig_shape)
        past_targets = past_targets.T.unsqueeze(-1)

        if self.add_nan_indicators_to_targets_for_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps].to(self.device).T.unsqueeze(-1)
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes, self.seq_len, 0, device=self.device)

        temporal_features = self.temporal_features[past_timestamps].to(self.device).transpose(0, 1)\
            .expand(self.num_nodes, -1, -1)
        spatiotemporal_features = self.spatiotemporal_features[past_timestamps].to(self.device).transpose(0, 1)
        spatial_features = self.spatial_features.to(self.device).squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)
        deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=2)

        return features

    def get_timestamp_features(self, timestamp):
        if self.provide_sequence_inputs:
            return self.get_timestamp_features_as_sequence_input(timestamp)
        else:
            return self.get_timestamp_features_as_single_input(timestamp)

    def get_timestamp_targets_for_loss(self, timestamp):
        future_timestamps = timestamp + self.future_timestamp_shifts_for_prediction
        targets = self.targets[future_timestamps].to(self.device)
        targets_orig_shape = targets.shape
        targets = self.targets_for_loss_transform.transform(targets.reshape(-1, 1)).reshape(*targets_orig_shape)
        targets = targets.T.squeeze(1)
        targets_nan_mask = self.targets_nan_mask[future_timestamps].to(self.device).T.squeeze(1)

        return targets, targets_nan_mask

    def get_timestamp_features_and_targets_for_loss(self, timestamp):
        features = self.get_timestamp_features(timestamp)
        targets, targets_nan_mask = self.get_timestamp_targets_for_loss(timestamp)

        return features, targets, targets_nan_mask

    def get_timestamps_batch_features_as_single_input(self, timestamps_batch):
        batch_size = len(timestamps_batch)

        # The shape of past_timestamps is [batch_size, past_targets_features_dim].
        past_timestamps = timestamps_batch[:, None] + self.past_timestamp_shifts_for_features[None, :]
        negative_mask = (past_timestamps < 0)
        past_timestamps[negative_mask] = 0

        past_targets = self.targets[past_timestamps].to(self.device)
        past_targets_orig_shape = past_targets.shape
        past_targets = self.targets_for_features_transform.transform(past_targets.reshape(-1, 1))\
            .reshape(*past_targets_orig_shape)

        # The shape of past targets (and later the shape of past_targets_nan_mask) changes from
        # [batch_size, past_targets_features_dim, num_nodes] to
        # [batch_size, num_nodes, past_targets_features_dim] to
        # [num_nodes * batch_size, past_targets_features_dim].
        past_targets = past_targets.transpose(1, 2).flatten(start_dim=0, end_dim=1)

        if self.add_nan_indicators_to_targets_for_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps]
            past_targets_nan_mask[negative_mask] = 1
            past_targets_nan_mask = past_targets_nan_mask.to(self.device).transpose(1, 2)\
                .flatten(start_dim=0, end_dim=1)
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes * batch_size, 0, device=self.device)

        temporal_features = self.temporal_features[timestamps_batch].to(self.device).squeeze(1)\
            .repeat_interleave(repeats=self.num_nodes, dim=0)
        spatiotemporal_features = self.spatiotemporal_features[timestamps_batch].to(self.device, non_blocking=True)\
            .flatten(start_dim=0, end_dim=1)

        if batch_size == self.train_batch_size:
            spatial_features = self.spatial_features_batched_train.squeeze(0)
            deepwalk_embeddings = self.deepwalk_embeddings_batched_train.squeeze(0)
        elif batch_size == self.eval_batch_size:
            spatial_features = self.spatial_features_batched_eval.squeeze(0)
            deepwalk_embeddings = self.deepwalk_embeddings_batched_eval.squeeze(0)
        else:
            spatial_features = self.spatial_features.to(self.device).squeeze(0).repeat(batch_size, 1)
            deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).squeeze(0).repeat(batch_size, 1)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=1)

        return features

    def get_timestamps_batch_features_as_sequence_input(self, timestamps_batch):
        batch_size = len(timestamps_batch)

        # The shape of past_timestamps is [batch_size, seq_len].
        past_timestamps = timestamps_batch[:, None] + self.past_timestamp_shifts_for_features[None, :]

        past_targets = self.targets[past_timestamps].to(self.device)
        past_targets_orig_shape = past_targets.shape
        past_targets = self.targets_for_features_transform.transform(past_targets.reshape(-1, 1))\
            .reshape(*past_targets_orig_shape)

        # The shape of past targets (and later the shape of past_targets_nan_mask) changes from
        # [batch_size, seq_len, num_nodes] to
        # [batch_size, num_nodes, seq_len] to
        # [num_nodes * batch_size, seq_len] to
        # [num_nodes * batch_size, seq_len, 1].
        past_targets = past_targets.transpose(1, 2).flatten(start_dim=0, end_dim=1).unsqueeze(-1)

        if self.add_nan_indicators_to_targets_for_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps].to(self.device).transpose(1, 2)\
                .flatten(start_dim=0, end_dim=1).unsqueeze(-1)
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes * batch_size, self.seq_len, 0, device=self.device)

        temporal_features = self.temporal_features[past_timestamps].to(self.device).squeeze(2)\
            .repeat_interleave(repeats=self.num_nodes, dim=0)
        spatiotemporal_features = self.spatiotemporal_features[past_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1)

        if batch_size == self.train_batch_size:
            spatial_features = self.spatial_features_batched_train
            deepwalk_embeddings = self.deepwalk_embeddings_batched_train
        elif batch_size == self.eval_batch_size:
            spatial_features = self.spatial_features_batched_eval
            deepwalk_embeddings = self.deepwalk_embeddings_batched_eval
        else:
            spatial_features = self.spatial_features.to(self.device).repeat(1, batch_size, 1)
            deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).repeat(1, batch_size, 1)

        spatial_features = spatial_features.squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)
        deepwalk_embeddings = deepwalk_embeddings.squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=2)

        return features

    def get_timestamps_batch_features(self, timestamps_batch):
        if self.provide_sequence_inputs:
            return self.get_timestamps_batch_features_as_sequence_input(timestamps_batch)
        else:
            return self.get_timestamps_batch_features_as_single_input(timestamps_batch)

    def get_timestamps_batch_targets_for_loss(self, timestamps_batch):
        # The shape of future_timestamps is [batch_size, targets_dim].
        future_timestamps = timestamps_batch[:, None] + self.future_timestamp_shifts_for_prediction[None, :]
        targets = self.targets[future_timestamps].to(self.device)
        targets_orig_shape = targets.shape
        targets = self.targets_for_loss_transform.transform(targets.reshape(-1, 1)).reshape(*targets_orig_shape)

        # The shape of targets and targets_nan_mask changes from
        # [batch_size, targets_dim, num_nodes] to
        # [batch_size, num_nodes, targets_dim] to
        # [num_nodes * batch_size, targets_dim],
        # and if targets_dim is 1, then the last dimension is squeezed.
        targets = targets.transpose(1, 2).flatten(start_dim=0, end_dim=1).squeeze(1)
        targets_nan_mask = self.targets_nan_mask[future_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1).squeeze(1)

        return targets, targets_nan_mask

    def get_timestamps_batch_features_and_targets_for_loss(self, timestamps_batch):
        features = self.get_timestamps_batch_features(timestamps_batch)
        targets, targets_nan_mask = self.get_timestamps_batch_targets_for_loss(timestamps_batch)

        return features, targets, targets_nan_mask

    def transform_preds_for_metrics(self, preds):
        preds_orig_shape = preds.shape
        preds = self.targets_for_loss_transform.inverse_transform(preds.reshape(-1, 1)).reshape(*preds_orig_shape)

        return preds

    def get_val_targets_for_metrics(self):
        targets = self.targets[self.all_val_timestamps]
        targets_nan_mask = self.targets_nan_mask[self.all_val_timestamps]

        targets, targets_nan_mask = self.prepare_targets_for_evaluation(targets, targets_nan_mask)

        return targets, targets_nan_mask

    def get_test_targets_for_metrics(self):
        targets = self.targets[self.all_test_timestamps]
        targets_nan_mask = self.targets_nan_mask[self.all_test_timestamps]

        targets, targets_nan_mask = self.prepare_targets_for_evaluation(targets, targets_nan_mask)

        return targets, targets_nan_mask

    def prepare_targets_for_evaluation(self, targets, targets_nan_mask):
        if self.targets_dim == 1:
            # If we only predict target for a single timestamp (which is the case if only_predict_at_end_of_horizon
            # is True or if prediction_horizon is 1), we need to shift all targets by the number of timestamps between
            # the observed timestamp and the timestamp of prediction.
            future_timestamp_shift = self.future_timestamp_shifts_for_prediction[0]
            targets = targets[future_timestamp_shift - 1:]
            targets_nan_mask = targets_nan_mask[future_timestamp_shift - 1:]
        else:
            # If we predict targets for multiple timestamps, we need to go over the targets with a sliding window
            # of length num_predictions and create a tensor of shape [num_timestamps, num_nodes, num_predictions].
            targets = targets.unfold(dimension=0, size=self.targets_dim, step=1)
            targets_nan_mask = targets_nan_mask.unfold(dimension=0, size=self.targets_dim, step=1)

        return targets, targets_nan_mask

    def _transform_feature_group(
            self,
            features_type: tp.Literal['temporal', 'spatial', 'spatiotemporal'],
            features: np.ndarray,
            feature_names: tp.Sequence[str],
            numerical_features_names_set: set[str],
            categorical_features_names_set: set[str],
            numerical_features_transform: tp.Literal[
                'none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                'quantile-transform-normal', 'quantile-transform-uniform'
            ],
            numerical_features_nan_imputation_strategy: tp.Literal['mean', 'median', 'most_frequent', 'constant'],
            train_slice: slice,
            skip: bool = False,
            checkpoint_dir: Path = Path("NULL"),
            nirvana: bool = False,
            disable_features_checkpointing: bool = True,
    ):
        numerical_features_mask = np.zeros(features.shape[2], dtype=bool)
        categorical_features_mask = np.zeros(features.shape[2], dtype=bool)
        for i, feature_name in enumerate(feature_names):
            if feature_name in numerical_features_names_set:
                numerical_features_mask[i] = True
            elif feature_name in categorical_features_names_set:
                categorical_features_mask[i] = True

        if skip:
            print(f'Skipped preprocessing {features_type} features.')
            return features, feature_names, numerical_features_mask

        print(
            f'{features_type=} {features.shape=} {feature_names=} '
            f'{numerical_features_mask=} {categorical_features_mask=} '
        )

        # Transform numerical features and impute NaNs in numerical features.
        if numerical_features_mask.any():
            numerical_features = features[:, :, numerical_features_mask]

            # Transform numerical features.
            numerical_features_transform = self.transforms[numerical_features_transform]()
            numerical_features_transform.fit(
                numerical_features.squeeze(0) if features_type == 'spatial' else
                numerical_features[train_slice].reshape(-1, numerical_features.shape[2])
            )
            numerical_features_orig_shape = numerical_features.shape
            numerical_features = numerical_features_transform.transform(
                numerical_features.reshape(-1, numerical_features.shape[2])
            ).reshape(*numerical_features_orig_shape)

            # Impute NaNs in numerical features. Note that NaNs are imputed based on spatial statistics, and are
            # thus not imputed for temporal features (features_group_idx == 0). It is expected that temporal
            # features do not have NaNs.
            if np.isnan(numerical_features).any():
                if features_type == 'temporal':
                    raise ValueError(
                        'NaN values in temporal features are not supported because imputation is done based on '
                        'spatial statistics.'
                    )

                numerical_features = numerical_features.transpose(1, 0, 2)
                numerical_features_imputer = SimpleImputer(missing_values=np.nan,
                                                           strategy=numerical_features_nan_imputation_strategy,
                                                           copy=False)
                numerical_features_transposed_shape = numerical_features.shape
                numerical_features = numerical_features_imputer.fit_transform(
                    numerical_features.reshape(numerical_features.shape[0], -1)
                ).reshape(*numerical_features_transposed_shape)
                numerical_features = numerical_features.transpose(1, 0, 2)

            # Put transformed and imputed numerical features back into features array.
            slow_idx = 0
            for fast_idx in range(features.shape[2]):
                if numerical_features_mask[fast_idx]:
                    features[:, :, fast_idx] = numerical_features[:, :, slow_idx]
                    slow_idx += 1

        # Apply one-hot encoding to categorical features.
        if categorical_features_mask.any():
            categorical_features = features[:, :, categorical_features_mask]

            if np.isnan(categorical_features).any():
                raise ValueError(
                    'NaN values in categorical features are not supported. It is suggested to replace them with their '
                    'own category.'
                )

            categorical_features_orig_shape = categorical_features.shape
            categorical_features = categorical_features.reshape(-1, categorical_features.shape[2])

            one_hot_encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
            categorical_features_encoded = one_hot_encoder.fit_transform(categorical_features)
            categorical_features_encoded = categorical_features_encoded.reshape(
                *categorical_features_orig_shape[:2], categorical_features_encoded.shape[1]
            )

            one_hot_encoder_output_feature_names = one_hot_encoder.get_feature_names_out(
                input_features=np.array(feature_names, dtype=object)[categorical_features_mask]
            ).tolist()

            # Change features array, feature names, and numerical features mask to include one-hot encoded
            # categorical features.
            features_new = []
            feature_names_new = []
            numerical_features_mask_new = []
            categorical_feature_idx = -1
            start_idx = 0
            for i in range(features.shape[2]):
                feature = features[:, :, i, None]
                if not categorical_features_mask[i]:
                    features_new.append(feature)
                    feature_names_new.append(feature_names[i])
                    numerical_features_mask_new.append(numerical_features_mask[i])
                else:
                    categorical_feature_idx += 1
                    num_categories = len(one_hot_encoder.categories_[categorical_feature_idx])
                    feature = categorical_features_encoded[:, :, start_idx:start_idx + num_categories]
                    features_new.append(feature)
                    feature_names_new += one_hot_encoder_output_feature_names[start_idx:start_idx + num_categories]
                    numerical_features_mask_new += [False for _ in range(num_categories)]
                    start_idx += num_categories

            features = np.concatenate(features_new, axis=2)
            feature_names = feature_names_new
            numerical_features_mask = np.array(numerical_features_mask_new, dtype=bool)

        print(f'Processed {features_type} features.')
        if checkpoint_dir.exists() and nirvana and not disable_features_checkpointing:  # NOTE we'll never step on this line for already preprocessed features, so there is no need to create additional conditions 
            print(f"Saved prepared {features_type} features for further checkpointing")
            prepared_features_file = str(checkpoint_dir / f"__{features_type}_features_prepared.npy")
            np.save(prepared_features_file, features)

        return features, feature_names, numerical_features_mask

    def _prepare_targets_or_return_from_state(self, checkpoint_dir: Path, data, targets_for_loss_transform, train_slice,
                                                  targets_for_features_transform, targets_for_features_nan_imputation_strategy,
                                                  nirvana, disable_features_checkpointing):
        targets_prepared_file = checkpoint_dir / "__targets_prepared.npy"
        targets_nan_mask_prepared_file = checkpoint_dir / "__targets_nan_mask_prepared.npy"
        targets_for_loss_transform_prepared_file = checkpoint_dir / "__targets_for_loss_transform_prepared.bin"
        targets_for_features_transform_prepared_file = checkpoint_dir / "__targets_for_features_transform_prepared.bin"

        if targets_prepared_file.exists() and targets_nan_mask_prepared_file.exists() and nirvana and not disable_features_checkpointing:
            print("Loading prepared targets")
            targets = np.load(str(targets_prepared_file))
            targets_nan_mask = np.load(str(targets_nan_mask_prepared_file))
            targets_for_loss_transform = joblib.load(targets_for_loss_transform_prepared_file)
            targets_for_features_transform = joblib.load(targets_for_features_transform_prepared_file)

        else:
            targets = data['targets'].astype(np.float32)
            targets_nan_mask = np.isnan(targets)

            # Prepare the transform that will be applied to targets from future timestamps which will be used for loss
            # computation during training (note that during evaluation the predictions will be passed through the reverse
            # transform and the metrics will be computed using the original untransformed targets).
            targets_for_loss_transform = self.transforms[targets_for_loss_transform]()
            targets_for_loss_transform.fit(targets[train_slice].reshape(-1, 1))

            # Prepare the transform that will be applied to targets from the past timestamps and the current timestamp and
            # provided as features to the model.
            targets_for_features_transform = self.transforms[targets_for_features_transform]()
            targets_for_features_transform.fit(targets[train_slice].reshape(-1, 1))

            # Impute NaNs in targets.
            if targets_for_features_nan_imputation_strategy == 'prev':
                if targets_nan_mask.all(axis=0).any():
                    # Before imputing any other targets, let's handle the nodes in the dataset for which no targets are
                    # available at all. We may not want to remove such nodes because their position in the graph and their
                    # features may provide useful information. But we need to impute their targets somehow and we cannot do
                    # it with previous target values because there are no such values for these nodes. We will impute these
                    # targets in one of two ways.
                    if not targets_nan_mask.all(axis=1).any():
                        # If there are no timestamps for which all targets are NaN, then we will impute targets of nodes
                        # with no known targets at each timestamp with the mean of all known targets at this timestamp.
                        targets[:, targets_nan_mask.all(axis=0)] = np.nanmean(targets, axis=1, keepdims=True)
                    else:
                        # If there are timestamps for which all targets are NaN, then we will impute targets of nodes
                        # with no known targets at each timestamp with the mean of all known targets of all nodes at all
                        # train timestamps.
                        targets[:, targets_nan_mask.all(axis=0)] = np.nanmean(targets[train_slice])

                if np.isnan(targets)[train_slice].all(axis=0).any():
                    raise RuntimeError(
                        'There are nodes in the dataset for which all train targets are NaN. "prev" imputation strategy '
                        'for NaN targets cannot be applied in this case. Modify the dataset (e.g., by removing these '
                        'nodes) or set imputation_startegy_for_nan_targets argument to "zero".'
                    )

                # Now, we will impute NaN targets with the latest known target value by running forward fill.
                targets_df = pd.DataFrame(targets, copy=False)
                targets_df.ffill(axis=0, inplace=True)

                # If some nodes have NaN targets starting from the first train timestamp, these NaN values are still left
                # not imputed after forward fill. We will impute them with the first known target value by running backward
                # fill. Note that we have already verified that there are at least some train target values that are not
                # NaN for each node, and thus it is guaranteed that this will not lead to future targets leakage from val
                # and test timestamps.
                if np.isnan(targets).any():
                    targets_df.bfill(axis=0, inplace=True)

                # targets numpy array has been modified by modifying targets_df pandas dataframe which shares the data
                # with it. We do not need the targets_df pandas dataframe anymore.
                del targets_df

            elif targets_for_features_nan_imputation_strategy == 'zero':
                targets[targets_nan_mask] = 0

            else:
                raise ValueError(f'Unsupported value for targets_for_features_nan_imputation_strategy: '
                                f'{targets_for_features_nan_imputation_strategy}. Supported values are: "prev", "zero".')

            if not disable_features_checkpointing and nirvana and checkpoint_dir.exists():
                np.save(str(targets_prepared_file), targets)
                np.save(str(targets_nan_mask_prepared_file), targets_nan_mask)
                joblib.dump(targets_for_loss_transform, targets_for_loss_transform_prepared_file)
                joblib.dump(targets_for_features_transform, targets_for_features_transform_prepared_file)

        return targets, targets_nan_mask, targets_for_loss_transform, targets_for_features_transform

    def _prepare_temporal_features_or_return_from_state(self, checkpoint_dir: Path, data, do_not_use_temporal_features: bool, num_timestamps: int, nirvana: bool,
                                                        disable_features_checkpointing: bool):
        temporal_features_prepared_file = checkpoint_dir / "__temporal_features_prepared.npy"
        if temporal_features_prepared_file.exists() and nirvana and not disable_features_checkpointing:
            temporal_features = np.load(str(temporal_features_prepared_file))
            temporal_feature_names = data['temporal_node_feature_names'].tolist()
            skip_temporal_features = True
            print("Loaded prepared temporal features")
        else:
            if do_not_use_temporal_features:
                temporal_features = np.empty((num_timestamps, 1, 0), dtype=np.float32)
                temporal_feature_names = []
                skip_temporal_features = True
            else:
                temporal_features = data['temporal_node_features'].astype(np.float32)
                temporal_feature_names = data['temporal_node_feature_names'].tolist()
                skip_temporal_features = False

        return temporal_features, temporal_feature_names, skip_temporal_features

    def _prepare_spatial_features_or_return_from_state(self, checkpoint_dir: Path, data, do_not_use_spatial_features: bool, num_nodes: int, nirvana: bool,
                                                       disable_features_checkpointing: bool):
        spatial_features_prepared_file = checkpoint_dir / "__spatial_features_prepared.npy"
        if spatial_features_prepared_file.exists() and nirvana and not disable_features_checkpointing:
            spatial_features = np.load(str(spatial_features_prepared_file))
            spatial_feature_names = data['spatial_node_feature_names'].tolist()
            skip_spatial_features = True
            print("Loaded prepared spatial features")
        else:
            if do_not_use_spatial_features:
                spatial_features = np.empty((1, num_nodes, 0), dtype=np.float32)
                spatial_feature_names = []
                skip_spatial_features = True
            else:
                spatial_features = data['spatial_node_features'].astype(np.float32)
                spatial_feature_names = data['spatial_node_feature_names'].tolist()
                skip_spatial_features = False
        return spatial_features, spatial_feature_names, skip_spatial_features

    def _prepare_spatiotemporal_features_or_return_from_state(self, checkpoint_dir: Path, data, do_not_use_spatiotemporal_features: bool, num_timestamps, num_nodes,
                                                              spatiotemporal_features_local_processed_memmap_name, data_root, nirvana: bool,
                                                              disable_features_checkpointing: bool):
        spatiotemporal_features_prepared_file = checkpoint_dir / "__spatiotemporal_features_prepared.npy"
        if spatiotemporal_features_prepared_file.exists() and nirvana and not disable_features_checkpointing and spatiotemporal_features_local_processed_memmap_name is None:
            spatiotemporal_features = np.load(str(spatiotemporal_features_prepared_file))
            spatiotemporal_feature_names = data['spatiotemporal_node_feature_names'].tolist()
            skip_spatiotemporal_features = True
            print("Loaded prepared (not memmap) spatiotemporal features")
        else:
            if do_not_use_spatiotemporal_features:
                spatiotemporal_features = np.empty((num_timestamps, num_nodes, 0), dtype=np.float32)
                spatiotemporal_feature_names = []
                skip_spatiotemporal_features = True
            else:
                spatiotemporal_feature_names = data['spatiotemporal_node_feature_names'].tolist()
                if spatiotemporal_features_local_processed_memmap_name is None:
                    spatiotemporal_features = data['spatiotemporal_node_features'].astype(np.float32)
                    skip_spatiotemporal_features = False
                    
                else:
                    spatiotemporal_features = read_memmap(
                        filepath=os.path.join(data_root, spatiotemporal_features_local_processed_memmap_name),
                        shape=(num_timestamps, num_nodes, len(spatiotemporal_feature_names)),
                    )
                    skip_spatiotemporal_features = True
                    print("Loaded preprocessed memmap features from YT")
        return spatiotemporal_features, spatiotemporal_feature_names, skip_spatiotemporal_features
