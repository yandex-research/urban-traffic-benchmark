import argparse
from tqdm import tqdm
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from dataset import Dataset
from models import ModelRegistry
from utils import Logger, get_parameter_groups, DummyHandler, NirvanaStateHandler, StateHandler

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


VAL_PREDICTIONS = None
TEST_PREDICTIONS = None
VAL_TARGETS = None
TEST_TARGETS = None
VAL_TARGETS_NAN_MASK = None
TEST_TARGETS_NAN_MASK = None


def get_args(add_name: bool = True):
    parser = argparse.ArgumentParser()

    if add_name:
        # This is needed for automatic config generation.
        parser.add_argument('--name', type=str, required=True, help='Experiment name.')

    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='pems-bay',
                        help='Dataset name (for an existing dataset in the data directory) or a path to a .npz file '
                             'with data. Possible dataset names: metr-la, pems-bay, largest, largest-2019.')
    parser.add_argument('--metric', type=str, default='MAE', choices=['MAE', 'RMSE'])
    parser.add_argument('--do_not_evaluate_on_test', default=False, action='store_true',
                        help='Only evaluate the model on val data, but not on test data. '
                             'Speeds up experiments when test metrics are not needed (e.g., hyperparameter search).')

    # Select future timestamps targets from which will be predicted by the model.
    parser.add_argument('--prediction_horizon', type=int, default=12)
    parser.add_argument('--only_predict_at_end_of_horizon', default=False, action='store_true')

    # Select past timestamps targets from which will be used as node features and passed as input to the model.
    parser.add_argument('--direct_lookback_num_steps', type=int, default=48)
    parser.add_argument('--seasonal_lookback_periods', nargs='+', type=int, default=None,
                        help='Should have the same number of values as seasonal_lookback_num_steps argument.')
    parser.add_argument('--seasonal_lookback_num_steps', nargs='+', type=int, default=None,
                        help='Should have the same number of values as seasonal_lookback_periods argument.')
    parser.add_argument('--drop_early_train_timestamps', type=str, default='direct', choices=['all', 'direct', 'none'])

    # Only for directed graphs: select which edge directions in the graph will be used.
    # Use at most one of these three arguments.
    parser.add_argument('--reverse_edges', default=False, action='store_true',
                        help='Reverse all edges in the graph.')
    parser.add_argument('--to_undirected', default=False, action='store_true',
                        help='Transform the graph to undirected by converting each directed edge into '
                             'an undirected one.')
    parser.add_argument('--use_forward_and_reverse_edges_as_different_edge_types', default=False, action='store_true',
                        help='The graph will be transformed to a heterogeneous graph with two edge types: '
                             'forward (original) and reverse edges. Graph neighborhood aggregation wiil be run for '
                             'each edge type separately and its results will be concatenated before being passed '
                             'to the following MLP module in the model.')

    # The next two arguments can be used to transform targets from the future timestamps that will be used for loss
    # computation during training and targets from the past timestamps and the current timestamp that will be provided
    # as features to the model. These two transformations can be different. Note that the metrics during evaluation are
    # always computed using the original untransformed targets.

    # Transformation applied to targets that will be used for loss computation (targets from the future timestamps that
    # will be predicted by the model during training).
    parser.add_argument('--targets_for_loss_transform', type=str, default='standard-scaler',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'quantile-transform-normal', 'quantile-transform-uniform'])

    # Transformation applied to targets that will be provided as features to the model (targets from the past timestamps
    # and the current timestamp).
    parser.add_argument('--targets_for_features_transform', type=str, default='standard-scaler',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'quantile-transform-normal', 'quantile-transform-uniform'])

    # NaN value imputation applied to targets that will be used for features (targets from past timestamps and current
    # timestamp).
    parser.add_argument('--targets_for_features_nan_imputation_strategy', type=str, default='prev',
                        choices=['prev', 'zero'])
    parser.add_argument('--add_nan_indicators_to_targets_for_features', default=False, action='store_true')

    # Drop unwanted node features.
    parser.add_argument('--do_not_use_temporal_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_spatial_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_spatiotemporal_features', default=False, action='store_true')

    # The next two arguments determine which time-based temporal features will be created. These features will be used
    # even if do_not_use_temporal_features argument is True. To prevent using them, set any of these two arguments to
    # an empty list.
    parser.add_argument('--time_based_features_types', nargs='*', type=str, choices=['one-hot', 'sin-cos'],
                        default=['one-hot', 'sin-cos'])
    parser.add_argument('--time_based_features_periods', nargs='*', type=str,
                        choices=['time-in-hour', 'time-in-day', 'hour-in-day', 'day-in-week', 'day-in-month',
                                 'day-in-year', 'week-in-year', 'month-in-year', 'auto'], default=['auto'],
                        help='auto will add periods based on train timespan and timestamp frequency from '
                             'the following set: '
                             'time-in-day (if there are at leat 5 train days and timestamp frequency is less than '
                             '1 day), '
                             'day-in-week (if there are at least 2 train weeks), '
                             'week-in-year (if there are at least 2 train years), '
                             'month-in-year (if there are at least 2 train years).')

    # Add additional node features.
    parser.add_argument('--use_deepwalk_node_embeddings', default=False, action='store_true')
    parser.add_argument('--use_learnable_node_embeddings', default=False, action='store_true',
                        help='Not used if model_class is Linear.')
    parser.add_argument('--learnable_node_embeddings_dim', type=int, default=128,
                        help='Only used if use_learnable_node_embeddings is True.')
    parser.add_argument('--initialize_learnable_node_embeddings_with_deepwalk', default=False, action='store_true',
                        help='Initializes learnable node embeddings with DeepWalk node embeddings. '
                             'This can be used instead of or in addition to using fixed (non-trainable) DeepWalk node '
                             'embeddings (which are controlled by use_deepwalk_node_embeddings arguments). '
                             'Only used if use_learnable_node_embeddings is True.')

    # Numerical features preprocessing.
    parser.add_argument('--numerical_features_transform', type=str, default='quantile-transform-normal',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'quantile-transform-normal', 'quantile-transform-uniform'])
    parser.add_argument('--numerical_features_nan_imputation_strategy', type=str, default='most_frequent',
                        choices=['mean', 'median', 'most_frequent'],
                        help='NaN imputation for numerical features. Imputation is done based on spatial statistics '
                             'and is thus only performed for spatial and spatiotemporal numerical features, but not '
                             'for temporal numerical features. It is expected that temporal numerical features have '
                             'no NaNs.')

    # PLR embeddings for numerical features. Not used if model_class is Linear.
    parser.add_argument('--use_plr_for_numerical_features', default=False, action='store_true',
                        help='Apply PLR embeddings to numerical features.')
    parser.add_argument('--plr_numerical_features_frequencies_dim', type=int, default=48,
                        help='Only used if plr_numerical_features is True.')
    parser.add_argument('--plr_numerical_features_frequencies_scale', type=float, default=0.01,
                        help='Only used if plr_numerical_features is True.')
    parser.add_argument('--plr_numerical_features_embedding_dim', type=int, default=16,
                        help='Only used if plr_numerical_features is True.')
    parser.add_argument('--plr_numerical_features_shared_linear', default=False, action='store_true',
                        help='Only used if plr_numerical_features is True.')
    parser.add_argument('--plr_numerical_features_shared_frequencies', default=False, action='store_true',
                        help='Only used if plr_numerical_features is True.')

    # PLR embeddings for past targets. Not used if model_class is Linear.
    parser.add_argument('--use_plr_for_past_targets', default=False, action='store_true',
                        help='Apply PLR embeddings to past targets.')
    parser.add_argument('--plr_past_targets_frequencies_dim', type=int, default=48,
                        help='Only used if plr_past_targets is True.')
    parser.add_argument('--plr_past_targets_frequencies_scale', type=float, default=0.01,
                        help='Only used if plr_past_targets is True.')
    parser.add_argument('--plr_past_targets_embedding_dim', type=int, default=16,
                        help='Only used if plr_past_targets is True.')
    parser.add_argument('--plr_past_targets_shared_linear', default=False, action='store_true',
                        help='Only used if plr_past_targets is True.')
    parser.add_argument('--plr_past_targets_shared_frequencies', default=False, action='store_true',
                        help='Only used if plr_past_targets is True.')

    # Use already preprocessed features.
    parser.add_argument('--spatiotemporal_preprocessed_features_filepath', default=None, type=str,
                        help='Optional argument indicating path to already preprocessed spatiotemporal features of '
                             'the dataset that can be used to save preprocessing time and memory.')

    # Model type selection.
    parser.add_argument('--model_class', type=str, default='SingleInputGNN',
                        choices=['LinearModel', 'ResNet', 'SingleInputGNN', 'SequenceInputGNN', 'BaselineModel'])
    parser.add_argument('--neighborhood_aggregation', type=str, default='MeanAggr',
                        choices=['MeanAggr', 'MaxAggr', 'GCNAggr', 'AttnGATAggr', 'AttnTrfAggr'],
                        help='Graph neighborhood aggregation (aka message passing) function for GNNs. '
                             'Only used if model_class is SingleInputGNN or SequenceInputGNN.')
    parser.add_argument('--do_not_separate_ego_node_representation', default=False, action='store_true',
                        help='Use ego node representation in graph neighborhood aggregation as if it is one more '
                             'neighbor representation instead of treating it separately by concatenating it to '
                             'aggregated neighbor representations.')
    parser.add_argument('--sequence_encoder', type=str, default='RNN',
                        choices=['RNN', 'Transformer'],
                        help='Timestamp sequence encoder applied before graph neighborhood aggregation. '
                             'Only used if model_class is SequenceInputGNN.')
    parser.add_argument('--normalization', type=str, default='LayerNorm',
                        choices=['none', 'LayerNorm', 'BatchNorm'],
                        help='Normalization applied in the beginning of each residual block. '
                             'Not used if model_class is LinearModel.')

    # Model architecture hyperparameters.
    # baseline parameters were moved separately from main parameters. Please don't change it as it's going on right now in nirvana
    # temporal_* are used both in baselines and SequenceInputGNN models
    parser.add_argument('--baseline_name', type=str, default='DCRNN',
                        choices=['AGCRN', 'ASTGCN', 'DCRNN', 'EGCN', 'GWN', 'GGN',
                                 'GRUGCN', 'GWNv2', 'STGCN', 'STGODE', 'STTN'])
    parser.add_argument('--num_spatiotemporal_blocks', type=int, default=2,
                        help='Number of spatiotemporal blocks in time-and-space baseline models.')
    parser.add_argument('--num_temporal_blocks', type=int, default=2,
                        help='Number of temporal blocks in time-then-space baseline models.')
    parser.add_argument('--num_spatial_blocks', type=int, default=2,
                        help='Number of spatial blocks in time-then-space baseline models.')
    parser.add_argument('--num_residual_blocks', type=int, default=2,
                        help='Number of residual blocks, where each residual block consists of the following sequence '
                             'of layers: normalization, sequence encoder (if model_class is SequenceInputGNN), '
                             'graph neighborhood aggregation (if model_class is SingleInputGNN or SequenceInputGNN), '
                             'two-layer MLP. '
                             'Not used if model_class is LinearModel.')
    parser.add_argument('--temporal_kernel_size', type=int, default=3,
                        help='Kernel size for temporal convolutions.')
    parser.add_argument('--temporal_dilation', type=int, default=2,
                        help='Dilation for temporal convolutions.')
    parser.add_argument('--spatial_kernel_size', type=int, default=2,
                        help='Kernel size for spatial convolutions.')

    # Common parameters (not baselines-exclusively)
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Not used if model_class is LinearModel.')
    parser.add_argument('--neighborhood_aggr_attn_num_heads', type=int, default=4,
                        help='Number of attention heads for attention-based graph neighborhood aggregation. '
                             'Only used if model_class is SingleInputGNN or SequenceInputGNN and '
                             'neighborhood_aggregation is AttnGAT or AttnTrf.')
    parser.add_argument('--seq_encoder_num_layers', type=int, default=4,
                        help='Number of layers in sequence encoder used in each residual block of the model. '
                             'Only used if model_class is SequenceInputGNN.')
    parser.add_argument('--seq_encoder_rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='RNN type used as sequence encoder. '
                             'Only used if model_class is SequenceInputGNN and sequence_encoder is RNN.')
    parser.add_argument('--seq_encoder_attn_num_heads', type=int, default=8,
                        help='Number of attention heads for attention-based sequence encoders. '
                             'Only used if model_class is SequenceInputGNN and sequence_encoder is Transformer.')
    parser.add_argument('--seq_encoder_bidir_attn', default=False, action='store_true',
                        help='Use bidirectional attention instead of unidirectional (aka causal) attention '
                             'in sequence encoder. '
                             'Only used if model_class is SequenceInputGNN and sequence_encoder is Transformer.')

    # Regularization.
    parser.add_argument('--dropout', type=float, default=0, help='Not used if model_class is LinearModel.')
    parser.add_argument('--weight_decay', type=float, default=0)

    # Training parameters.
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=10,
                        help='Effective batch size for each optimization step equals '
                             'train_batch_size * num_accumulation_steps.')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='If None, it is set to be the same as train_batch_size. But since evaluation requires '
                             'less VRAM than training, larger batch size can be used.')
    parser.add_argument('--num_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1000,
                        help='Evaluate after this many optimization steps. If None, only evaluate at the end of epoch.')
    parser.add_argument('--eval_max_num_predictions_per_step', type=int, default=1_000_000_000,
                        help='The maximum number of predictions that will be put on GPU for loss computation during '
                             'evaluation. Decrease this value if you face GPU OOM issues during evaluation.')

    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no_amp', default=False, action='store_true')
    parser.add_argument('--no_gradscaler', default=False, action='store_true')
    parser.add_argument('--num_threads', type=int, default=32)
    parser.add_argument('--nirvana', default=False, action='store_true',
                        help='Indicates that experiment is being run in Nirvana.')
    parser.add_argument('--disable_features_checkpointing', default=False, action='store_true',
                        help='Indicates whether to not to do checkpointing of features.')
    parser.add_argument('--checkpoint_steps_interval', type=int, default=1000,
                        help='Only used in Nirvana: interval for saving experiment state to $SNAPSHOT_PATH.')
    parser.add_argument('--compile', default=False, action='store_true',
                        help='Enables model compilation.')

    args = parser.parse_args()

    return args, parser


def compute_loss(model, dataset: Dataset, timestamps_batch, loss_fn, amp=True):
    features, targets, targets_nan_mask = dataset.get_timestamps_batch_features_and_targets_for_loss(timestamps_batch)

    with torch.autocast(enabled=amp, device_type=features.device.type):
        preds = model(graph=dataset.train_batched_graph, x=features)
        loss = loss_fn(input=preds, target=targets, reduction='none')
        loss[targets_nan_mask] = 0
        loss = loss.sum() / (~targets_nan_mask).sum()

    return loss


def optimizer_step(optimizer, gradscaler):
    gradscaler.step(optimizer)
    gradscaler.update()
    optimizer.zero_grad()


def compute_metric(preds, targets, targets_nan_mask, dataset, loss_fn, metric, apply_transform_to_preds=True):
    targets_transformed = targets.clone()

    if len(preds) < dataset.eval_max_num_timestamps_per_step:
        # Loss can be computed on GPU in one step.

        preds = preds.to(dataset.device)
        if apply_transform_to_preds:
            preds = dataset.transform_preds_for_metrics(preds)

        preds_transformed = preds.cpu().clone()

        targets = targets.to(dataset.device)
        targets_nan_mask = targets_nan_mask.to(dataset.device)

        loss = loss_fn(input=preds, target=targets, reduction='none')
        loss[targets_nan_mask] = 0
        loss_mean = loss.sum() / (~targets_nan_mask).sum()

    else:
        # Computing loss on GPU will be done in multiple steps.
        preds_transformed = dataset.transform_preds_for_metrics(preds.to(dataset.device)) if apply_transform_to_preds else preds.clone()
        preds_targets_dataset = TensorDataset(preds, targets, targets_nan_mask)
        preds_targets_loader = DataLoader(preds_targets_dataset, batch_size=dataset.eval_max_num_timestamps_per_step,
                                          shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

        loss_sum = 0
        loss_count = 0
        for cur_preds, cur_targets, cur_targets_nan_mask in preds_targets_loader:
            cur_preds = cur_preds.to(dataset.device)
            if apply_transform_to_preds:
                cur_preds = dataset.transform_preds_for_metrics(cur_preds)

            cur_targets = cur_targets.to(dataset.device)
            cur_targets_nan_mask = cur_targets_nan_mask.to(dataset.device)

            cur_loss = loss_fn(input=cur_preds, target=cur_targets, reduction='none')
            cur_loss[cur_targets_nan_mask] = 0
            cur_loss_sum = cur_loss.sum()
            cur_loss_count = (~cur_targets_nan_mask).sum()

            loss_sum += cur_loss_sum
            loss_count += cur_loss_count



        loss_mean = loss_sum / loss_count

    metric = loss_mean.sqrt().item() if metric == 'RMSE' else loss_mean.item()

    return metric, preds_transformed, targets_transformed


@torch.no_grad()
def evaluate_on_val_or_test(model, dataset, split, timestamps_loader, loss_fn, metric, amp=True):


    global VAL_PREDICTIONS
    global TEST_PREDICTIONS
    global VAL_TARGETS
    global TEST_TARGETS
    global VAL_TARGETS_NAN_MASK
    global TEST_TARGETS_NAN_MASK


    preds = []
    for timestamps_batch in timestamps_loader:
        padded = False
        if len(timestamps_batch) != dataset.eval_batch_size:
            padding_size = dataset.eval_batch_size - len(timestamps_batch)
            padding = torch.zeros(padding_size, dtype=torch.int32)
            timestamps_batch = torch.cat([timestamps_batch, padding], axis=0)
            padded = True

        cur_features = dataset.get_timestamps_batch_features(timestamps_batch)
        with torch.autocast(enabled=amp, device_type=cur_features.device.type):
            cur_preds = model(graph=dataset.eval_batched_graph, x=cur_features)

        cur_preds = cur_preds.reshape(dataset.eval_batch_size, dataset.num_nodes, dataset.targets_dim).squeeze(2)
        if padded:
            cur_preds = cur_preds[:-padding_size]

        preds.append(cur_preds.cpu())

    preds = torch.cat(preds, axis=0)

    if split == 'val':
        targets, targets_nan_mask = dataset.get_val_targets_for_metrics()
    elif split == 'test':
        targets, targets_nan_mask = dataset.get_test_targets_for_metrics()
    else:
        raise ValueError(f'Unknown split: {split}. Split argument should be either val or test.')

    metric, preds_transformed, targets_transformed = compute_metric(preds=preds, targets=targets, targets_nan_mask=targets_nan_mask, dataset=dataset,
                            loss_fn=loss_fn, metric=metric, apply_transform_to_preds=True)

    if split == 'val':
        VAL_PREDICTIONS = preds_transformed
        VAL_TARGETS = targets_transformed
        VAL_TARGETS_NAN_MASK = targets_nan_mask
    else:
        TEST_PREDICTIONS = preds_transformed
        TEST_TARGETS = targets_transformed
        TEST_TARGETS_NAN_MASK = targets_nan_mask

    return metric


@torch.no_grad()
def evaluate(model, dataset, val_timestamps_loader, test_timestamps_loader, loss_fn, metric, amp=True,
             do_not_evaluate_on_test=False):
    metrics = {}
    val_metric = evaluate_on_val_or_test(model=model, dataset=dataset, split='val',
                                         timestamps_loader=val_timestamps_loader, loss_fn=loss_fn,
                                         metric=metric, amp=amp)
    metrics[f'val {metric}'] = val_metric

    if not do_not_evaluate_on_test:
        test_metric = evaluate_on_val_or_test(model=model, dataset=dataset, split='test',
                                              timestamps_loader=test_timestamps_loader, loss_fn=loss_fn,
                                              metric=metric, amp=amp)
        metrics[f'test {metric}'] = test_metric

    return metrics


def train(model, dataset, loss_fn, metric, logger: Logger, num_epochs, num_accumulation_steps, eval_every, lr,
          weight_decay, run_id, device, state_handler: StateHandler, amp=True, use_gradscaler=True, seed=None,
          do_not_evaluate_on_test=False, nirvana=False):

    if seed is not None:
        torch.manual_seed(seed)
    elif nirvana:
        raise ValueError(
            'You must specify seed when training in Nirvana to ensure the same behaviour after every rescheduling.'
        )

    train_timestamps_loader = DataLoader(dataset.train_timestamps, batch_size=dataset.train_batch_size, shuffle=True,
                                         drop_last=True)
    val_timestamps_loader = DataLoader(dataset.val_timestamps, batch_size=dataset.eval_batch_size, shuffle=False,
                                       drop_last=False)
    test_timestamps_loader = DataLoader(dataset.test_timestamps, batch_size=dataset.eval_batch_size, shuffle=False,
                                        drop_last=False)

    num_steps = len(train_timestamps_loader) * num_epochs

    model.to(device)

    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay)
    gradscaler = torch.amp.GradScaler(enabled=use_gradscaler)

    state_handler.add_model(model=model)
    state_handler.add_optimizer(optimizer=optimizer)
    state_handler.add_grad_scaler(scaler=gradscaler)

    logger.start_run(run=run_id)
    epoch = state_handler.epochs_finished + 1
    steps_till_optimizer_step = num_accumulation_steps
    optimizer_steps_till_eval = eval_every
    metrics = {}
    train_timestamps_loader_iterator = iter(train_timestamps_loader)
    model.train()
    starting_step_idx = state_handler.steps_after_run_start
    with tqdm(total=num_steps, desc=f'Run {run_id}') as progress_bar:
        progress_bar.n = starting_step_idx
        if starting_step_idx > 0:
            t1 = perf_counter()
            print(f'Skipping {starting_step_idx} batches after rescheduling.')
            for step_to_skip in range(starting_step_idx % len(train_timestamps_loader_iterator)):
                next(train_timestamps_loader_iterator)
            t2 = perf_counter()
            print(f'Skipped {starting_step_idx} in {(t2 - t1):.3f} seconds.')

        for step in range(starting_step_idx + 1, num_steps + 1):
            cur_train_timestamps_batch = next(train_timestamps_loader_iterator)
            state_handler.loss = compute_loss(model=model, dataset=dataset, timestamps_batch=cur_train_timestamps_batch,
                                        loss_fn=loss_fn, amp=amp)

            steps_till_optimizer_step -= 1

            # we backward for each minibatch to free computation graph
            gradscaler.scale(state_handler.loss / num_accumulation_steps).backward()

            progress_bar.update()
            progress_bar.set_postfix(
                {metric: f'{value:.2f}' for metric, value in metrics.items()} |
                {'cur step loss': f'{state_handler.loss.item():.2f}', 'epoch': epoch}
            )


            if steps_till_optimizer_step == 0:
                optimizer_step(optimizer=optimizer, gradscaler=gradscaler)
                state_handler.loss = 0
                state_handler.optimizer_steps_done += 1

                steps_till_optimizer_step = num_accumulation_steps
                optimizer_steps_till_eval -= 1

            if (
                optimizer_steps_till_eval == 0 or
                train_timestamps_loader_iterator._num_yielded == len(train_timestamps_loader)
            ):
                progress_bar.set_postfix_str('     Evaluating...     ' + progress_bar.postfix)
                model.eval()
                metrics = evaluate(model=model, dataset=dataset, val_timestamps_loader=val_timestamps_loader,
                                test_timestamps_loader=test_timestamps_loader, loss_fn=loss_fn, metric=metric,
                                amp=amp, do_not_evaluate_on_test=do_not_evaluate_on_test)
                logger.update_metrics(metrics=metrics, step=state_handler.optimizer_steps_done, epoch=epoch)
                model.train()
                if step != num_steps and train_timestamps_loader_iterator._num_yielded != len(train_timestamps_loader): # prevent state handler to save 3 checkpoints at the same time on the last step
                    state_handler.save_checkpoint()

                if optimizer_steps_till_eval == 0:
                    optimizer_steps_till_eval = eval_every

            state_handler.step()

            if train_timestamps_loader_iterator._num_yielded == len(train_timestamps_loader):
                train_timestamps_loader_iterator = iter(train_timestamps_loader)
                epoch += 1
                if epoch < num_epochs: # prevent state handler to save 3 checkpoints at the same time on the last step
                    state_handler.finish_epoch()
                # check that logger, model and optimizer are shared also for state wrapper

    logger.finish_run()

    state_handler.finish_run()

    model.cpu()


def main():
    args, _ = get_args()

    torch.set_num_threads(args.num_threads)

    Model = ModelRegistry.get_model_class(args.model_class)

    # moved checkpoint logic to the front as it will be needed for dataset preprocessing
    CHECKPOINT_DIR = Path(args.save_dir)
    CHECKPOINT_STATE_FILENAME = CHECKPOINT_DIR / 'state.pt'

    checkpoint_steps_interval = args.checkpoint_steps_interval
    if args.nirvana:
        state_handler: StateHandler = NirvanaStateHandler(checkpoint_file_path=CHECKPOINT_STATE_FILENAME,
                                                          checkpoint_dir=CHECKPOINT_DIR,
                                                          checkpoint_steps_interval=checkpoint_steps_interval)
    else:
        state_handler: StateHandler = DummyHandler(checkpoint_file_path=CHECKPOINT_STATE_FILENAME,
                                                   checkpoint_dir=CHECKPOINT_DIR,
                                                   checkpoint_steps_interval=checkpoint_steps_interval)
    state_handler.load_checkpoint(initial_loading=True)
    whether_checkpoint_exists = CHECKPOINT_STATE_FILENAME.exists()
    logger = Logger(args=args, start_from_scratch=not whether_checkpoint_exists)
    state_handler.add_logger(logger=logger)

    use_edge_index = args.model_class == 'BaselineModel'

    dataset = Dataset(
        name_or_path=args.dataset,
        state_handler=state_handler,
        prediction_horizon=args.prediction_horizon,
        only_predict_at_end_of_horizon=args.only_predict_at_end_of_horizon,
        provide_sequnce_inputs=Model.sequence_input,
        direct_lookback_num_steps=args.direct_lookback_num_steps,
        seasonal_lookback_periods=args.seasonal_lookback_periods,
        seasonal_lookback_num_steps=args.seasonal_lookback_num_steps,
        drop_early_train_timestamps=args.drop_early_train_timestamps,
        reverse_edges=args.reverse_edges,
        to_undirected=args.to_undirected,
        use_forward_and_reverse_edges_as_different_edge_types=\
            args.use_forward_and_reverse_edges_as_different_edge_types,
        add_self_loops=args.do_not_separate_ego_node_representation,
        targets_for_loss_transform=args.targets_for_loss_transform,
        targets_for_features_transform=args.targets_for_features_transform,
        targets_for_features_nan_imputation_strategy=args.targets_for_features_nan_imputation_strategy,
        add_nan_indicators_to_targets_for_features=args.add_nan_indicators_to_targets_for_features,
        do_not_use_temporal_features=args.do_not_use_temporal_features,
        do_not_use_spatial_features=args.do_not_use_spatial_features,
        do_not_use_spatiotemporal_features=args.do_not_use_spatiotemporal_features,
        time_based_features_types=args.time_based_features_types,
        time_based_features_periods=args.time_based_features_periods,
        use_deepwalk_node_embeddings=args.use_deepwalk_node_embeddings,
        initialize_learnable_node_embeddings_with_deepwalk=args.initialize_learnable_node_embeddings_with_deepwalk,
        numerical_features_transform=args.numerical_features_transform,
        numerical_features_nan_imputation_strategy=args.numerical_features_nan_imputation_strategy,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_max_num_predictions_per_step=args.eval_max_num_predictions_per_step,
        device=args.device,
        nirvana=args.nirvana,
        spatiotemporal_features_local_processed_memmap_name=args.spatiotemporal_preprocessed_features_filepath,
        disable_features_checkpointing=args.disable_features_checkpointing,
        use_edge_index=use_edge_index,
    )

    if args.metric == 'RMSE':
        loss_fn = F.mse_loss
    elif args.metric == 'MAE':
        loss_fn = F.l1_loss
    else:
        raise ValueError(f'Unsupported metric: {args.metric}.')

    for run in range(state_handler.num_runs_completed + 1, args.num_runs + 1):
        model = Model(
            baseline_name=args.baseline_name,
            neighborhood_aggregation_name=args.neighborhood_aggregation,
            neighborhood_aggregation_sep=not args.do_not_separate_ego_node_representation,
            sequence_encoder_name=args.sequence_encoder,
            normalization_name=args.normalization,
            num_edge_types=1 if dataset.use_edge_index else len(dataset.graph.etypes),
            num_residual_blocks=args.num_residual_blocks,
            num_spatiotemporal_blocks=args.num_spatiotemporal_blocks,
            num_temporal_blocks=args.num_temporal_blocks,
            num_spatial_blocks=args.num_spatial_blocks,
            features_dim=dataset.features_dim,
            hidden_dim=args.hidden_dim,
            output_dim=dataset.targets_dim,
            temporal_kernel_size=args.temporal_kernel_size,
            temporal_dilation=args.temporal_dilation,
            spatial_kernel_size=args.spatial_kernel_size,
            neighborhood_aggr_attn_num_heads=args.neighborhood_aggr_attn_num_heads,
            seq_encoder_num_layers=args.seq_encoder_num_layers,
            seq_encoder_rnn_type_name=args.seq_encoder_rnn_type,
            seq_encoder_attn_num_heads=args.seq_encoder_attn_num_heads,
            seq_encoder_bidir_attn=args.seq_encoder_bidir_attn,
            seq_encoder_seq_len=args.direct_lookback_num_steps,
            dropout=args.dropout,
            use_learnable_node_embeddings=args.use_learnable_node_embeddings,
            num_nodes=dataset.num_nodes,
            batch_size=dataset.train_batch_size,
            edge_index_batched=dataset.train_batched_graph if use_edge_index else None,
            learnable_node_embeddings_dim=args.learnable_node_embeddings_dim,
            initialize_learnable_node_embeddings_with_deepwalk=args.initialize_learnable_node_embeddings_with_deepwalk,
            deepwalk_node_embeddings=dataset.deepwalk_embeddings_for_initializing_learnable_embeddings,
            use_plr_for_numerical_features=args.use_plr_for_numerical_features,
            numerical_features_mask=dataset.numerical_features_mask,
            plr_numerical_features_frequencies_dim=args.plr_numerical_features_frequencies_dim,
            plr_numerical_features_frequencies_scale=args.plr_numerical_features_frequencies_scale,
            plr_numerical_features_embedding_dim=args.plr_numerical_features_embedding_dim,
            plr_numerical_features_shared_linear=args.plr_numerical_features_shared_linear,
            plr_numerical_features_shared_frequencies=args.plr_numerical_features_shared_frequencies,
            use_plr_for_past_targets=args.use_plr_for_past_targets,
            past_targets_mask=dataset.past_targets_mask,
            plr_past_targets_frequencies_dim=args.plr_past_targets_frequencies_dim,
            plr_past_targets_frequencies_scale=args.plr_past_targets_frequencies_scale,
            plr_past_targets_embedding_dim=args.plr_past_targets_embedding_dim,
            plr_past_targets_shared_linear=args.plr_past_targets_shared_linear,
            plr_past_targets_shared_frequencies=args.plr_past_targets_shared_frequencies
        )

        if args.compile:
            model = torch.compile(model, dynamic=True, mode='reduce-overhead')

        train(model=model, dataset=dataset, loss_fn=loss_fn, metric=args.metric, logger=logger,
              num_epochs=args.num_epochs, num_accumulation_steps=args.num_accumulation_steps,
              eval_every=args.eval_every, lr=args.lr, weight_decay=args.weight_decay, run_id=run,
              device=args.device, amp=not args.no_amp, use_gradscaler=not args.no_gradscaler, seed=run,
              do_not_evaluate_on_test=args.do_not_evaluate_on_test, nirvana=args.nirvana, state_handler=state_handler,
              )

        state_handler.load_checkpoint()

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
