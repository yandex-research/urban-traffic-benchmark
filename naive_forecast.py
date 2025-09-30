import argparse
import warnings
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from dataset import Dataset
from run_single_experiment import compute_metric
from utils import DummyHandler


def get_args():
    parser = argparse.ArgumentParser()

    # These arguments define the task.
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (for an existing dataset in the data directory) or a path to a .npz file '
                             'with data. Possible dataset names: metr-la, pems-bay, largest, largest-2019.')
    parser.add_argument('--metric', type=str, default='MAE', choices=['MAE', 'RMSE'])
    parser.add_argument('--prediction_horizon', type=int, default=12)
    parser.add_argument('--only_predict_at_end_of_horizon', default=False, action='store_true')

    # These arguments define the forecast methods to be used.
    parser.add_argument('--methods', nargs='+', type=str, required=True,
                        choices=['constant', 'per-node-constant', 'prev-latest', 'prev-periodic'])
    parser.add_argument('--constants', nargs='+', type=lambda x: x if x in ('mean', 'median') else int(x),
                        help='Used for "constant" forecast method. Each value of this argument should be '
                             'either an int or one of the strings ["mean", "median"].')
    parser.add_argument('--per-node-constants', nargs='+', type=str, choices=['mean', 'median'],
                        help='Used for "per-node-constant" forecast method.')
    parser.add_argument('--periods', nargs='+', type=int, help='Used for "prev-periodic" forecast method.')

    parser.add_argument('--eval_max_num_predictions_per_step', type=int, default=1_000_000_000)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    if 'constant' in args.methods and args.constants is None:
        raise ValueError(
            'If "constant" is one of the values of argument methods, than at least one value for argument '
            'constants should be provided.'
        )

    if 'per-node-constant' in args.methods and args.per_node_constants is None:
        raise ValueError(
            'If "per-node-constant" is one of the values of argument methods, than at least one value for argument '
            'per-node-constants should be provided.'
        )

    if 'prev-periodic' in args.methods and args.periods is None:
        raise ValueError(
            'If "prev-periodic" is one of the values of argument methods, than at least one value for argument '
            'periods should be provided.'
        )

    if 'prev-periodic' in args.methods:
        for period in args.periods:
            if period < args.prediction_horizon:
                raise ValueError(
                    'Values of argument periods cannot be smaller than the value of argument prediction_horizon, '
                    'otherwise future data leakage will happen.'
                )

    return args


def compute_and_print_metrics(val_preds, test_preds, val_targets, test_targets, val_targets_nan_mask,
                              test_targets_nan_mask, dataset, loss_fn, metric, print_header):
    val_metric, _, _ = compute_metric(preds=val_preds, targets=val_targets, targets_nan_mask=val_targets_nan_mask,
                                dataset=dataset, loss_fn=loss_fn, metric=metric, apply_transform_to_preds=False)
    test_metric, _, _ = compute_metric(preds=test_preds, targets=test_targets, targets_nan_mask=test_targets_nan_mask,
                                 dataset=dataset, loss_fn=loss_fn, metric=metric, apply_transform_to_preds=False)

    print(print_header + ':')
    print(f'val {metric}: {val_metric:.4f}, test {metric}: {test_metric:.4f}')
    print()


def main():
    args = get_args()

    dataset = Dataset(name_or_path=args.dataset,
                      state_handler=DummyHandler(Path('dummy_path'), Path('.'), 1),
                      prediction_horizon=args.prediction_horizon,
                      only_predict_at_end_of_horizon=args.only_predict_at_end_of_horizon,
                      drop_early_train_timestamps='none',
                      targets_for_features_nan_imputation_strategy='prev',
                      do_not_use_temporal_features=True,
                      do_not_use_spatial_features=True,
                      do_not_use_spatiotemporal_features=True,
                      time_based_features_types=[],
                      time_based_features_periods=[],
                      eval_max_num_predictions_per_step=args.eval_max_num_predictions_per_step,
                      device=args.device)

    if args.metric == 'RMSE':
        loss_fn = F.mse_loss
    elif args.metric == 'MAE':
        loss_fn = F.l1_loss
    else:
        raise ValueError(f'Unsupported metric: {args.metric}.')

    val_targets, val_targets_nan_mask = dataset.get_val_targets_for_metrics()
    test_targets, test_targets_nan_mask = dataset.get_test_targets_for_metrics()

    print()

    if 'constant' in args.methods:
        for const in args.constants:
            str_value = const if const in ('mean', 'median') else ''
            if str_value:
                train_targets = dataset.targets[dataset.all_train_timestamps]
                train_targets_nan_mask = dataset.targets_nan_mask[dataset.all_train_timestamps]
                known_train_targets = train_targets[~train_targets_nan_mask]
                if str_value == 'mean':
                    const = known_train_targets.mean().item()
                elif str_value == 'median':
                    const = known_train_targets.median().item()
                else:
                    raise ValueError(
                        f'Unsupported value for argument constants: {str_value}. Supported values are: '
                        f'"mean", "median", any float.'
                    )

            val_preds = torch.tensor(const).expand_as(val_targets)
            test_preds = torch.tensor(const).expand_as(test_targets)

            print_header = f'Constant forecast with value {const:.4f}'
            print_header = print_header + f' (train target {str_value})' if str_value else print_header
            compute_and_print_metrics(val_preds=val_preds, test_preds=test_preds, val_targets=val_targets,
                                      test_targets=test_targets, val_targets_nan_mask=val_targets_nan_mask,
                                      test_targets_nan_mask=test_targets_nan_mask, dataset=dataset, loss_fn=loss_fn,
                                      metric=args.metric, print_header=print_header)

    if 'per-node-constant' in args.methods:
        train_targets = dataset.targets[dataset.all_train_timestamps].numpy().copy()
        train_targets_nan_mask = dataset.targets_nan_mask[dataset.all_train_timestamps].numpy()
        train_targets[train_targets_nan_mask] = np.nan
        for str_value in args.per_node_constants:
            if str_value == 'mean':
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Mean of empty slice')
                    consts = np.nanmean(train_targets, axis=0)
            elif str_value == 'median':
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                    consts = np.nanmedian(train_targets, axis=0)
            else:
                raise ValueError(
                    f'Unsupported value for argument per_node_constants: {str_value}. Supported values are: '
                    f'"mean", "median".'
                )

            consts = torch.from_numpy(consts)[None, :, None]
            val_preds = consts.expand_as(val_targets)
            test_preds = consts.expand_as(test_targets)

            print_header = f'Per node constant forecast with per node train target {str_value}'
            compute_and_print_metrics(val_preds=val_preds, test_preds=test_preds, val_targets=val_targets,
                                      test_targets=test_targets, val_targets_nan_mask=val_targets_nan_mask,
                                      test_targets_nan_mask=test_targets_nan_mask, dataset=dataset, loss_fn=loss_fn,
                                      metric=args.metric, print_header=print_header)

    if 'prev-latest' in args.methods:
        val_preds = dataset.targets[dataset.val_timestamps]
        test_preds = dataset.targets[dataset.test_timestamps]
        if not args.only_predict_at_end_of_horizon:
            val_preds = val_preds.unsqueeze(2).expand(-1, -1, args.prediction_horizon)
            test_preds = test_preds.unsqueeze(2).expand(-1, -1, args.prediction_horizon)

        print_header = 'Forecast with the latest known value'
        compute_and_print_metrics(val_preds=val_preds, test_preds=test_preds, val_targets=val_targets,
                                  test_targets=test_targets, val_targets_nan_mask=val_targets_nan_mask,
                                  test_targets_nan_mask=test_targets_nan_mask, dataset=dataset, loss_fn=loss_fn,
                                  metric=args.metric, print_header=print_header)

    if 'prev-periodic' in args.methods:
        for period in args.periods:
            if args.only_predict_at_end_of_horizon:
                val_preds = dataset.targets[dataset.val_timestamps - period + args.prediction_horizon]
                test_preds = dataset.targets[dataset.test_timestamps - period + args.prediction_horizon]

            else:
                val_idx = dataset.val_timestamps - period + 1
                val_idx = torch.cat([
                    val_idx, torch.arange(val_idx[-1] + 1, val_idx[-1] + args.prediction_horizon, dtype=torch.int64)
                ], axis=0)
                val_preds = dataset.targets[val_idx].unfold(dimension=0, size=args.prediction_horizon, step=1)

                test_idx = dataset.test_timestamps - period + 1
                test_idx = torch.cat([
                    test_idx, torch.arange(test_idx[-1] + 1, test_idx[-1] + args.prediction_horizon, dtype=torch.int64)
                ], axis=0)
                test_preds = dataset.targets[test_idx].unfold(dimension=0, size=args.prediction_horizon, step=1)

            print_header = f'Forecast with previous periodic values with a period of {period} steps'
            compute_and_print_metrics(val_preds=val_preds, test_preds=test_preds, val_targets=val_targets,
                                      test_targets=test_targets, val_targets_nan_mask=val_targets_nan_mask,
                                      test_targets_nan_mask=test_targets_nan_mask, dataset=dataset, loss_fn=loss_fn,
                                      metric=args.metric, print_header=print_header)


if __name__ == '__main__':
    main()
