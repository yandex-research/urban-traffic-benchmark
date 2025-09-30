__doc__ = "This scripts prepares raw data from framework-specific `.npz` files"


from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def save_with_fallback(df: pd.DataFrame, out_base: Path):
    """
    Try Parquet first (smaller, preserves dtypes).
    If pyarrow/fastparquet isn't installed, fall back to CSV.
    Returns the final file path.
    """
    try:
        fp = out_base.with_suffix(".parquet")
        # Keep index (timestamps, etc.)
        df.to_parquet(fp)
        return fp
    except Exception as e:
        print(f"Caught exception: {e}")
        fp = out_base.with_suffix(".csv")
        df.to_csv(fp)
        return fp

def shift_coords(spatial_features):
    x_coordinate_start, y_coordinate_start, x_coordinate_end, y_coordinate_end = spatial_features[0, :, -4:].T

    x_mean_roadwise = (x_coordinate_start + x_coordinate_end) / 2
    y_mean_roadwise = (y_coordinate_start + y_coordinate_end) / 2

    mean_in_dataset_x = x_mean_roadwise.mean()
    mean_in_dataset_y = y_mean_roadwise.mean()

    spatial_features[0, :, -4] -= mean_in_dataset_x
    spatial_features[0, :, -2] -= mean_in_dataset_x

    spatial_features[0, :, -3] -= mean_in_dataset_y
    spatial_features[0, :, -1] -= mean_in_dataset_y

    return spatial_features


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--dataset_prefix", help="Prefix will be added to all files at Kaggle")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("file_speed")
    parser.add_argument("file_volume")
    parser.add_argument("features_path")

    args = parser.parse_args()

    dataset_speed = dict(np.load(args.file_speed, allow_pickle=True))
    dataset_volume = dict(np.load(args.file_volume, allow_pickle=True))

    speed_targets = dataset_speed["targets"]
    volume_targets = dataset_volume["targets"]

    spatial_features_new = pd.read_csv(args.features_path, index_col=0).values[None, ...]

    spatial_features = dataset_speed["spatial_node_features"][0]


    timestamps_datetime = dataset_speed["unix_timestamps"].astype("datetime64[s]")

    static_df = pd.DataFrame(spatial_features, columns=dataset_speed["spatial_node_feature_names"])
    categorical_columns_mask = [col in dataset_speed["cat_feature_names"] for col in static_df.columns]
    binary_columns_mask = [col in dataset_speed["bin_feature_names"] for col in static_df.columns]

    static_df.loc[:, categorical_columns_mask] = static_df.loc[:, categorical_columns_mask].astype(int)
    static_df.loc[:, binary_columns_mask] = static_df.loc[:, binary_columns_mask].astype(bool)


    print("Created static features df")

    edges_df = pd.DataFrame(dataset_speed["edges"], columns=["source", "target"])

    print("Created edges df")

    speed_df = pd.DataFrame(speed_targets, index=timestamps_datetime)
    speed_df.columns = [f"node_{i}" for i in range(speed_df.shape[1])]
    speed_df.index.name = "timestamp"

    print("created speed df")
    volume_df = pd.DataFrame(volume_targets, index=timestamps_datetime)
    volume_df.index.name = "timestamp"
    volume_df.columns = [f"node_{i}" for i in range(volume_df.shape[1])]

    print("created volume df")

    print("DataFrames ready: static_df, edges_df, speed_df, volume_df")


    outdir = Path(args.outdir)
    out_static = save_with_fallback(static_df, outdir / f"{args.dataset_prefix}_static_features")
    out_edges  = save_with_fallback(edges_df,  outdir / f"{args.dataset_prefix}_edges")
    out_speed  = save_with_fallback(speed_df,  outdir / f"{args.dataset_prefix}_speed")
    out_volume = save_with_fallback(volume_df, outdir) / f"{args.dataset_prefix}_volume")

    # staged = [out_static, out_edges, out_speed, out_volume]
    staged = [out_static]
    print("Staged files:")
    for p in staged:
        print("  -", p.name, f"({p.stat().st_size/1e6:.2f} MB)")


    np.savez_compressed(outdir / Path(args.file_speed).name, **dataset_speed)
    print("Saved speed with shifted coords")

    np.savez_compressed(outdir / Path(args.file_volume).name, **dataset_volume)
    print("Saved volume with shifted coords")
