# Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks

# Installation
To install all packages, you need to install `conda` package manager. Then, run the following commands: 

```{bash}
export DGLBACKEND=pytorch  # set default backed for DGL to torch

# installation via mamba/conda
conda create -n graph_ml python==3.11 -y && conda activate graph_ml && \
pip install --no-cache-dir torch_geometric && \
pip install --no-cache-dir torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html && \
pip install -r requirements.txt

```

# Datasets

## Access to `city-traffic-M/L`
1) To access the datasets, download the files from Kaggle -- [datasets page](https://www.kaggle.com/datasets/mightyneghbor/city-traffic-benchmarks/data).
2) After download, place the files in the `data` directory

**NOTE**: Croissant file for the dataset is available in `croissant` directory.

## Datasets specifications

The `data` directory is be used to keep datasets for this project. If you provide a dataset name (rather than a file path)
to the `--dataset` command line ergument of the `run_single_experiment.py` script, it will look for the dataset in this directory.

Each dataset is stored in a `.npz` file (NumPy's compressed zipped archive 
containing objects &mdash; typically, numpy arrays &mdash; and their names).

To use a dataset with our code, the following objects should be present in the `.npz` file under corresponding names (they are presented in `city-traffic-M/L`):

- **`unix_timestamps`**: an array of shape `[num_timestamps]` with corresponding UTC-timestamp value for each timestamp.

- **`targets`**: an array of shape `[num_timestamps, num_nodes]` containing the time series values to be predicted
for each node (the dependent/endogeneous variables). Currently, only one time series per node is supported
(i.e., a time series in each node is univariate). This array may contain `NaN` values.

- **`train_timestamps`**, **`val_timestamps`**, **`test_timestamps`**: data splits &mdash; three 1-dimensional arrays
containing indices of timestamps corresponding to the train, val, and test splits respectively.
It is expected that the indices in each array are ordered and that
`len(train_timestamps) + len(val_timestamps) + len(test_timestamps) == num_timestamps`.
Even if some timestamps contain only `NaN` target values, do not drop them &mdash; this will be handled by
the data preprocessing code.

- **`spatial_node_features`**: an array of shape `[1, num_nodes, spatial_features_dim]`
containing spatial features &mdash; the features that do not change over time but are different for different nodes,
see the **Spatial, temporal, and spatiotemporal node features** section below for details.
Note that `spatial_features_dim` can be zero (if there are no spatial features in the dataset).

- **`spatial_node_feature_names`**: a 1-dimensional array of length `spatial_features_dim` containing
a name for each spatial feature, see the **Node feature names** section below for details.

- **`temporal_node_features`**: an array of shape `[num_timestamps, 1, temporal_features_dim]`
containing temporal node features &mdash; the features that change over time but are shared between all nodes,
see the **Spatial, temporal, and spatiotemporal node features** section below for details.
Note that `temporal_features_dim` can be zero (if there are no temporal features in the dataset).

- **`temporal_node_feature_names`**: a 1-dimensional array of length `temporal_features_dim` containing
a name for each temporal feature, see the **Node feature names** section below for details.

- **`spatiotemporal_node_features`**: an array of shape `[num_timestamps, num_nodes, spatiotemporal_features_dim]`
containing spatiotemporal features &mdash; the features that change over time and are different for different nodes. Note that `spatiotemporal_features_dim` can be zero (if there are no spatiotemporal features in the dataset).

- **`spatiotemporal_node_feature_names`**: a 1-dimensional array of length `spatiotemporal_features_dim` containing
a name for each spatiotemporal feature, see the **Node feature names** section below for details.

- **`num_feature_names`**: a 1-dimensional array of length `num_features_dim`
containing the names of numerical features.
Note that `num_features_dim` can be zero (if there are no numerical features in the dataset).

- **`bin_feature_names`**: a 1-dimensional array of length `bin_features_dim`
containing the names of binary features.
Note that `bin_features_dim` can be zero (if there are no binary features in the dataset).

- **`cat_feature_names`**: a 1-dimensional array of length `cat_features_dim`
containing the names of categorical features.
See the **Numerical, binary, and categorical node features** section below for more details.
Note that `cat_features_dim` can be zero (if there are no categorical features in the dataset).

- **`edges`**: an array of shape `[num_edges, 2]` containing directed edges of the graph in the edge list format.
If your graph is undirected, duplicate each edge (i.e., put both `(u, v)` and `(v, u)` edges in the array).

  **`unix_timestamps`**: a 1-dimensional array of length `num_timestamps` containing the Unix time of each timestamp.
It will be used to automatically create time-based temporal features such as day of week.

Other optional objects and metadata can be added to the `.npz` dataset files as well. For example, in all our datasets,
we also provide the following objects: `num_timestamps` (of type `int`) and `num_nodes` (of type `int`)
(if not provided, these two values are inferred from the `targets` array shape),
`first_timestamp_datetime` (of type `datetime.datetime`), `last_timestamp_datetime` (of type `datetime.datetime`),
`timestamp_frequency` (of type `datetime.timedelta`) (these can be used to infer the time of any timestamp),
`deepwalk_node_embeddings` (these embeddings can be used as additional node features).

### Spatial, temporal, and spatiotemporal node features

In general, all features can be stored as an array of shape `[num_timestamps, num_nodes, features_dim]`. However, this might be highly inefficient, as there are often
features that are static (do not change over time), or features that are dynamic (change over time) but are
the same for all nodes. Thus, for efficiency, we divide all node features into three types:
**spatial features**, **temporal features**, and **spatiotemporal features**. Different types of storage are used
for these three types of features.

**Spatiotemporal features** is the most general feature type. It is used for features that change over time and
are different for every node at a fixed timestamp. There is no trivial way to store them more efficiently,
so we store them as an array of shape `[num_timestamps, num_nodes, spatiotemporal_features_dim]`. These  characteristics change over time and are different in different regions of space (which can
be modeled as graph nodes). Examples of such characteristics are traffic flow or traffic speed for a node in a graph for a given timestamp. 

**Spatial features** are features that do not change over time (are static), but are different for every node.
We store them as an array of shape `[1, num_nodes, spatial_features_dim]`, thus compressing the time dimension
of the array. These features are different for for every node, but do not change over time. Examples of such features are coordinates of traffic sensors or starts and ends of roads, as well as
static features of roads such as speed limit, which can be used in traffic modeling. 

**Temporal features** are features that are the same for all nodes at a fixed timestamp but change over time
(are dynamic). We store them as an array of shape `[num_tiemstamps, 1, temporal_features_dim]`, thus compressing
the spatial dimension of the array. Examples of such features are hour/day/month features that can be used in
weather modeling, traffic modeling, or demand modeling, as well as holiday indicators that can be used in
traffic modeling or demand modeling. These features change over time, but have the same value for all nodes at
a fixed timestamp. **Note**: we do not store temporal features for our datasets, instead, they are derived dynamically from the `unix_timestamp` field during training.


## Add your datasets
If you want to add custom dataset, ensure that your `.npz` file contains all fields described above.

# Launch experiments

To launch experiments on city-traffic-M/L datasets, you can use these snippets, according to their names in Kaggle:
```{bash}
CITY_TRAFFIC_L_VOLUME=city_traffic_l_volume
CITY_TRAFFIC_L_SPEED=city_traffic_l_speed
CITY_TRAFFIC_M_VOLUME=city_traffic_m_volume
CITY_TRAFFIC_M_SPEED=city_traffic_m_speed
```

Choose the dataset (for instance, let's choose `city-traffic-M-speed`):
```
DATASET=$CITY_TRAFFIC_M_SPEED
```


### Train models

Here is the example of launching main experiment:

```{bash}
python run_single_experiment.py \
    --name test_run
    --dataset $DATASET \
    --metric MAE \
    --prediction_horizon 12 \
    --direct_lookback_num_steps 48 \
    --add_nan_indicators_to_targets_for_features \
    --model_class SingleInputGNN \
    --neighborhood_aggregation MeanAggr \
    --device cuda:0

```

To get full list of parameters, run:
```{bash}
python run_single_experiment.py -h
```

### Naive baselines

Here is the example of launching naive baselines on the dataset:

```{bash}
python naive_forecast.py \
    --dataset $DATASET \
    --metric MAE \
    --prediction_horizon 12 \
    --methods constant per-node-constant \
    --constants mean median \
    --per-node-constants mean median 
```


To get full list of parameters for naive baselines, run:
```
python naive_forecast.py -h
```

### Reproduce experiments
To reproduce experiments from the paper, you can run two commands:

- Reproduce main models:
  
```{bash}
bash reproduce_experiments_models.sh
```

- Reproduce naive baseline methods:

```{bash}
bash reproduce_experiments_naive_baselines.sh
```
