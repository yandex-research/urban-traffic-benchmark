#!/bin/bash

LOG_DIR="experiments/baselines"


COMMAND="naive_forecast.py --methods constant per-node-constant prev-latest prev-periodic --constants mean median 0 --per-node-constants mean median --eval_max_num_predictions_per_step 10000000 "

DATASETS_OPTIONS=(
"--dataset city_traffic_m_speed --prediction_horizon 12 --periods 12 24 48 288 2016 5032"
"--dataset city_traffic_m_volume --prediction_horizon 12 --periods 12 24 48 288 2016 5032"
"--dataset city_traffic_l_speed --prediction_horizon 12 --periods 12 24 48 288 2016 5032"
"--dataset city_traffic_l_volume --prediction_horizon 12 --periods 12 24 48 288 2016 5032"
)

DATASETS=("city_traffic_m_speed" "city_traffic_m_volume" "city_traffic_l_speed" "city_traffic_l_volume")

mkdir -p "$LOG_DIR"

for i in "${!DATASETS_OPTIONS[@]}"; do
    DATASET_PARAMS=${DATASETS_OPTIONS[i]}
    DATASET=${DATASETS[i]}

    echo "============================================="
    echo "Run naive forecasts for $DATASET"
    echo "============================================="

    CMD="python $COMMAND $DATASET_PARAMS --device cuda:0"  # specify other device if you want

    echo "Running command:"
    echo "$CMD"

    LOG_FILE="$LOG_DIR/${DATASET}.log"

    $CMD > "$LOG_FILE" 2>&1

    echo "---------------------------------------------"
done

echo "All runs are done!"
