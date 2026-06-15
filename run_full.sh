#!/bin/bash
# Full Pipeline Script for SmolVLA
# Runs from start to end assuming data is already downloaded.
# Does NOT download any data.

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="/home/jupyter-238w1a5447/bridge_v2_data"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"

cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export NCCL_IGNORE_NVML=1
export NCCL_NVLS_DISABLE=1

echo "=== Starting Full Training Pipeline (No Download) ==="

echo "=== Stage 1: Preprocessing Full Dataset ==="
# Process all available data in DATA_DIR
python3 src/dataset_prep.py --data_root $DATA_DIR --output_dir $PROCESSED_DIR

echo "=== Stage 2: Training on Full Processed Data ==="
./scripts/run_train.sh --full

echo "=== Full Pipeline Complete ==="
