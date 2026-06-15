#!/bin/bash
# Training Pipeline Script for SmolVLA
# Preprocesses the currently available data and starts training.

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="/home/jupyter-238w1a5447/bridge_v2_data"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"

cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export NCCL_IGNORE_NVML=1
export NCCL_NVLS_DISABLE=1

echo "=== Stage 1: Preprocessing (Extracting SigLIP Embeddings) ==="
# This skips episodes that are already processed.
python3 src/dataset_prep.py --data_root $DATA_DIR --output_dir $PROCESSED_DIR

echo "=== Stage 2: Launching Training ==="
./scripts/run_train.sh --full
