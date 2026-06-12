#!/bin/bash
# Visualization Script for SmolVLA
# Picks a random sample and compares trajectories in 3D

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

CHECKPOINT="robotmodel/models/checkpoints/latest.pt"
DATA_DIR="data/processed"
OUTPUT="viz/random_evaluation"

echo "=== Running 3D Trajectory Visualization ==="
export TF_CPP_MIN_LOG_LEVEL=3
export NCCL_DEBUG=WARN
python3 src/visualize_random.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT"

echo "=== Done ==="
echo "Static plot: ${OUTPUT}.png"
echo "Rotating video: ${OUTPUT}.mp4"
