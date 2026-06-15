#!/bin/bash
# Visualization Script for SmolVLA
# Picks a random sample from the canonical dataset and compares trajectories

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Ensure output directory exists
mkdir -p viz

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

CHECKPOINT="robotmodel/models/checkpoints/latest.pt"
DATA_DIR="data/processed"
OUTPUT="viz/random_evaluation"

echo "=== Running 3D Trajectory Visualization (Canonical) ==="
echo "Checkpoint: $CHECKPOINT"
echo "Data Dir: $DATA_DIR"
echo "Loading VLM Backbone and VLA Weights..."

# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=3
export NCCL_DEBUG=WARN

python3 src/visualize_random.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "=== Success ==="
    echo "Static plot: ${OUTPUT}.png"
    echo "Rotating video: ${OUTPUT}.mp4"
else
    echo "=== Visualization Failed ==="
    exit 1
fi
