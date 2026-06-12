#!/bin/bash
# Train script for SmolVLA 4-DOF
# Optimized for 2x L4 GPUs

# Ensure we are in the project root
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Workarounds for NCCL Driver/Library mismatch errors
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

echo "Attempting to launch training on 2 GPUs..."
# Overfit is now default, so we just pass arguments (like --full if they want the whole set)
accelerate launch --multi_gpu --num_processes 2 src/train.py "$@"

if [ $? -ne 0 ]; then
    echo "Multi-GPU launch failed (likely due to NCCL/Driver mismatch)."
    echo "Falling back to single-GPU training (Default: Overfit Mode)..."
    python3 src/train.py "$@"
fi
