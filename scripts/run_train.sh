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
export NCCL_IGNORE_NVML=1
export NCCL_NVLS_DISABLE=1
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=0 # Keep SHM but disable P2P
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Attempting to launch training on 2 GPUs using torchrun..."
# Switch to torchrun for more direct distributed management
torchrun --nproc_per_node 2 src/train.py --full "$@"

if [ $? -ne 0 ]; then
    echo "Multi-GPU launch failed (NCCL/Driver error)."
    echo "Falling back to single-GPU training..."
    python3 src/train.py --full "$@"
fi
