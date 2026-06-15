#!/bin/bash
# Script to capture a single simulation observation and plot the model's first guess

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Ensure output directory exists
mkdir -p viz

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

echo "=== Capturing Model First Guess from Random Simulation Scene ==="
echo "Loading VLM and VLA Weights..."

# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=3
export NCCL_DEBUG=WARN

python3 src/first_guess.py

if [ $? -eq 0 ]; then
    echo "=== Success ==="
    echo "Result saved to: viz/first_guess.png"
    echo "Rotating video saved to: viz/first_guess.mp4"
    echo "Plot shows predicted (red) vs objects (Cube/Bucket)."
else
    echo "=== Execution Failed ==="
    exit 1
fi
