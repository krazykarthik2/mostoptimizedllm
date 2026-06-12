#!/bin/bash
# Shortcut to run the randomized environment simulation
# Records 5 videos in the viz/ directory

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Ensure output directory exists
mkdir -p viz

# Add project root to PYTHONPATH so 'src' can be imported
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

echo "=== Starting Randomized Robot Simulation ==="
echo "Output will be saved to: $PROJECT_ROOT/viz/"
echo "This might take a minute to load models..."

python3 src/sim_viz.py

echo "=== Simulation Finished ==="
