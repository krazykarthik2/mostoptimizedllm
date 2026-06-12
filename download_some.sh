#!/bin/bash
# Download a subset of BridgeData V2 for quick testing
# Downloads meta files and the first chunk only (~12GB)

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="/home/jupyter-238w1a5447/bridge_v2_data"

echo "=== Downloading Subset of BridgeData V2 (meta + chunk-000) ==="
# Download meta files (task instructions, etc.)
huggingface-cli download jesbu1/bridge_v2_lerobot \
    --repo-type dataset \
    --local-dir $DATA_DIR \
    --local-dir-use-symlinks False \
    --include "meta/*"

# Download the first data chunk (parquet files and videos)
huggingface-cli download jesbu1/bridge_v2_lerobot \
    --repo-type dataset \
    --local-dir $DATA_DIR \
    --local-dir-use-symlinks False \
    --include "data/chunk-000/*" "videos/chunk-000/*"

echo "=== Subset Download Complete ==="
echo "Data stored in: $DATA_DIR"
