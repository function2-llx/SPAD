#!/bin/bash
# Launch UCPT training on homogeneous GPU nodes.
# Usage:
#   pixi run -e <env> scripts/ucpt/launch.sh [train.py args...]
# Auto-resume: train.py resumes from <save_dir>/latest-run/checkpoint-latest.pt
# when present and continues the same wandb run.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$(uname -m)" in
    x86_64)
        CACHE_ARCHIVE=outputs/ucpt/compile-cache/v4.tar.zst
        ;;
    aarch64|arm64)
        CACHE_ARCHIVE=outputs/ucpt/compile-cache/v4-arm.tar.zst
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

# All nodes must have the same visible GPU count and model.
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
RDZV_ADDR=${MASTER_ADDR:-127.0.0.1}
RDZV_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( NPROC_PER_NODE < 1 )); then
    echo "UCPT training requires at least one visible GPU" >&2
    exit 1
fi
GLOBAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

UCPT_PREFLIGHT_WORLD_SIZE=$GLOBAL_WORLD_SIZE \
    python -u -m pumit.ucpt.train.preflight "$@"

echo "Launching UCPT: arch=$(uname -m) pixi_env=${PIXI_ENV:-unknown} nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE world_size=$GLOBAL_WORLD_SIZE"

# The compilation cache lives in tmpfs and is archived for reuse after restarts.
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$RDZV_ADDR" \
    --master_port="$RDZV_PORT" \
    -m pumit.ucpt.train \
    --cache-archive "$CACHE_ARCHIVE" \
    "$@"
