#!/bin/bash
# Materialize and finalize one complete UCPT latent stream.
set -euo pipefail

STREAM=${1:?Usage: $0 STREAM SOURCE_LATENT_DIR}
SOURCE_LATENT_DIR=${2:?Usage: $0 STREAM SOURCE_LATENT_DIR}
MEMORY_BUDGET_GB=${MEMORY_BUDGET_GB:-80}
REPLAY_THREADS=${REPLAY_THREADS:-8}
FINALIZE_WORKERS=${FINALIZE_WORKERS:-32}
COMPILE_MODE=${COMPILE_MODE:-default}
LATENT_FILTER=${LATENT_FILTER:-canonical-inplane}

case "$(uname -m)" in
    x86_64)
        CACHE_ARCHIVE=${CACHE_ARCHIVE:-outputs/ucpt/compile-cache/latent-flux2-${COMPILE_MODE}-static.tar.zst}
        ;;
    aarch64|arm64)
        CACHE_ARCHIVE=${CACHE_ARCHIVE:-outputs/ucpt/compile-cache/latent-flux2-${COMPILE_MODE}-arm-static.tar.zst}
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

CODEC_CHECKPOINT=${CODEC_CHECKPOINT:?Set CODEC_CHECKPOINT to the codec checkpoint path}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$STREAM/meta.yaml" ]]; then
    echo "$STREAM must be finalized before latent materialization" >&2
    exit 2
fi

SOURCE_STREAM=$(python - "$STREAM/meta.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

meta = yaml.safe_load(Path(sys.argv[1]).read_text())
print(meta['source_stream'])
PY
)
COST_MODEL=$(python - "$STREAM/meta.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

meta = yaml.safe_load(Path(sys.argv[1]).read_text())
print(meta['cost_model_source'])
PY
)

python -u -m pumit.ucpt.stream encode-latents \
    --filter "$LATENT_FILTER" \
    --stream "$STREAM" \
    --source-latent-dir "$SOURCE_LATENT_DIR" \
    --codec-checkpoint "$CODEC_CHECKPOINT" \
    --codec-model flux2 \
    --memory-budget-gb "$MEMORY_BUDGET_GB" \
    --num-workers "$REPLAY_THREADS" \
    --compile-mode "$COMPILE_MODE" \
    --cache-archive "$CACHE_ARCHIVE"

if [[ ! -f "$STREAM/latents/stats.safetensors" ]]; then
    python -u -m pumit.ucpt.stream stats \
        --stream "$STREAM" \
        --workers "$FINALIZE_WORKERS"
fi

if [[ ! -f "$STREAM/READY.json" ]]; then
    python -u -m pumit.ucpt.stream verify \
        --stream "$STREAM" \
        --source-stream "$SOURCE_STREAM" \
        --source-latent-dir "$SOURCE_LATENT_DIR" \
        --cost-model "$COST_MODEL" \
        --workers "$FINALIZE_WORKERS"
fi
