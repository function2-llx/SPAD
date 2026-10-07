#!/bin/bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: $0 <checkpoint-dir> <output-root> <step> [<step> ...]" >&2
    exit 2
fi

checkpoint_dir=$1
output_root=$2
shift 2

for step in "$@"; do
    checkpoint="$checkpoint_dir/checkpoint-$step.pt"
    output_dir="$output_root/full-step-$step-stitch-first"
    rows=("$output_dir"/*-ema.jsonl)
    summaries=("$output_dir"/*-ema-summary.json)

    if (( ${#rows[@]} == 1 && ${#summaries[@]} == 1 )) && [[ -f "${rows[0]}" && -f "${summaries[0]}" ]]; then
        echo "[seg-eval-series] step $step already complete"
        continue
    fi
    if compgen -G "$output_dir/*-ema.jsonl" >/dev/null || compgen -G "$output_dir/*-ema-summary.json" >/dev/null; then
        echo "[seg-eval-series] incomplete final outputs for step $step in $output_dir" >&2
        exit 1
    fi

    echo "[seg-eval-series] evaluating step $step"
    python scripts/ucpt/eval_seg.py \
        --checkpoint "$checkpoint" \
        --output-dir "$output_dir" \
        --interpolation-order stitch-then-interpolate
done
