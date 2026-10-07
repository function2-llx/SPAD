#!/usr/bin/env bash
# Launch one SPAD U-Net nnU-Net run. Completed runs are skipped, interrupted ones resume.
#
#   launch.sh Dataset591_SPADCTUniversalV2 SPADPlannedZSumGB8Plans \
#       --trainer SPADUniversalTrainer --record-dir experiment-records/spad-unet/<run>
#   launch.sh Dataset591_SPADCTUniversalV2 CorpusGridZ1GB8Plans \
#       --trainer CorpusGridUniversalTrainer --record-dir experiment-records/spad-unet/<run>
#   launch.sh Dataset517_BTCV-region nnUNetResEncUNetLPlans1p5x1x1FOV192 \
#       --trainer nnUNetTrainerRetainCheckpoints --configuration 3d_fullres_specialist_b4
set -euo pipefail

usage() {
    printf 'usage: %s DATASET PLAN --trainer NAME [--configuration NAME] [--fold N]\n' "$0" >&2
    printf '       [--record-dir PATH] [--inference-mode {floor|ceil|cross|endpoints}]\n' >&2
    printf '       [--tile-step-size FLOAT] [--sliding-window-batch-size INT]\n' >&2
    printf '       [--case-evaluation-workers INT] [--gpus LOCAL_N]\n' >&2
}

if (( $# < 2 )); then
    usage
    exit 2
fi

dataset=$1
plan=$2
shift 2
trainer=
configuration=3d_fullres
fold=0
record_dir=
inference_mode=
tile_step_size=
sliding_window_batch_size=
case_evaluation_workers=
num_local_ranks=
while (( $# )); do
    case "$1" in
        --trainer|--configuration|--fold|--record-dir|--inference-mode|--tile-step-size|--sliding-window-batch-size|--case-evaluation-workers|--gpus)
            if (( $# < 2 )); then
                printf '%s requires a value\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --trainer) trainer=$2 ;;
                --configuration) configuration=$2 ;;
                --fold) fold=$2 ;;
                --record-dir) record_dir=$2 ;;
                --inference-mode) inference_mode=$2 ;;
                --tile-step-size) tile_step_size=$2 ;;
                --sliding-window-batch-size) sliding_window_batch_size=$2 ;;
                --case-evaluation-workers) case_evaluation_workers=$2 ;;
                --gpus) num_local_ranks=$2 ;;
            esac
            shift 2
            ;;
        *)
            printf 'unexpected argument %q\n' "$1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! $dataset =~ ^Dataset[0-9]{3}_.+$ ]]; then
    printf 'dataset must be a full nnU-Net name such as Dataset591_SPADCTUniversalV2: %s\n' "$dataset" >&2
    exit 2
fi
if [[ -z $plan || $plan == *__* ]]; then
    printf 'plan must be a valid nnU-Net plans identifier: %s\n' "$plan" >&2
    exit 2
fi
if [[ -z $configuration ]]; then
    printf 'configuration must not be empty\n' >&2
    exit 2
fi
if [[ ! $fold =~ ^[0-9]+$ ]]; then
    printf 'fold must be a non-negative integer: %s\n' "$fold" >&2
    exit 2
fi
if [[ -n $num_local_ranks && ! $num_local_ranks =~ ^[1-9][0-9]*$ ]]; then
    printf 'gpus must be a positive integer: %s\n' "$num_local_ranks" >&2
    exit 2
fi

case "$trainer" in
    SPADUniversalTrainer|CorpusGridUniversalTrainer|nnUNetTrainerRetainCheckpoints)
        ;;
    '')
        printf '%s\n' '--trainer is required' >&2
        usage
        exit 2
        ;;
    *)
        printf 'unsupported trainer: %s\n' "$trainer" >&2
        exit 2
        ;;
esac

case "$inference_mode" in
    ''|floor|ceil|cross|endpoints)
        ;;
    *)
        printf 'unsupported inference mode: %s\n' "$inference_mode" >&2
        exit 2
        ;;
esac
if [[ $trainer != SPADUniversalTrainer \
    && -n $inference_mode \
    && $inference_mode != cross ]]; then
    printf '%s does not read SPAD_UNIVERSAL_INFERENCE_MODE\n' "$trainer" >&2
    exit 2
fi
if [[ $trainer != SPADUniversalTrainer && -n $tile_step_size ]]; then
    printf '%s does not read SPAD_UNIVERSAL_TILE_STEP_SIZE\n' "$trainer" >&2
    exit 2
fi
if [[ $trainer != SPADUniversalTrainer && -n $sliding_window_batch_size ]]; then
    printf '%s does not read SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE\n' "$trainer" >&2
    exit 2
fi
if [[ $trainer != SPADUniversalTrainer && -n $case_evaluation_workers ]]; then
    printf '%s does not read SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS\n' "$trainer" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export nnUNet_compile=true
export nnUNet_results="$repo_root/nnUNet_data/results"
export PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY="${PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY:-5}"

if [[ -n $record_dir ]]; then
    mkdir -p "$record_dir"
    record_dir=$(cd -- "$record_dir" && pwd)
fi

plans_file="$repo_root/nnUNet_data/preprocessed/$dataset/$plan.json"
if [[ ! -f "$plans_file" ]]; then
    printf 'missing plans: %s\n' "$plans_file" >&2
    exit 1
fi
if [[ -z $inference_mode ]]; then
    if [[ $trainer == SPADUniversalTrainer ]]; then
        inference_mode=$(python -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["pumit_spad_unet"].get("inference_mode", "cross"))' \
            "$plans_file")
    else
        inference_mode=cross
    fi
fi
case "$inference_mode" in
    floor|ceil|cross|endpoints) ;;
    *)
        printf 'plans declare unsupported inference mode: %s\n' "$inference_mode" >&2
        exit 2
        ;;
esac

if [[ $trainer == SPADUniversalTrainer ]]; then
    tile_step_size=${tile_step_size:-0.5}
    requested_tile_step_size=$tile_step_size
    if ! tile_step_size=$(python -c '
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0 < value <= 1:
    raise ValueError(f"tile step size must be in (0, 1], got {value!r}")
print(repr(value))
' "$tile_step_size"); then
        printf 'invalid tile step size: %s\n' "$requested_tile_step_size" >&2
        exit 2
    fi
    sliding_window_batch_size=${sliding_window_batch_size:-1}
    if [[ ! $sliding_window_batch_size =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid sliding-window batch size: %s\n' "$sliding_window_batch_size" >&2
        exit 2
    fi
    case_evaluation_workers=${case_evaluation_workers:-4}
    if [[ ! $case_evaluation_workers =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid case-evaluation worker count: %s\n' "$case_evaluation_workers" >&2
        exit 2
    fi
fi

case "$trainer" in
    SPADUniversalTrainer)
        export SPAD_UNIVERSAL_INFERENCE_MODE="$inference_mode"
        export SPAD_UNIVERSAL_TILE_STEP_SIZE="$tile_step_size"
        export SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE="$sliding_window_batch_size"
        export SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS="$case_evaluation_workers"
        # The trainer hashes the archive path into its own /dev/shm subdirectory, so the root stays shared.
        if [[ -n $record_dir ]]; then
            case "$(uname -m)" in
                aarch64) cache_arch=arm ;;
                x86_64) cache_arch=x86 ;;
                *)
                    printf 'unsupported compile-cache architecture: %s\n' "$(uname -m)" >&2
                    exit 1
                    ;;
            esac
            export SPAD_UNIVERSAL_COMPILE_CACHE_ARCHIVE="$record_dir/spad-torchinductor-runtime-cache-${cache_arch}.tar.zst"
        fi
        ;;
    CorpusGridUniversalTrainer)
        if [[ -n $record_dir ]]; then
            export TORCHINDUCTOR_CACHE_DIR="$record_dir/corpus-torchinductor-cache-arm"
            export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
        fi
        ;;
    nnUNetTrainerRetainCheckpoints)
        # Lives in the downstream/seg extension directory, not the spad-unet one the pixi env points at.
        export nnUNet_extTrainer="$repo_root/src/pumit/downstream/seg/nnunet_ext"
        ;;
esac

visible_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( visible_gpus < 1 )); then
    printf '%s\n' 'SPAD U-Net training requires at least one visible GPU' >&2
    exit 1
fi
if [[ -z $num_local_ranks ]]; then
    num_local_ranks=$visible_gpus
elif (( num_local_ranks > visible_gpus )); then
    printf 'requested %s local DDP ranks, but only %s GPUs are visible\n' \
        "$num_local_ranks" "$visible_gpus" >&2
    exit 1
fi
printf 'launching SPAD U-Net: local_ranks=%s\n' "$num_local_ranks"

output_folder="$nnUNet_results/$dataset/${trainer}__${plan}__${configuration}/fold_${fold}"
# SPADUniversalTrainer names its output folder after the inference mode; the others use nnU-Net's default.
case "$trainer" in
    SPADUniversalTrainer)
        validation_dir="validation_${inference_mode}"
        if [[ $tile_step_size != 0.5 ]]; then
            validation_dir+="_step${tile_step_size}"
        fi
        if [[ $sliding_window_batch_size != 1 ]]; then
            validation_dir+="_batch${sliding_window_batch_size}"
        fi
        ;;
    *) validation_dir=validation ;;
esac
# checkpoint_final.pth predates the validation summary and is not a completion marker.
summary_file="$output_folder/$validation_dir/summary.json"
if [[ -f "$summary_file" ]]; then
    printf 'already complete: %s\n' "$summary_file"
    exit 0
fi

checkpoint_args=()
if [[ -f "$output_folder/checkpoint_final.pth" ]]; then
    checkpoint_args=(--val)
    # SPAD full-volume inference unwraps the compiled network and runs an eager predictor.
    # Avoid compiling and warming every training path before validation-only execution.
    if [[ $trainer == SPADUniversalTrainer ]]; then
        export nnUNet_compile=false
    fi
elif [[ -f "$output_folder/checkpoint_latest.pth" \
    || -f "$output_folder/checkpoint_best.pth" ]]; then
    checkpoint_args=(--c)
fi

exec nnUNetv2_train \
    "$dataset" "$configuration" "$fold" \
    -tr "$trainer" \
    -p "$plan" \
    -num_gpus "$num_local_ranks" \
    "${checkpoint_args[@]}"
