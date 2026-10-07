# SPAD

SPAD provides spatially adaptive operators, a Vision Transformer, and U-Net models for medical images.
This source snapshot includes model code, training, runtime data pipelines, downstream evaluation, configurations, and tests.
The Python package remains named `pumit` to preserve existing imports and checkpoint integration.

## Source layout

| Directory | Contents |
|---|---|
| `src/pumit/spadop` | Spatially adaptive operators |
| `src/pumit/model` | Vision Transformer and 3D rotary position embedding |
| `src/pumit/spad_unet` | U-Net architectures, trainers, and runtime data loading |
| `src/pumit/ucpt` | Continued pretraining, objectives, and replay data pipeline |
| `src/pumit/data`, `src/pumit/transforms` | Sampling, loading, augmentation, and batching |
| `src/pumit/downstream` | Downstream models, training, and evaluation |
| `third_party/nnUNet` | Adapted nnU-Net implementation |
| `configs`, `tests` | Model and training configurations, regression tests |

## Data boundary

Raw dataset download, dataset-specific preprocessing, and historical data migration scripts are not included.
Legacy dataset migration is omitted from the stream-generation tools.
Training retains its existing prepared-data contracts.
The pretraining replay loader consumes stream metadata and shards, latent tensors, masks, text embeddings, and fingerprint metadata.
Stream generation and latent encoding tools are included; they operate on prepared data rather than raw dataset files.
It does not convert arbitrary raw medical images into those artifacts.
U-Net training consumes nnU-Net datasets and plans.

Training-input tools:

- `python -m pumit.ucpt.stream --help`: stream building, composition, latent encoding, statistics, and verification.
- `scripts/ucpt/materialize_latents.sh`: latent materialization for a finalized stream; set `CODEC_CHECKPOINT` explicitly.
- `scripts/codec/gen_batches.py`: codec training stream generation.
- `scripts/downstream/spad_unet/gen_replay_stream.py` and `gen_sqrt_replay_stream.py`: U-Net replay generation.

Data, weights, caches, experiment records, task scheduling tools, and experiment-specific job generators and reporting scripts are not included.
Paths and logging accounts must be supplied for the environment where the code is used.

## Preparation status

This is a source-code reference, not a turnkey experiment reproduction package.
nnU-Net training supports a single machine with one or more GPUs; its launcher does not provide multi-node execution.
The environment configuration and lockfile are dependency references; a clean installation has not been validated.

The project license has not been selected.
Bundled third-party sources retain their existing attribution and license notices; redistribution review is not complete.
