"""Explore feature consistency across simulated anisotropy levels.

Measures how stable CLS embeddings are when the same volume is viewed at different
simulated anisotropy levels (DA=0 -> DA=1, DA=2 via depth downsampling).

Protocol:
1. Select qualifying samples (DA=0, depth >= 64, valid spacing)
2. Crop each to [C, 64, 128, 128] deterministically
3. Simulate DA=1: avg_pool3d(kernel=(2,1,1)) -> depth=32
4. Simulate DA=2: avg_pool3d(kernel=(4,1,1)) -> depth=16
5. For SPAD-ViT: encode downsampled at its DA level
6. For Standard 3D ViT (--no-da): repeat_interleave back to depth=64, encode at da=0
7. Report cosine similarity (mean + 95% CI) and self-retrieval accuracy

Usage:
    pixi run -e default python scripts/downstream/spatial/eval_anisotropy_consistency.py \
        --encoder-ckpt outputs/legacy/ssl/runs/ssl/checkpoint-final.pt \
        --checkpoint-format ucpt --normalization pumit \
        --n-samples 1000 --device cuda
"""

import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pumit.codec.config import MAX_DA
from pumit.data import build_training_data, compute_da
from pumit.downstream.spatial.encoder import (
    CHECKPOINT_FORMATS,
    NORMALIZATIONS,
    load_encoder,
    normalize_volumes,
)
from pumit.model.vit import ViT


# --- Sample selection ---


def select_qualifying_samples(
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Select qualifying samples: DA=0, valid spacing, depth >= 64.

    Returns list of dicts with keys: img_path, shape, spacing.
    """
    # Load training data (no depth tier filtering, no weight fn needed)
    train_data, _, _ = build_training_data(
        weight_fn=None, depth_tiers=None, verbose=False, max_da=MAX_DA,
    )

    qualifying = []
    for idx, row in train_data.iterrows():
        shape = row['shape']  # (D, H, W) spatial
        spacing = row['spacing']  # np.ndarray of 3

        # Must be 3D with depth >= 64
        depth = shape[0]
        if depth < 64:
            continue

        # Spacing must be all valid (no NaN)
        if np.any(np.isnan(spacing)):
            continue

        # Must be DA=0
        da = compute_da(spacing, depth, max_da=MAX_DA)
        if da != 0:
            continue

        qualifying.append({
            'img_path': row['img'],
            'shape': shape,
            'spacing': spacing,
        })

    if len(qualifying) < n_samples:
        raise ValueError(
            f'only {len(qualifying)} qualifying samples found, requested {n_samples}'
        )

    # Random subset
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(qualifying), size=n_samples, replace=False)
    selected = [qualifying[i] for i in indices]
    print(f"Selected {len(selected)} samples from {len(qualifying)} qualifying (seed={seed})")
    return selected


# --- Crop and encode ---


def deterministic_crop(vol: torch.Tensor, depth: int = 64, h: int = 128, w: int = 128) -> torch.Tensor:
    """Center-crop a volume to [C, depth, h, w].

    If dimension is smaller than target, this will fail (caller ensures depth >= 64).
    For H, W: center crop if large enough, else pad.
    """
    # vol: [C, D, H, W]
    _, d, vh, vw = vol.shape

    # Depth: center crop
    d_start = (d - depth) // 2
    vol = vol[:, d_start:d_start + depth, :, :]

    # Height: center crop or pad
    if vh >= h:
        h_start = (vh - h) // 2
        vol = vol[:, :, h_start:h_start + h, :]
    else:
        pad_h = h - vh
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        vol = F.pad(vol, (0, 0, pad_top, pad_bot))

    # Width: center crop or pad
    if vw >= w:
        w_start = (vw - w) // 2
        vol = vol[:, :, :, w_start:w_start + w]
    else:
        pad_w = w - vw
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        vol = F.pad(vol, (pad_left, pad_right))

    return vol


def encode_batch(
    encoder: ViT,
    volumes: torch.Tensor,
    da: int,
    normalization: str,
) -> torch.Tensor:
    """Encode a batch of volumes, return CLS tokens.

    Args:
        encoder: frozen ViT
        volumes: [B, C, D, H, W] float tensor
        da: depth adaptation level
        normalization: Evaluated model input normalization.

    Returns:
        CLS tokens [B, embed_dim]
    """
    x = normalize_volumes(volumes, normalization)
    autocast = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if volumes.device.type == 'cuda'
        else nullcontext()
    )
    with autocast:
        cls, _ = encoder.encode_image(x, da=da)
    return cls.float()


# --- Main evaluation ---


def compute_metrics(
    cos_sims: list[float],
) -> dict[str, float]:
    """Compute mean and 95% CI for cosine similarities."""
    arr = np.array(cos_sims)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = len(arr)
    ci_95 = 1.96 * std / np.sqrt(n)
    return {'mean': float(mean), 'std': float(std), 'ci_95': float(ci_95), 'n': n}


def compute_self_retrieval(
    cls_ref: torch.Tensor,
    cls_sim: torch.Tensor,
) -> float:
    """Self-retrieval accuracy: fraction where each sample's simulated embedding
    is closest to its own DA=0 embedding.

    Args:
        cls_ref: [N, D] DA=0 CLS embeddings
        cls_sim: [N, D] simulated DA CLS embeddings

    Returns:
        accuracy in [0, 1]
    """
    # NxN cosine similarity matrix
    cls_ref_normed = F.normalize(cls_ref, dim=1)
    cls_sim_normed = F.normalize(cls_sim, dim=1)
    sim_matrix = cls_sim_normed @ cls_ref_normed.T  # [N, N]
    # For each row, check if max is on diagonal
    max_indices = sim_matrix.argmax(dim=1)
    correct = (max_indices == torch.arange(len(cls_ref), device=cls_ref.device)).float()
    return correct.mean().item()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate anisotropy robustness via CLS cosine similarity'
    )
    parser.add_argument('--encoder-ckpt', type=str, required=True, help='Path to SSL checkpoint')
    parser.add_argument('--checkpoint-format', choices=CHECKPOINT_FORMATS, required=True)
    parser.add_argument('--normalization', choices=NORMALIZATIONS, required=True)
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda, cpu)')
    parser.add_argument('--no-da', action='store_true', help='Standard 3D ViT mode: repeat_interleave to restore depth, encode at da=0')
    parser.add_argument('--wandb-name', type=str, default=None, help='W&B run name (enables wandb if set)')
    parser.add_argument('--wandb-project', type=str, default='pumit-probe', help='W&B project name')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load encoder
    print(f"Loading encoder from {args.encoder_ckpt}...")
    encoder = load_encoder(
        args.encoder_ckpt,
        checkpoint_format=args.checkpoint_format,
        device=device,
    )
    print(f"  embed_dim={encoder.config.hidden_size}, layers={encoder.config.num_hidden_layers}")

    # Select qualifying samples
    samples = select_qualifying_samples(args.n_samples, args.seed)

    # Optional wandb
    wandb_run = None
    if args.wandb_name is not None:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # Encode all samples at DA=0, DA=1, DA=2
    # DA levels to simulate: (pool_factor, da_level_for_encoding)
    sim_levels = [
        (1, 0, 'DA=0 (reference)'),
        (2, 1, 'DA=1 (pool 2x)'),
        (4, 2, 'DA=2 (pool 4x)'),
    ]

    # Collect CLS embeddings per level
    cls_per_level: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}

    print(f"\nEncoding {len(samples)} samples...")
    with torch.inference_mode():
        for sample in tqdm(samples, desc='Processing samples'):
            # Load volume
            vol = torch.from_numpy(
                np.load(sample['img_path'])
            ).float()  # [C, D, H, W]

            # Deterministic crop to [C, 64, 128, 128]
            crop = deterministic_crop(vol, depth=64, h=128, w=128)
            # crop: [C, 64, 128, 128]

            for pool_factor, da_level, _ in sim_levels:
                if pool_factor == 1:
                    # Original (DA=0)
                    x = crop.unsqueeze(0).to(device)  # [1, C, 64, 128, 128]
                    cls = encode_batch(encoder, x, da=0, normalization=args.normalization)
                else:
                    # Downsample depth via avg_pool
                    downsampled = F.avg_pool3d(
                        crop.unsqueeze(0),
                        kernel_size=(pool_factor, 1, 1),
                    )  # [1, C, 64//factor, 128, 128]

                    if args.no_da:
                        # Standard 3D ViT: repeat_interleave back to original depth
                        restored = downsampled.repeat_interleave(pool_factor, dim=2)
                        x = restored.to(device)
                        cls = encode_batch(encoder, x, da=0, normalization=args.normalization)
                    else:
                        # SPAD-ViT: encode at the simulated DA level
                        x = downsampled.to(device)
                        cls = encode_batch(
                            encoder,
                            x,
                            da=da_level,
                            normalization=args.normalization,
                        )

                cls_per_level[da_level].append(cls.cpu())

    # Stack CLS embeddings
    cls_stacked: dict[int, torch.Tensor] = {}
    for level in cls_per_level:
        cls_stacked[level] = torch.cat(cls_per_level[level], dim=0)  # [N, D]

    # Compute cosine similarities: DA=0 vs DA=1, DA=0 vs DA=2
    print("\n" + "=" * 60)
    print("ANISOTROPY ROBUSTNESS RESULTS")
    if args.no_da:
        print("  Mode: Standard 3D ViT (--no-da, repeat_interleave)")
    else:
        print("  Mode: SPAD-ViT (encode at simulated DA level)")
    print("=" * 60)

    results: dict[str, dict] = {}
    for da_level in [1, 2]:
        # Per-sample cosine similarity
        cos_sims = F.cosine_similarity(
            cls_stacked[0], cls_stacked[da_level], dim=1,
        ).tolist()

        metrics = compute_metrics(cos_sims)
        retrieval_acc = compute_self_retrieval(cls_stacked[0], cls_stacked[da_level])

        results[f'da{da_level}'] = {**metrics, 'retrieval_acc': retrieval_acc}

        print(f"\n  DA=0 vs DA={da_level}:")
        print(f"    Cosine similarity: {metrics['mean']:.4f} +/- {metrics['ci_95']:.4f} (95% CI)")
        print(f"    Std: {metrics['std']:.4f}, N={metrics['n']}")
        print(f"    Self-retrieval accuracy: {retrieval_acc:.4f} ({retrieval_acc * 100:.1f}%)")

    # Wandb logging
    if wandb_run is not None:
        summary = {}
        for da_level in [1, 2]:
            r = results[f'da{da_level}']
            summary[f'robustness/da{da_level}_cos_mean'] = r['mean']
            summary[f'robustness/da{da_level}_cos_ci95'] = r['ci_95']
            summary[f'robustness/da{da_level}_retrieval_acc'] = r['retrieval_acc']
        wandb_run.summary.update(summary)
        wandb_run.finish()

    print("\nDone.")


if __name__ == '__main__':
    main()
