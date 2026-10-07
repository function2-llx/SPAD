"""Extract and cache global features from a frozen encoder for spacing evaluation.

Reads prepared crops, runs batched encoder inference grouped by DA, and saves `train.pt` and `val.pt`.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import orjson
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pumit.model.vit import ViT

from .encoder import (
    CHECKPOINT_FORMATS,
    NORMALIZATIONS,
    load_encoder,
    normalize_volumes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract CLS features from a frozen ViT encoder')
    parser.add_argument('--encoder-ckpt', type=str, required=True, help='Path to the encoder checkpoint')
    parser.add_argument(
        '--checkpoint-format',
        choices=CHECKPOINT_FORMATS,
        required=True,
        help='Checkpoint model-state namespace',
    )
    parser.add_argument(
        '--normalization',
        choices=NORMALIZATIONS,
        required=True,
        help='Evaluated model input normalization',
    )
    parser.add_argument(
        '--crops-dir',
        type=str,
        required=True,
        help='Pre-cropped volumes from scripts/downstream/spatial/prepare_spacing_crops.py',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save train.pt and val.pt',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument(
        '--no-depth-rope',
        action='store_true',
        help='Disable the depth component of 3D RoPE',
    )
    return parser.parse_args()


class PrecroppedDataset(Dataset):
    """Load pre-cropped `.npy` files."""

    def __init__(self, keys: list[str], spacings: list[list[float]], das: list[int], crops_dir: Path):
        self.keys = keys
        self.spacings = spacings
        self.das = das
        self.crops_dir = crops_dir

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = np.load(self.crops_dir / f'{key}.npy')
        spacing = np.array(self.spacings[idx], dtype=np.float32)
        da = self.das[idx]
        return torch.from_numpy(img).float(), torch.from_numpy(spacing), da


def process_split(
    keys: list[str],
    spacings: list[list[float]],
    das: list[int],
    crops_dir: Path,
    encoder: ViT,
    device: str,
    normalization: str,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> dict[str, torch.Tensor]:
    """Batched inference grouped by DA on pre-cropped data."""
    dataset = PrecroppedDataset(keys, spacings, das, crops_dir)

    # Group by DA
    da_groups: dict[int, list[int]] = {}
    for i, da in enumerate(das):
        da_groups.setdefault(da, []).append(i)

    cls_list: list[torch.Tensor] = []
    spacing_list: list[torch.Tensor] = []
    pbar = tqdm(total=len(dataset), desc=desc)
    torch_device = torch.device(device)

    for da, indices in sorted(da_groups.items()):
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        for batch_imgs, batch_spacings, _ in loader:
            batch_imgs = normalize_volumes(batch_imgs.to(device), normalization)
            autocast = (
                torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if torch_device.type == 'cuda'
                else nullcontext()
            )

            with torch.inference_mode(), autocast:
                cls_tokens, _ = encoder.encode_image(batch_imgs, da=da)

            cls_list.append(cls_tokens.float().cpu())
            spacing_list.append(batch_spacings)
            pbar.update(len(batch_imgs))

    pbar.close()
    return {
        "cls": torch.cat(cls_list),
        "spacing": torch.cat(spacing_list),
    }


def main():
    args = parse_args()
    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata from precrop
    meta_path = crops_dir / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f'no meta.json in {crops_dir}; run scripts/downstream/spatial/prepare_spacing_crops.py'
        )
    meta = orjson.loads(meta_path.read_bytes())

    # Split into train/val
    train_keys, train_spacings, train_das = [], [], []
    val_keys, val_spacings, val_das = [], [], []
    for key, entry in meta.items():
        crop_path = crops_dir / f'{key}.npy'
        if not crop_path.exists():
            raise FileNotFoundError(f'metadata references missing crop: {crop_path}')
        if entry['split'] == 'train':
            train_keys.append(key)
            train_spacings.append(entry['spacing'])
            train_das.append(entry['da'])
        elif entry['split'] == 'val':
            val_keys.append(key)
            val_spacings.append(entry['spacing'])
            val_das.append(entry['da'])
        else:
            raise ValueError(f"unsupported spacing split {entry['split']!r} for {key}")

    print(f'Found {len(train_keys)} train + {len(val_keys)} val pre-cropped samples')

    print(f'Loading encoder from {args.encoder_ckpt}')
    encoder = load_encoder(
        args.encoder_ckpt,
        checkpoint_format=args.checkpoint_format,
        device=args.device,
    )
    print(
        f'Encoder: hidden_size={encoder.config.hidden_size}, '
        f'layers={encoder.config.num_hidden_layers}'
    )

    if args.no_depth_rope:
        encoder.rope.disable_depth = True
        print('  [no-depth-rope] Depth positional encoding disabled')

    print('Extracting train features...')
    train_data = process_split(
        train_keys, train_spacings, train_das, crops_dir, encoder,
        args.device, args.normalization, args.batch_size, args.num_workers, desc='train',
    )
    print(f"Train: {train_data['cls'].shape}")

    print('Extracting val features...')
    val_data = process_split(
        val_keys, val_spacings, val_das, crops_dir, encoder,
        args.device, args.normalization, args.batch_size, args.num_workers, desc='val',
    )
    print(f"Val: {val_data['cls'].shape}")

    torch.save(train_data, output_dir / 'train.pt')
    torch.save(val_data, output_dir / 'val.pt')
    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    main()
