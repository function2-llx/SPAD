"""Extract and cache each backbone's frozen native global features for MedMNIST.

Feeds the frozen-probe table: the encoder never trains, so its features are extracted once and
every probe head reads the cache. The cached vector is the backbone's own official global readout
(the first element of the adapter's forward tuple). Each file records the geometry that produced
it, since holding the token grid equal across backbones is what makes that table a comparison.

Usage:
    pixi run -e cls python -m pumit.downstream.cls.extract_features \
        --backbone dinov3 --weights pretrained/dinov3-vitl16/model.safetensors \
        --vit-config configs/downstream/cls/vit_l.yaml --img-size 192 \
        --datasets configs/downstream/cls/datasets.yaml \
        --flags nodulemnist3d \
        --out precompute/downstream/cls/dinov3/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .config import DatasetSpec, load_dataset_specs, load_vit_config
from .data import DEFAULT_DATA_ROOT, build_arrays
from .registry import BACKBONES

SPLITS = ('train', 'val', 'test')


@torch.inference_mode()
def extract_flag(encoder, transform_batch, spec: DatasetSpec, out_dir: Path,
                 batch_size: int, device: str, data_root: str,
                 provenance: dict | None = None) -> None:
    """Cache one dataset's frozen features, recording the geometry that produced them.

    `provenance` carries the requested config (backbone, img_size, weights); the observed token
    count is measured here and stored alongside it. Features are cached and reused across probe
    runs, so without that record a cache extracted at one geometry is indistinguishable from one
    extracted at another -- and the comparison this feeds depends on every backbone matching.
    """
    for split in SPLITS:
        images, labels = build_arrays(spec.flag, spec.size, split, root=data_root)
        global_list = []
        tokens: int | None = None
        n = images.shape[0]
        for i in tqdm(range(0, n, batch_size), desc=f'{spec.flag}/{split}'):
            xb = transform_batch(images[i:i + batch_size], is_3d=spec.is_3d)
            xb = {k: v.to(device) for k, v in xb.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                global_features, patch_tokens = encoder(**xb)
            if tokens is None:
                tokens = int(patch_tokens.shape[1])
            global_list.append(global_features.float().cpu())
        data = {
            'native': torch.cat(global_list),
            'label': torch.from_numpy(labels),
            'meta': {**(provenance or {}), 'flag': spec.flag, 'split': split,
                     'tokens': tokens, 'embed_dim': int(encoder.embed_dim)},
        }
        split_dir = out_dir / spec.flag
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, split_dir / f'{split}.pt')
        print(f'{spec.flag}/{split}: native {tuple(data["native"].shape)} '
              f'label {tuple(data["label"].shape)} tokens {tokens}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=sorted(BACKBONES))
    parser.add_argument('--weights', default=None)
    parser.add_argument('--vit-config', default=None)
    parser.add_argument('--img-size', type=int, default=None,
                        help='resize edge for backbones that expose one (dinov3/ucpt/eva02/'
                             'biomedclip, where it sets the token grid); the rest pin their own '
                             'input and reject it')
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--flags', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--seed', type=int, default=0,
                        help='init seed, only matters for a weightless (random-init) backbone')
    args = parser.parse_args()

    backbone = BACKBONES[args.backbone]
    specs = load_dataset_specs(args.datasets)
    # forward each optional kwarg only when given, so a backbone that does not declare it raises
    # instead of discarding it (see build_trainable_encoder in finetune.py)
    extra = {}
    if args.vit_config is not None:
        extra['vit_config'] = load_vit_config(args.vit_config)
    if args.img_size is not None:
        extra['img_size'] = args.img_size
    if args.weights is not None:
        extra['weights'] = args.weights
    output_dir = Path(args.out)
    flags = sorted(args.flags, key=lambda flag: specs[flag].is_3d)
    encoder, current_dims = None, None
    for flag in flags:
        spec = specs[flag]
        dims = 3 if spec.is_3d else 2
        if dims != current_dims:
            torch.manual_seed(args.seed)
            encoder = backbone.create_model(dims=dims, device=args.device, trainable=False,
                                            **extra)
            current_dims = dims
        extract_flag(
            encoder,
            backbone.transform_batch,
            spec,
            output_dir,
            args.batch_size,
            args.device,
            args.data_root,
            provenance={'backbone': args.backbone, 'img_size': args.img_size,
                        'weights': args.weights, 'vit_config': args.vit_config,
                        'seed': args.seed},
        )


if __name__ == '__main__':
    main()
