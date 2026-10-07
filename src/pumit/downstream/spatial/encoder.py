"""Encoder loading and input normalization for controlled spatial evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from pumit.model.vit import ViT, ViTConfig

CHECKPOINT_FORMATS = ('ucpt', 'legacy-ssl')
NORMALIZATIONS = ('pumit', 'imagenet')

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_vit_config(training_config: Mapping[str, object]) -> ViTConfig:
    """Build the bare evaluation encoder configuration stored in a training checkpoint."""
    required = ('embed_dim', 'depth', 'num_heads')
    missing = [key for key in required if key not in training_config]
    if missing:
        raise KeyError(f'checkpoint config is missing ViT fields: {missing}')
    embed_dim = int(training_config['embed_dim'])
    return ViTConfig(
        hidden_size=embed_dim,
        num_hidden_layers=int(training_config['depth']),
        num_attention_heads=int(training_config['num_heads']),
        intermediate_size=int(embed_dim * float(training_config.get('mlp_ratio', 4.0))),
        num_register_tokens=int(training_config.get('n_register_tokens', 4)),
    )


def slice_encoder_state(
    model_state: Mapping[str, Tensor],
    checkpoint_format: str,
) -> dict[str, Tensor]:
    """Extract a bare ViT state dict from an explicitly identified checkpoint format."""
    if checkpoint_format == 'ucpt':
        prefix = 'teacher_vit.'
    elif checkpoint_format == 'legacy-ssl':
        prefix = 'encoder.'
    else:
        raise ValueError(
            f'unsupported checkpoint format {checkpoint_format!r}; expected one of {CHECKPOINT_FORMATS}'
        )

    encoder_state: dict[str, Tensor] = {}
    for key, value in model_state.items():
        key = key.removeprefix('_orig_mod.')
        if key.startswith(prefix):
            encoder_state[key.removeprefix(prefix)] = value
    if not encoder_state:
        raise KeyError(f'checkpoint model state contains no {prefix}* keys')
    return encoder_state


def load_encoder(
    checkpoint_path: str | Path,
    *,
    checkpoint_format: str,
    device: str | torch.device,
) -> ViT:
    """Load a frozen bare ViT from a UCPT or legacy SSL checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint.get('config'), Mapping):
        raise TypeError('checkpoint config must be a mapping')
    if not isinstance(checkpoint.get('model'), Mapping):
        raise TypeError('checkpoint model must be a mapping')

    encoder = ViT(build_vit_config(checkpoint['config']))
    encoder.load_state_dict(
        slice_encoder_state(checkpoint['model'], checkpoint_format),
        strict=True,
    )
    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def normalize_volumes(volumes: Tensor, normalization: str) -> Tensor:
    """Normalize `[B,C,D,H,W]` volumes according to the evaluated model's input contract."""
    if volumes.ndim != 5:
        raise ValueError(f'expected volumes with shape [B,C,D,H,W], got {tuple(volumes.shape)}')
    if volumes.shape[1] == 1:
        volumes = volumes.expand(-1, 3, -1, -1, -1)
    elif volumes.shape[1] != 3:
        raise ValueError(f'expected one or three input channels, got {volumes.shape[1]}')

    if normalization == 'pumit':
        return volumes * 2 - 1
    if normalization == 'imagenet':
        mean = volumes.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        std = volumes.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1)
        return (volumes - mean) / std
    raise ValueError(f'unsupported normalization {normalization!r}; expected one of {NORMALIZATIONS}')
