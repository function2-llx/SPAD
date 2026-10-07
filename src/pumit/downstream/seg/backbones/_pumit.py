"""Shared UCPT checkpoint handling for PUMIT segmentation backbones."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from pumit.downstream.seg.adapters.checkpoint import adapt_patch_embed_weight
from pumit.downstream.spatial.encoder import build_vit_config, slice_encoder_state
from pumit.model.vit import ViT


CHECKPOINT_FORMAT = 'ucpt'
_PATCH_WEIGHT_KEY = 'embeddings.patch_embeddings.weight'
_NECK_PREFIX = 'ema_seg.neck.'


def load_checkpoint(weights: Path | None, *, consumer: str) -> Mapping[str, object]:
    if weights is None:
        raise ValueError(f'{consumer} requires a UCPT checkpoint path')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError('PUMIT checkpoint must be a mapping')
    return checkpoint


def _model_config(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    training_config = checkpoint.get('config')
    if not isinstance(training_config, Mapping):
        raise TypeError('PUMIT checkpoint config must be a mapping')
    model_config = training_config.get('model')
    if not isinstance(model_config, Mapping):
        raise TypeError('PUMIT checkpoint config must contain a model mapping')
    return model_config


def vit_architecture(
    checkpoint: Mapping[str, object],
    *,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    return dataclasses.asdict(
        dataclasses.replace(
            build_vit_config(_model_config(checkpoint)),
            pos_embed_rescale=None,
            grad_ckpt=gradient_checkpointing,
        )
    )


def neck_hidden_size(checkpoint: Mapping[str, object]) -> int:
    """Read the pretrained segmentation pyramid width from the UCPT training config."""
    model_config = _model_config(checkpoint)
    if 'seg_hidden_size' not in model_config:
        raise KeyError('PUMIT checkpoint config is missing seg_hidden_size')
    return int(model_config['seg_hidden_size'])


def load_vit(
    vit: ViT,
    checkpoint: Mapping[str, object],
    *,
    consumer: str,
) -> None:
    model_state = checkpoint.get('model')
    if not isinstance(model_state, Mapping):
        raise TypeError('PUMIT checkpoint model must be a mapping')
    patch_embedding = vit.embeddings.patch_embeddings
    if not isinstance(patch_embedding, nn.Conv3d):
        raise TypeError(
            f'downstream {consumer} requires a fixed Conv3d patch embedding, got '
            f'{type(patch_embedding).__name__}'
        )
    state_dict = slice_encoder_state(model_state, CHECKPOINT_FORMAT)
    if _PATCH_WEIGHT_KEY in state_dict:
        state_dict[_PATCH_WEIGHT_KEY] = adapt_patch_embed_weight(
            state_dict[_PATCH_WEIGHT_KEY],
            patch_embedding.kernel_size,
        )
    vit.load_state_dict(state_dict, strict=True)


def load_neck(
    neck: nn.Module,
    checkpoint: Mapping[str, object],
    *,
    consumer: str,
) -> None:
    """Load the EMA segmentation neck, excluding every other ``ema_seg`` component."""
    model_state = checkpoint.get('model')
    if not isinstance(model_state, Mapping):
        raise TypeError('PUMIT checkpoint model must be a mapping')
    state_dict = {}
    for key, value in model_state.items():
        key = key.removeprefix('_orig_mod.')
        if key.startswith(_NECK_PREFIX):
            state_dict[key.removeprefix(_NECK_PREFIX)] = value
    if not state_dict:
        raise KeyError(
            f'downstream {consumer} requires {_NECK_PREFIX}* keys in the checkpoint model state'
        )
    neck.load_state_dict(state_dict, strict=True)
