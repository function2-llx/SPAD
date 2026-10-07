"""Backbone registry: name -> Backbone(model_builder, transform_batch) pair.

`model_builder` is a per-variant named factory (eva02_base, eva02_large, dinov3, ucpt, ...)
returning an nn.Module with forward(x)->(global_features,patch_tokens). `transform_batch` is a free,
model-independent function (runs in DataLoader workers, must not depend on the built model);
each experiment owns its own. BACKBONES.keys() is the single source of valid backbone names.

Backbone modules are imported LAZILY (on first `BACKBONES[name]` access): the cls-m3d env is py3.11
and cannot import the SPAD ViT chain (PEP-695 `type` syntax), while the cls env is py3.13
and cannot import M3D's monai-1.3; lazy dispatch lets each env load only the backbone it runs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from torch import Tensor, nn

ModelBuilder = Callable[..., nn.Module]
TransformBatch = Callable[..., dict[str, Tensor]]


@dataclass(frozen=True)
class Backbone:
    """One experiment entry: how to build the model + how to transform a batch."""
    create_model: ModelBuilder
    transform_batch: TransformBatch


# name -> (backbones submodule, builder_attr, transform_attr). Modules resolved lazily in _LazyBackbones.
# DINOv3 and ucpt-imagenet use ImageNet normalization; EVA/BiomedCLIP use CLIP.
# ucpt and `_`-prefixed keys preserve legacy PUMIT [-1,1] normalization.
_SPECS: dict[str, tuple[str, str, str]] = {
    'eva02-b':    ('eva02', 'eva02_base', 'transform_batch'),
    'eva02-l':    ('eva02', 'eva02_large', 'transform_batch'),
    '_eva02-b':   ('eva02', 'eva02_base', 'transform_batch_pm1'),
    '_eva02-l':   ('eva02', 'eva02_large', 'transform_batch_pm1'),
    'dinov3':     ('vit_spad', 'dinov3', 'transform_batch_imagenet'),
    '_dinov3':    ('vit_spad', 'dinov3', 'transform_batch'),
    'ucpt':       ('vit_spad', 'ucpt', 'transform_batch'),
    'ucpt-imagenet': ('vit_spad', 'ucpt', 'transform_batch_imagenet'),
    'biomedclip': ('biomedclip', 'biomedclip', 'transform_batch'),
    'sam-med3d':  ('sam_med3d', 'sam_med3d', 'transform_batch'),
    '3dino':      ('three_dino', 'three_dino', 'transform_batch'),
    'unimiss-plus': ('unimiss_plus', 'unimiss_plus', 'transform_batch'),
    'unimiss':     ('unimiss', 'unimiss', 'transform_batch'),
    'voco-l':     ('voco', 'voco_l', 'transform_batch'),
    'genesis':    ('unet3d', 'models_genesis', 'transform_batch'),
    'suprem':     ('unet3d', 'suprem', 'transform_batch'),
    'm3d':        ('m3d_clip', 'm3d', 'transform_batch'),
}


class _LazyBackbones(Mapping):
    """Read-only mapping name -> Backbone; imports the backbone module only on access."""

    def __getitem__(self, name: str) -> Backbone:
        if name not in _SPECS:
            raise KeyError(name)
        modname, builder, transform = _SPECS[name]
        mod = import_module(f'pumit.downstream.cls.backbones.{modname}')
        return Backbone(getattr(mod, builder), getattr(mod, transform))

    def __iter__(self):
        return iter(_SPECS)

    def __len__(self) -> int:
        return len(_SPECS)


BACKBONES = _LazyBackbones()
