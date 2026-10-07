"""Lazy registry for dense segmentation encoder adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from torch import nn

from .plan import EncoderPlan

PrepareConfig = Callable[[Path | None, str | None, bool], dict[str, object]]
BuildEncoder = Callable[[EncoderPlan, int, Mapping[str, object]], nn.Module]
LoadPretrained = Callable[[nn.Module, Path | None], None]


@dataclass(frozen=True)
class SegBackbone:
    """Construction and training-only initialization for one dense encoder."""

    config_keys: frozenset[str]
    optional_config_keys: frozenset[str]
    prepare_config: PrepareConfig
    build_encoder: BuildEncoder
    load_pretrained: LoadPretrained

    @property
    def requires_vit_patch_size(self) -> bool:
        return 'vit_patch_size' in self.config_keys

    @property
    def is_vit_adapter(self) -> bool:
        return {'spatial_prior_input', 'with_high_resolution_stem'} <= self.config_keys

    @property
    def can_drop_stem(self) -> bool:
        """Whether the encoder can drop its raw-image P0-P1 stem and expose P2-P5 only."""
        return 'with_high_resolution_stem' in self.config_keys | self.optional_config_keys

    def validate_config(self, config: Mapping[str, object]) -> None:
        missing = self.config_keys - config.keys()
        unexpected = config.keys() - self.config_keys - self.optional_config_keys
        if missing or unexpected:
            raise ValueError(
                f'backbone config keys mismatch: '
                f'missing={sorted(missing)}, '
                f'unexpected={sorted(unexpected)}'
            )


_SPECS: dict[str, str] = {
    'pumit-simple-fpn': 'pumit_simple_fpn',
    'pumit-pretrained-neck': 'pumit_pretrained_neck',
    'pumit-vit-adapter': 'pumit_vit_adapter',
    'random-vit-adapter': 'random_vit_adapter',
    'random-simple-fpn': 'random_simple_fpn',
    'dinov3': 'dinov3',
    'dinov3-vit-adapter': 'dinov3_vit_adapter',
    '3dino': 'three_dino',
    '3dino-vit-adapter': 'three_dino_vit_adapter',
    'biomedclip': 'biomedclip',
    'biomedclip-vit-adapter': 'biomedclip_vit_adapter',
    'eva02-l': 'eva02',
    'eva02-l-vit-adapter': 'eva02_vit_adapter',
    'unimiss': 'unimiss',
    'unimiss-plus': 'unimiss_plus',
    'sam-med3d': 'sam_med3d',
    'sam-med3d-vit-adapter': 'sam_med3d_vit_adapter',
    'suprem-unet': 'suprem_unet',
    'stu-net-l': 'stu_net',
    'voco-b': 'voco',
    'voco-l': 'voco_l',
    'segvol-vit-adapter': 'segvol_vit_adapter',
    'sat-pro': 'sat_pro',
}


class _LazyBackbones(Mapping[str, SegBackbone]):
    def __getitem__(self, name: str) -> SegBackbone:
        if name not in _SPECS:
            raise KeyError(name)
        module = import_module(f'pumit.downstream.seg.backbones.{_SPECS[name]}')
        return SegBackbone(
            config_keys=module.CONFIG_KEYS,
            optional_config_keys=getattr(module, 'OPTIONAL_CONFIG_KEYS', frozenset()),
            prepare_config=module.prepare_config,
            build_encoder=module.build_encoder,
            load_pretrained=module.load_pretrained,
        )

    def __iter__(self):
        return iter(_SPECS)

    def __len__(self) -> int:
        return len(_SPECS)


BACKBONES = _LazyBackbones()


def build_encoder(
    name: str,
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> nn.Module:
    """Build one randomly initialized encoder without touching pretrained artifacts."""
    backbone = BACKBONES[name]
    backbone.validate_config(config)
    return backbone.build_encoder(plan, input_channels, config)


def load_pretrained(
    name: str,
    encoder: nn.Module,
    weights: str | Path | None,
) -> None:
    """Apply training-only pretrained initialization to a constructed encoder."""
    path = Path(weights) if weights is not None else None
    BACKBONES[name].load_pretrained(encoder, path)
