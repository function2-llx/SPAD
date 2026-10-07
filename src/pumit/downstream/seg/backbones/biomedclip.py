"""Dynamic-grid BiomedCLIP flat ViT adapter for dense segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import einops
import timm
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pumit.downstream.seg.adapters.pyramid import PlanAlignedPyramidEncoder3D, SimpleFPNEncoder3D, pyramid_options
from pumit.downstream.seg.adapters.timm_vit import (
    DynamicPatchEmbed3D,
    assert_reviewed_seg_timm,
    inflate_patch_projection,
    lift_2d_position_embedding,
)
from pumit.downstream.seg.adapters.vit import (
    make_parameter_layers,
    pinned_feature_layers,
    select_evenly_spaced_layers,
)
from pumit.downstream.seg.plan import EncoderPlan

_MODEL_NAME = 'vit_base_patch16_224'
_PATCH_SIZE = 16
_PROJECTION_DIM = 512
_DEPTH = 12
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
CONFIG_KEYS = frozenset({'gradient_checkpointing'})
# Plans without ``feature_layers`` predate the multi-depth readout and build the final projected-map SimpleFPN.
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch', 'feature_layers'})


def _build_visual_trunk(
    input_channels: int,
    patch_size: Sequence[int],
    *,
    gradient_checkpointing: bool,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build the reviewed BiomedCLIP visual trunk with a fixed 3D patch embedding."""
    patch_size = tuple(int(value) for value in patch_size)
    assert_reviewed_seg_timm()
    trunk = timm.create_model(
        _MODEL_NAME,
        pretrained=False,
        num_classes=0,
        global_pool='token',
        drop_path_rate=drop_path_rate,
    )
    if (
        trunk.dynamic_img_size
        or trunk.num_prefix_tokens != 1
        or trunk.reg_token is not None
        or trunk.no_embed_class
        or trunk.global_pool != 'token'
    ):
        raise RuntimeError('unexpected BiomedCLIP ViT topology for the reviewed 3D adapter')
    trunk.patch_embed = DynamicPatchEmbed3D(
        inflate_patch_projection(
            trunk.patch_embed.proj,
            input_channels=input_channels,
            patch_size=patch_size,
        )
    )
    trunk.grad_checkpointing = gradient_checkpointing
    if len(trunk.blocks) != _DEPTH:
        raise RuntimeError(f'expected {_DEPTH} BiomedCLIP blocks, got {len(trunk.blocks)}')
    return trunk


class BiomedCLIPFeatureBackbone(nn.Module):
    """Expose BiomedCLIP's visual trunk on a dynamic 3D input.

    Without ``feature_layers`` the output is the final patch grid after the CLIP projection (512-d); with them it is
    one normalised 768-d trunk grid per depth, the projection being a CLIP-head artifact with no per-depth analogue.
    """

    feature_stride = (_PATCH_SIZE,) * 3

    def __init__(
        self,
        input_channels: int,
        *,
        gradient_checkpointing: bool,
        patch_size: Sequence[int] = (_PATCH_SIZE,) * 3,
        feature_layers: tuple[int, ...] | None = None,
    ):
        super().__init__()
        patch_size = tuple(int(value) for value in patch_size)
        self.trunk = _build_visual_trunk(
            input_channels,
            patch_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.feature_stride = patch_size
        self.feature_layers = feature_layers
        if feature_layers is None:
            self.feature_channels = _PROJECTION_DIM
            self.projection = nn.Linear(self.trunk.embed_dim, _PROJECTION_DIM, bias=False)
        else:
            self.feature_channels = (self.trunk.embed_dim,) * len(feature_layers)
            self.feature_strides = (patch_size,) * len(feature_layers)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        grid = tuple(
            size // step
            for size, step in zip(x.shape[2:], self.feature_stride, strict=True)
        )
        tokens = self.trunk.patch_embed(x)
        tokens = torch.cat([self.trunk.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)
        tokens = tokens + lift_2d_position_embedding(
            self.trunk.pos_embed,
            grid,
            num_prefix_tokens=self.trunk.num_prefix_tokens,
        )
        tokens = self.trunk.patch_drop(self.trunk.pos_drop(tokens))
        tokens = self.trunk.norm_pre(tokens)

        def to_grid(patch_tokens: Tensor) -> Tensor:
            return einops.rearrange(patch_tokens, 'b (d h w) c -> b c d h w', d=grid[0], h=grid[1], w=grid[2])

        wanted = set(self.feature_layers or ())
        hidden = []
        for depth, block in enumerate(self.trunk.blocks, start=1):
            if self.trunk.grad_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if depth in wanted:
                hidden.append(self.trunk.norm(tokens)[:, self.trunk.num_prefix_tokens:])
        if self.feature_layers is None:
            return to_grid(self.projection(self.trunk.norm(tokens)[:, self.trunk.num_prefix_tokens:]))
        return tuple(to_grid(patch_tokens) for patch_tokens in hidden)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return make_parameter_layers(
            (
                *self.trunk.patch_embed.parameters(),
                self.trunk.cls_token,
                self.trunk.pos_embed,
            ),
            self.trunk.blocks,
            (
                *self.trunk.norm.parameters(),
                *(self.projection.parameters() if self.feature_layers is None else ()),
            ),
        )


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate the released OpenCLIP state dict and serialize adapter options."""
    if weights is None:
        raise ValueError('BiomedCLIP requires open_clip_pytorch_model.bin')
    if checkpoint_format is not None:
        raise ValueError('BiomedCLIP does not accept checkpoint_format')
    return {'gradient_checkpointing': gradient_checkpointing, 'feature_layers': list(_FEATURE_LAYERS)}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D:
    """Construct a random BiomedCLIP visual tower plus the shared pyramid readout."""
    feature_layers = pinned_feature_layers(config, _FEATURE_LAYERS, consumer='BiomedCLIP')
    backbone = BiomedCLIPFeatureBackbone(
        input_channels,
        gradient_checkpointing=bool(config['gradient_checkpointing']),
        feature_layers=feature_layers,
    )
    encoder_class = SimpleFPNEncoder3D if feature_layers is None else PlanAlignedPyramidEncoder3D
    return encoder_class(backbone, plan, input_channels=input_channels, **pyramid_options(config))


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the released BiomedCLIP visual trunk, plus the projection when the readout uses it."""
    if weights is None:
        raise ValueError('BiomedCLIP requires open_clip_pytorch_model.bin')
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(f'expected a pyramid encoder, got {type(encoder).__name__}')
    projection_weight = _load_pretrained_trunk(
        encoder.backbone.trunk,
        weights,
        input_channels=encoder.input_channels,
    )
    if encoder.backbone.feature_layers is None:
        encoder.backbone.projection.load_state_dict(
            {'weight': projection_weight},
            strict=True,
        )


def _load_pretrained_trunk(
    trunk: nn.Module,
    weights: Path,
    *,
    input_channels: int,
) -> Tensor:
    """Load the released visual trunk and return its separate CLIP projection."""
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    trunk_state = {
        key.removeprefix('visual.trunk.'): value
        for key, value in state.items()
        if key.startswith('visual.trunk.')
    }
    patch_key = 'patch_embed.proj.weight'
    source_weight = trunk_state[patch_key]
    source_patch = nn.Conv2d(
        source_weight.shape[1],
        source_weight.shape[0],
        kernel_size=_PATCH_SIZE,
        stride=_PATCH_SIZE,
        bias='patch_embed.proj.bias' in trunk_state,
    )
    source_patch.weight = nn.Parameter(trunk_state[patch_key])
    if source_patch.bias is not None:
        source_patch.bias = nn.Parameter(trunk_state['patch_embed.proj.bias'])
    trunk_state[patch_key] = inflate_patch_projection(
        source_patch,
        input_channels=input_channels,
        patch_size=trunk.patch_embed.patch_size,
    ).weight.detach()
    trunk.load_state_dict(trunk_state, strict=True)
    return state['visual.head.proj.weight']
