"""Dynamic-grid EVA-02-L flat ViT adapter for dense segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import einops
import timm
import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pumit.downstream.cls.backbones.eva02 import (
    EVA_DEPTH_ROPE_BASE,
    EVA_ROPE_BASE,
    _extend_eva_rope_3d,
)
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

_MODEL_NAME = 'eva02_large_patch14_224'
_NATIVE_PATCH_SIZE = 14
_PATCH_SIZE = 16
_FEATURE_CHANNELS = 1024
_DEPTH = 24
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
CONFIG_KEYS = frozenset({'gradient_checkpointing'})
# Plans without ``feature_layers`` predate the multi-depth readout and build the final-map SimpleFPN.
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch', 'feature_layers'})


def _validate_eva(model: nn.Module) -> None:
    """Validate the pinned timm EVA topology used by the dynamic 3D adapter."""
    if (
        model.dynamic_img_size
        or getattr(model, 'rope_mixed', False)
        or model.num_prefix_tokens != 1
        or model.reg_token is not None
        or model.no_embed_class
        or model.global_pool != 'avg'
        or model.patch_drop is not None
        or not isinstance(model.norm_pre, nn.Identity)
        or model.pos_drop.p != 0
    ):
        raise RuntimeError('unexpected EVA-02-L topology for the reviewed dense adapter')
    patch_embed = model.patch_embed
    if (
        not isinstance(patch_embed.norm, nn.Identity)
        or patch_embed.proj.kernel_size != (_NATIVE_PATCH_SIZE,) * 2
        or patch_embed.proj.stride != (_NATIVE_PATCH_SIZE,) * 2
        or model.embed_dim != _FEATURE_CHANNELS
    ):
        raise RuntimeError('unexpected EVA-02-L patch embedding for the reviewed dense adapter')
    rope = model.rope
    head_dim = model.embed_dim // model.blocks[0].attn.num_heads
    if (
        rope is None
        or rope.dim != head_dim
        or rope.temperature != EVA_ROPE_BASE
        or rope.in_pixels
        or tuple(rope.feat_shape) != (16, 16)
        or tuple(rope.ref_feat_shape) != (16, 16)
    ):
        raise RuntimeError('unexpected EVA-02-L RoPE configuration for the reviewed dense adapter')


def _build_eva_model(
    input_channels: int,
    patch_size: Sequence[int],
    *,
    gradient_checkpointing: bool,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build the reviewed EVA-02-L model with a fixed 3D patch embedding."""
    patch_size = tuple(int(value) for value in patch_size)
    assert_reviewed_seg_timm()
    model = timm.create_model(
        _MODEL_NAME, pretrained=False, num_classes=0, drop_path_rate=drop_path_rate,
    )
    _validate_eva(model)
    model.patch_embed = DynamicPatchEmbed3D(
        inflate_patch_projection(
            model.patch_embed.proj,
            input_channels=input_channels,
            patch_size=patch_size,
        )
    )
    model.grad_checkpointing = gradient_checkpointing
    model.fc_norm.requires_grad_(False)
    if len(model.blocks) != _DEPTH:
        raise RuntimeError(f'expected {_DEPTH} EVA-02-L blocks, got {len(model.blocks)}')
    return model


class Eva02LargeFeatureBackbone(nn.Module):
    """Expose EVA-02-L's patch grids on the planned 3D input: the final one, or one normalised grid per depth."""

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
        self.model = _build_eva_model(
            input_channels,
            patch_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.feature_stride = patch_size
        self.feature_layers = feature_layers
        if feature_layers is None:
            self.feature_channels = _FEATURE_CHANNELS
        else:
            self.feature_channels = (_FEATURE_CHANNELS,) * len(feature_layers)
            self.feature_strides = (patch_size,) * len(feature_layers)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        grid = tuple(
            size // step
            for size, step in zip(x.shape[2:], self.feature_stride, strict=True)
        )
        tokens = self.model.patch_embed(x)
        tokens = torch.cat([self.model.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)
        tokens = tokens + lift_2d_position_embedding(
            self.model.pos_embed,
            grid,
            num_prefix_tokens=self.model.num_prefix_tokens,
        )
        tokens = self.model.norm_pre(self.model.pos_drop(tokens))

        rope_2d = self.model.rope._get_pos_embed_values(
            grid[1:],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        rope_3d = _extend_eva_rope_3d(rope_2d, grid[0], EVA_DEPTH_ROPE_BASE)

        def to_grid(hidden: Tensor) -> Tensor:
            return einops.rearrange(
                self.model.norm(hidden)[:, self.model.num_prefix_tokens:],
                'b (d h w) c -> b c d h w',
                d=grid[0],
                h=grid[1],
                w=grid[2],
            )

        wanted = set(self.feature_layers or ())
        features = []
        for depth, block in enumerate(self.model.blocks, start=1):
            if self.model.grad_checkpointing and self.training:
                tokens = checkpoint(block, tokens, rope=rope_3d, use_reentrant=False)
            else:
                tokens = block(tokens, rope=rope_3d)
            if depth in wanted:
                features.append(to_grid(tokens))
        if self.feature_layers is None:
            return to_grid(tokens)
        return tuple(features)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return make_parameter_layers(
            (
                *self.model.patch_embed.parameters(),
                self.model.cls_token,
                self.model.pos_embed,
            ),
            self.model.blocks,
            self.model.norm.parameters(),
        )


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate the released timm checkpoint and serialize adapter options."""
    if weights is None or weights.suffix != '.safetensors':
        raise ValueError('EVA-02-L requires the released model.safetensors checkpoint')
    if checkpoint_format is not None:
        raise ValueError('EVA-02-L does not accept checkpoint_format')
    return {'gradient_checkpointing': gradient_checkpointing, 'feature_layers': list(_FEATURE_LAYERS)}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D:
    """Construct a random EVA-02-L trunk plus the shared pyramid readout."""
    feature_layers = pinned_feature_layers(config, _FEATURE_LAYERS, consumer='EVA-02-L')
    backbone = Eva02LargeFeatureBackbone(
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
    """Load and adapt the released EVA-02-L MIM-in22k state dict."""
    if weights is None:
        raise ValueError('EVA-02-L requires the released model.safetensors checkpoint')
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(f'expected a pyramid encoder, got {type(encoder).__name__}')

    state = _adapted_pretrained_state(
        weights,
        input_channels=encoder.input_channels,
        patch_size=encoder.backbone.model.patch_embed.patch_size,
    )
    encoder.backbone.model.load_state_dict(state, strict=True)


def _adapted_pretrained_state(
    weights: Path,
    *,
    input_channels: int,
    patch_size: Sequence[int],
) -> dict[str, Tensor]:
    """Load EVA and adapt its 2D patch projection to one fixed 3D patch."""
    state = dict(load_file(weights, device='cpu'))
    patch_key = 'patch_embed.proj.weight'
    source_weight = state[patch_key]
    source_patch = nn.Conv2d(
        source_weight.shape[1],
        source_weight.shape[0],
        kernel_size=_NATIVE_PATCH_SIZE,
        stride=_NATIVE_PATCH_SIZE,
        bias='patch_embed.proj.bias' in state,
    )
    source_patch.weight = nn.Parameter(source_weight)
    if source_patch.bias is not None:
        source_patch.bias = nn.Parameter(state['patch_embed.proj.bias'])
    state[patch_key] = inflate_patch_projection(
        source_patch,
        input_channels=input_channels,
        patch_size=patch_size,
    ).weight.detach()
    return state
