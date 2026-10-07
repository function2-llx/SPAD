"""Fixed-patch ViT runtime adaptation for dense segmentation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import einops
import torch
from torch import Tensor, nn
from torch.utils import checkpoint as torch_checkpoint

from pumit.model.rope import build_rope
from pumit.model.vit import ViT, ViTConfig

from .vit import make_parameter_layers
from .vit_adapter.backbone import InteractiveViTBackbone


def validate_fixed_patch_size(
    patch_size: Sequence[int],
    *,
    in_plane_patch_size: int,
) -> tuple[int, int, int]:
    """Validate one explicitly configured fixed 3D patch size.

    `in_plane_patch_size` is the native pretrained kernel size; the configured in-plane size may be any
    power-of-two divisor of it (reduced by in-plane group summation at checkpoint adaptation).
    """
    patch_size = tuple(int(value) for value in patch_size)
    valid_sizes = tuple(
        1 << exponent
        for exponent in range(in_plane_patch_size.bit_length())
        if (1 << exponent) <= in_plane_patch_size
    )
    if (
        len(patch_size) != 3
        or patch_size[0] not in valid_sizes
        or patch_size[1] != patch_size[2]
        or patch_size[1] not in valid_sizes
    ):
        raise ValueError(
            f'fixed ViT patch size must be (D, P, P) with D in {valid_sizes} and P a power-of-two '
            f'divisor of {in_plane_patch_size}, got {patch_size}'
        )
    return patch_size


def replace_with_fixed_patch_embedding(
    vit: ViT,
    patch_size: Sequence[int],
) -> None:
    """Replace the ViT patch projection with a fixed non-overlapping 3D convolution."""
    patch_size = tuple(int(value) for value in patch_size)
    source = vit.embeddings.patch_embeddings
    vit.embeddings.patch_embeddings = nn.Conv3d(
        source.weight.shape[1],
        source.weight.shape[0],
        kernel_size=patch_size,
        stride=patch_size,
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )


def fixed_patch_size(vit: ViT) -> tuple[int, int, int]:
    """Read the non-overlapping patch stride of an already fixed patch embedding."""
    patch_embedding = vit.embeddings.patch_embeddings
    if isinstance(patch_embedding, nn.Conv3d):
        patch_size = tuple(int(value) for value in patch_embedding.kernel_size)
        if tuple(patch_embedding.stride) != patch_size:
            raise ValueError(
                f'fixed patch embedding stride {tuple(patch_embedding.stride)} does not match '
                f'its kernel size {patch_size}'
            )
        return patch_size
    patch_size = getattr(patch_embedding, 'patch_size', None)
    if patch_size is None:
        raise TypeError(
            f'patch embedding {type(patch_embedding).__name__} does not expose a fixed patch size'
        )
    return tuple(int(value) for value in patch_size)


def prepare_fixed_patch_inputs(
    vit: ViT,
    x: Tensor,
) -> tuple[Tensor, Tensor, tuple[int, int, int]]:
    """Embed one patch grid and build the prefixed token sequence with its RoPE table."""
    patch_size = fixed_patch_size(vit)
    input_shape = tuple(int(size) for size in x.shape[2:])
    if any(size % step for size, step in zip(input_shape, patch_size, strict=True)):
        raise ValueError(
            f'input shape {input_shape} must be divisible by fixed ViT patch size {patch_size}'
        )
    patch_grid = vit.embeddings.patch_embeddings(x)
    patch_shape = tuple(int(size) for size in patch_grid.shape[2:])
    expected_patch_shape = tuple(
        size // step
        for size, step in zip(input_shape, patch_size, strict=True)
    )
    if patch_shape != expected_patch_shape:
        raise RuntimeError(
            f'patch embedding produced grid {patch_shape}, expected {expected_patch_shape}'
        )
    patch_tokens = einops.rearrange(patch_grid, 'b c d h w -> b (d h w) c')
    prefix = torch.cat(
        [
            vit.embeddings.cls_token.expand(x.shape[0], -1, -1),
            vit.embeddings.register_tokens.expand(x.shape[0], -1, -1),
        ],
        dim=1,
    )
    tokens = torch.cat([prefix, patch_tokens], dim=1)
    patch_rope = vit.rope.compute(patch_shape, training=vit.training)
    rope = build_rope(patch_rope, n_prefix=vit.n_prefix)
    if rope.ndim == 3:
        rope = rope.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    return tokens, rope, patch_shape


class FixedPatchViTPyramidBackbone(InteractiveViTBackbone):
    """Expose normalized patch grids from selected depths of a fixed-patch ViT."""

    def __init__(
        self,
        vit: ViT,
        input_channels: int,
        feature_layers: Sequence[int],
    ):
        super().__init__(
            embed_dim=vit.embed_dim,
            feature_layers=feature_layers,
            num_prefix_tokens=vit.n_prefix,
            gradient_checkpointing=vit._grad_ckpt,
        )
        if input_channels not in (1, vit.config.in_channels):
            raise ValueError(
                f'fixed-patch ViT accepts one input channel or its native '
                f'{vit.config.in_channels} channels, got {input_channels}'
            )
        feature_layers = self.feature_layers
        if (
            not feature_layers
            or feature_layers[0] <= 0
            or tuple(sorted(set(feature_layers))) != feature_layers
            or feature_layers[-1] > vit.config.num_hidden_layers
        ):
            raise ValueError(
                f'feature_layers must be unique ascending 1-based depths within '
                f'{vit.config.num_hidden_layers}, got {feature_layers}'
            )
        self.vit = vit
        self.input_channels = input_channels
        self.feature_channels = (vit.embed_dim,) * len(feature_layers)
        feature_stride = fixed_patch_size(vit)
        self.feature_strides = (feature_stride,) * len(feature_layers)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if self.input_channels == 1 and self.vit.config.in_channels != 1:
            x = x.expand(-1, self.vit.config.in_channels, -1, -1, -1)
        tokens, rope, patch_shape = prepare_fixed_patch_inputs(self.vit, x)
        _, hidden_states = self.vit(
            tokens,
            rope,
            hidden_layers=set(self.feature_layers),
        )
        return tuple(
            einops.rearrange(
                self.vit.norm(hidden)[:, self.vit.n_prefix:],
                'b (d h w) c -> b c d h w',
                d=patch_shape[0],
                h=patch_shape[1],
                w=patch_shape[2],
            )
            for hidden in hidden_states
        )

    def prepare_tokens(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int], Tensor]:
        if self.input_channels == 1 and self.vit.config.in_channels != 1:
            x = x.expand(-1, self.vit.config.in_channels, -1, -1, -1)
        tokens, rope, patch_shape = prepare_fixed_patch_inputs(self.vit, x)
        return tokens, patch_shape, rope

    def run_blocks(self, tokens: Tensor, start: int, end: int, context: Tensor) -> Tensor:
        for block_index in range(start, end):
            block = self.vit.layer[block_index]
            should_checkpoint = self.vit.training and self.vit._grad_ckpt and (
                self.vit._grad_ckpt_first_n_layers is None
                or block_index < self.vit._grad_ckpt_first_n_layers
            )
            if should_checkpoint:
                tokens = torch_checkpoint.checkpoint(
                    block,
                    tokens,
                    context,
                    None,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, context)
        return tokens

    def normalize_patch_tokens(self, tokens: Tensor) -> Tensor:
        return self.vit.norm(tokens)[:, self.num_prefix_tokens:]

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the fixed-patch ViT parameters from its embedding to final block."""
        return make_parameter_layers(
            self.vit.embeddings.parameters(),
            self.vit.layer,
            self.vit.norm.parameters(),
        )


class FixedPatchViTBackbone(nn.Module):
    """Expose the normalized final patch grid of a fixed-patch ViT as one feature map (SimpleFPN contract)."""

    def __init__(self, vit: ViT, input_channels: int):
        super().__init__()
        if input_channels not in (1, vit.config.in_channels):
            raise ValueError(
                f'fixed-patch ViT accepts one input channel or its native '
                f'{vit.config.in_channels} channels, got {input_channels}'
            )
        self.vit = vit
        self.input_channels = input_channels
        self.feature_channels = vit.embed_dim
        self.feature_stride = fixed_patch_size(vit)

    def forward(self, x: Tensor) -> Tensor:
        if self.input_channels == 1 and self.vit.config.in_channels != 1:
            x = x.expand(-1, self.vit.config.in_channels, -1, -1, -1)
        tokens, rope, patch_shape = prepare_fixed_patch_inputs(self.vit, x)
        patch_tokens = self.vit(tokens, rope)[:, self.vit.n_prefix:]
        return einops.rearrange(
            patch_tokens,
            'b (d h w) c -> b c d h w',
            d=patch_shape[0],
            h=patch_shape[1],
            w=patch_shape[2],
        )

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the fixed-patch ViT parameters from its embedding to final block."""
        return make_parameter_layers(
            self.vit.embeddings.parameters(),
            self.vit.layer,
            self.vit.norm.parameters(),
        )


def vit_config_from_mapping(config: Mapping[str, object]) -> ViTConfig:
    """Reconstruct a ViT config from a strict serializable mapping."""
    fields = {field.name for field in dataclasses.fields(ViTConfig)}
    missing = fields - config.keys()
    unexpected = config.keys() - fields
    if missing or unexpected:
        raise ValueError(
            f'ViT architecture config mismatch: missing={sorted(missing)}, '
            f'unexpected={sorted(unexpected)}'
        )
    return ViTConfig(**{key: config[key] for key in fields})
