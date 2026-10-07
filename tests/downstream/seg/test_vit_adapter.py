from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pumit.downstream.seg.adapters.fixed_patch_vit import (
    FixedPatchViTPyramidBackbone,
    replace_with_fixed_patch_embedding,
)
from pumit.downstream.seg.adapters.deformable_attention import (
    MultiScaleDeformableAttention3D,
    flatten_multiscale_features_3d,
    reference_points_3d,
    unflatten_multiscale_features_3d,
)
from pumit.downstream.seg.adapters.timm_vit import lift_2d_position_embedding
from pumit.downstream.seg.adapters.vit_adapter import (
    PlanAlignedViTProjection3D,
    TimmAbsPosInteractiveBackbone,
    ViTAdapterEncoder3D,
    vit_adapter_config,
)
from pumit.model.vit import ViT, ViTConfig
from tests.downstream.seg.conftest import ISOTROPIC_STRIDES, make_plan


def _plan(*, anisotropic: bool = False, num_stages: int = 6):
    strides = ([[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [2, 2, 2]] if anisotropic else ISOTROPIC_STRIDES)
    return make_plan([4, 8, 12, 16, 24, 32][:num_stages], strides[:num_stages])


def _tiny_encoder(
    *,
    anisotropic: bool = False,
    num_stages: int = 6,
    patch_depth: int = 16,
    spatial_prior_input: str = 'raw',
    with_high_resolution_stem: bool = True,
    spatial_prior_channels: tuple[int, ...] | None = None,
    output_fusion: str = 'add-then-project',
) -> ViTAdapterEncoder3D:
    vit = ViT(
        ViTConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=2,
            patch_size=16,
            in_channels=3,
            pos_embed_rescale=None,
        )
    )
    replace_with_fixed_patch_embedding(vit, (patch_depth, 16, 16))
    backbone = FixedPatchViTPyramidBackbone(vit, 1, (1, 2, 3, 4))
    return ViTAdapterEncoder3D(
        backbone,
        _plan(anisotropic=anisotropic, num_stages=num_stages),
        input_channels=1,
        adapter_dim=8,
        deform_attention_dim=4,
        deform_num_heads=2,
        num_points=2,
        conv_ffn_hidden_dim=4,
        injector_init_values=0.0,
        num_extra_extractors=1,
        spatial_prior_stem_dim=8,
        spatial_prior_input=spatial_prior_input,
        with_high_resolution_stem=with_high_resolution_stem,
        spatial_prior_channels=spatial_prior_channels,
        output_fusion=output_fusion,
    )


def test_vit_adapter_config_resolves_absolute_dinov2_style_widths():
    assert vit_adapter_config(1024) == {
        'adapter_dim': 1024,
        'deform_attention_dim': 512,
        'deform_num_heads': 16,
        'num_points': 4,
        'conv_ffn_hidden_dim': 256,
        'injector_init_values': 0.0,
        'num_extra_extractors': 2,
        'spatial_prior_stem_dim': 64,
        'spatial_prior_input': 'p1',
        'with_high_resolution_stem': True,
    }


def _patch_xformers_attention(monkeypatch) -> None:
    import pumit.model.vit as vit_module

    monkeypatch.setattr(
        vit_module.xops,
        'memory_efficient_attention',
        lambda query, key, value, attn_bias=None: F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2),
    )


def test_vit_adapter_five_stage_plan_builds_a_three_level_pyramid(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder(num_stages=5).eval()

    assert encoder.level_embeddings.shape[0] == 2
    assert len(encoder.vit_feature_projections) == 3

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 64, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 4, 64, 64, 64),
        (1, 8, 32, 32, 32),
        (1, 12, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 24, 4, 4, 4),
    ]


def test_vit_adapter_rejects_plans_shallower_than_four_stages():
    # encoder_plan_from_architecture_kwargs already refuses <=3 stages, so the adapter's own
    # four-stage minimum is unreachable through real plans.
    with pytest.raises(ValueError, match='more than 3 stages'):
        _tiny_encoder(num_stages=3)


class _TinyTimmPatchEmbed(nn.Module):
    patch_size = (2, 2, 2)

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _TinyTimmBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.seen_rope = None

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.seen_rope = rope
        update = self.projection(tokens)
        if rope is not None:
            update = update + rope[None]
        return tokens + update


class _TinyTimmModel(nn.Module):
    num_prefix_tokens = 1
    no_embed_class = False

    def __init__(self, *, gradient_checkpointing: bool):
        super().__init__()
        self.embed_dim = 8
        self.grad_checkpointing = gradient_checkpointing
        self.patch_embed = _TinyTimmPatchEmbed(self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 5, self.embed_dim))
        self.pos_drop = nn.Identity()
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList(
            _TinyTimmBlock(self.embed_dim)
            for _ in range(4)
        )
        self.norm = nn.LayerNorm(self.embed_dim)


class _TinyTimmInteractiveBackbone(TimmAbsPosInteractiveBackbone):
    def __init__(self, model: _TinyTimmModel, *, with_rope: bool):
        super().__init__(model, (1, 2, 3, 4))
        self.model = model
        self.with_rope = with_rope

    @property
    def _timm_model(self) -> nn.Module:
        return self.model

    def _prepare_block_kwargs(
        self,
        tokens: torch.Tensor,
        patch_shape: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        if not self.with_rope:
            return {}
        assert patch_shape == (2, 3, 4)
        return {'rope': tokens.new_full(tokens.shape[1:], 0.125)}


@pytest.mark.parametrize('gradient_checkpointing', [False, True])
def test_timm_abs_pos_backbone_unifies_token_block_and_norm_flow(
    gradient_checkpointing,
):
    model = _TinyTimmModel(gradient_checkpointing=gradient_checkpointing)
    backbone = _TinyTimmInteractiveBackbone(model, with_rope=True).train()
    x = torch.randn(1, 1, 4, 6, 8, requires_grad=True)

    tokens, patch_shape, block_kwargs = backbone.prepare_tokens(x)

    expected_tokens = model.patch_embed(x)
    expected_tokens = torch.cat((model.cls_token, expected_tokens), dim=1)
    expected_tokens = expected_tokens + lift_2d_position_embedding(
        model.pos_embed,
        patch_shape,
        num_prefix_tokens=1,
    )
    expected_tokens = model.norm_pre(model.patch_drop(model.pos_drop(expected_tokens)))
    assert patch_shape == (2, 3, 4)
    torch.testing.assert_close(tokens, expected_tokens)

    expected_output = tokens
    for block in model.blocks:
        expected_output = block(expected_output, **block_kwargs)
    split_output = backbone.run_blocks(tokens, 0, 2, block_kwargs)
    split_output = backbone.run_blocks(split_output, 2, 4, block_kwargs)
    torch.testing.assert_close(split_output, expected_output)
    assert all(block.seen_rope is block_kwargs['rope'] for block in model.blocks)

    normalized = backbone.normalize_patch_tokens(split_output)
    torch.testing.assert_close(normalized, model.norm(split_output)[:, 1:])
    normalized.square().mean().backward()
    assert x.grad is not None
    assert all(block.projection.weight.grad is not None for block in model.blocks)

    grouped_ids = [
        id(parameter)
        for layer in backbone.parameter_layers()
        for parameter in layer
        if parameter.requires_grad
    ]
    trainable_ids = {
        id(parameter)
        for parameter in backbone.parameters()
        if parameter.requires_grad
    }
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == trainable_ids


def test_timm_abs_pos_backbone_compiles_without_hiding_block_kwargs():
    model = _TinyTimmModel(gradient_checkpointing=False)
    backbone = _TinyTimmInteractiveBackbone(model, with_rope=True).eval()

    def forward(x):
        tokens, _, block_kwargs = backbone.prepare_tokens(x)
        tokens = backbone.run_blocks(tokens, 0, 4, block_kwargs)
        return backbone.normalize_patch_tokens(tokens)

    compiled_forward = torch.compile(forward, backend='eager', fullgraph=True)
    x = torch.randn(1, 1, 4, 6, 8)

    with torch.inference_mode():
        expected = forward(x)
        actual = compiled_forward(x)

    torch.testing.assert_close(actual, expected)


def test_deformable_attention_samples_matching_voxel_centers():
    attention = MultiScaleDeformableAttention3D(
        1,
        1,
        1,
        attention_dim=1,
        num_heads=1,
        num_levels=1,
        num_points=1,
    )
    nn.init.zeros_(attention.sampling_offsets.weight)
    nn.init.zeros_(attention.sampling_offsets.bias)
    nn.init.zeros_(attention.attention_weights.weight)
    nn.init.zeros_(attention.attention_weights.bias)
    nn.init.ones_(attention.value_proj.weight)
    nn.init.zeros_(attention.value_proj.bias)
    nn.init.ones_(attention.output_proj.weight)
    nn.init.zeros_(attention.output_proj.bias)
    spatial_shape = (2, 3, 4)
    value = torch.arange(24, dtype=torch.float32).reshape(1, 24, 1)
    query = torch.zeros_like(value)
    reference_points = reference_points_3d(
        (spatial_shape,),
        device=query.device,
    )

    output = attention(
        query,
        reference_points,
        value,
        (spatial_shape,),
    )

    torch.testing.assert_close(output, value)


def test_multiscale_features_round_trip_with_level_embeddings():
    features = (
        torch.randn(2, 4, 2, 3, 5),
        torch.randn(2, 4, 1, 2, 3),
    )
    level_embeddings = torch.randn(2, 4)

    tokens, spatial_shapes = flatten_multiscale_features_3d(
        features,
        level_embeddings,
    )
    restored = unflatten_multiscale_features_3d(tokens, spatial_shapes)
    expected_tokens = torch.cat(
        tuple(
            feature.flatten(2).transpose(1, 2) + embedding[None, None]
            for feature, embedding in zip(
                features,
                level_embeddings,
                strict=True,
            )
        ),
        dim=1,
    )

    assert spatial_shapes == ((2, 3, 5), (1, 2, 3))
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    for feature, embedding, output in zip(
        features,
        level_embeddings,
        restored,
        strict=True,
    ):
        torch.testing.assert_close(
            output,
            feature + embedding[None, :, None, None, None],
        )


def test_deformable_attention_keeps_sampling_geometry_in_fp32(monkeypatch):
    attention = MultiScaleDeformableAttention3D(
        1,
        1,
        1,
        attention_dim=1,
        num_heads=1,
        num_levels=1,
        num_points=1,
    ).to(dtype=torch.bfloat16)
    nn.init.zeros_(attention.sampling_offsets.weight)
    nn.init.zeros_(attention.sampling_offsets.bias)
    nn.init.zeros_(attention.attention_weights.weight)
    nn.init.zeros_(attention.attention_weights.bias)
    nn.init.ones_(attention.value_proj.weight)
    nn.init.zeros_(attention.value_proj.bias)
    nn.init.ones_(attention.output_proj.weight)
    nn.init.zeros_(attention.output_proj.bias)

    captured_dtypes = []
    grid_sample = F.grid_sample

    def capture_grid_dtype(input, grid, **kwargs):
        captured_dtypes.append((input.dtype, grid.dtype))
        return grid_sample(input, grid, **kwargs)

    monkeypatch.setattr(F, 'grid_sample', capture_grid_dtype)
    spatial_shape = (2, 3, 4)
    value = torch.arange(24, dtype=torch.bfloat16).reshape(1, 24, 1)
    query = torch.zeros_like(value)
    reference_points = reference_points_3d(
        (spatial_shape,),
        device=query.device,
    )

    output = attention(query, reference_points, value, (spatial_shape,))

    assert captured_dtypes == [(torch.float32, torch.float32)]
    torch.testing.assert_close(output, value)


def test_deformable_attention_backpropagates_across_unequal_levels():
    attention = MultiScaleDeformableAttention3D(
        12,
        10,
        8,
        attention_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=3,
    )
    query = torch.randn(2, 7, 12, requires_grad=True)
    value = torch.randn(2, 28, 10, requires_grad=True)
    reference_points = reference_points_3d(
        ((1, 1, 7),),
        device=query.device,
    ).expand(-1, -1, 2, -1)

    output = attention(
        query,
        reference_points,
        value,
        ((2, 3, 4), (1, 2, 2)),
    )
    output.square().mean().backward()

    assert tuple(output.shape) == (2, 7, 8)
    assert query.grad is not None
    assert value.grad is not None
    assert attention.sampling_offsets.weight.grad is not None
    assert attention.attention_weights.weight.grad is not None


def test_vit_adapter_returns_plan_aligned_pyramid_and_backpropagates(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder().train()
    x = torch.randn(1, 1, 64, 64, 64)

    projection_calls = [0] * len(encoder.vit_feature_projections)
    handles = [
        projection.register_forward_hook(
            lambda _module, _inputs, _output, index=index: projection_calls.__setitem__(
                index,
                projection_calls[index] + 1,
            )
        )
        for index, projection in enumerate(encoder.vit_feature_projections)
    ]

    try:
        skips = encoder(x)
    finally:
        for handle in handles:
            handle.remove()
    sum(skip.mean() for skip in skips).backward()

    assert len(encoder.high_resolution_stem.stages) == 2
    assert len(encoder.spatial_prior.coarse_stages) == 3
    assert encoder.spatial_prior.raw_stem is not None
    assert encoder.spatial_prior.p2_stage[0].out_channels == 12
    assert [stage[0].out_channels for stage in encoder.spatial_prior.coarse_stages] == [
        16,
        24,
        32,
    ]
    assert len(encoder.vit_feature_projections) == 4
    assert projection_calls == [1, 1, 1, 1]
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 4, 64, 64, 64),
        (1, 8, 32, 32, 32),
        (1, 12, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 24, 4, 4, 4),
        (1, 32, 2, 2, 2),
    ]
    assert encoder.backbone.vit.patch_embed.weight.grad is not None
    for interaction in encoder.interactions:
        assert interaction.injector.gamma.grad is not None
        assert interaction.extractor.attention.value_proj.weight.grad is not None
    assert encoder.spatial_prior.raw_stem[0][0].weight.grad is not None
    assert encoder.spatial_prior.p2_stage[0].weight.grad is not None
    assert encoder.spatial_prior.coarse_stages[0][0].weight.grad is not None
    assert encoder.high_resolution_stem.stages[0].blocks[0].conv1.conv.weight.grad is not None

    parameter_layers = tuple(encoder.backbone.parameter_layers())
    grouped_ids = {
        id(parameter)
        for layer in parameter_layers
        for parameter in layer
    }
    backbone_ids = {
        id(parameter)
        for parameter in encoder.backbone.parameters()
        if parameter.requires_grad
    }
    assert grouped_ids == backbone_ids
    assert id(encoder.level_embeddings) not in grouped_ids

    for projection in encoder.vit_feature_projections:
        assert isinstance(projection, PlanAlignedViTProjection3D)
        assert not any(
            isinstance(module, (nn.ConvTranspose3d, nn.MaxPool3d))
            for module in projection.modules()
        )


def test_v3_projection_order_and_spatial_widths_are_explicit(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder(
        spatial_prior_input='p1',
        spatial_prior_channels=(8, 8, 8, 8),
        output_fusion='project-then-add',
    ).train()

    skips = encoder(torch.randn(1, 1, 64, 64, 64))
    sum(skip.mean() for skip in skips).backward()

    assert encoder.output_projections is None
    assert encoder.spatial_output_projections is not None
    assert [
        stage[0].out_channels
        for stage in encoder.spatial_prior.coarse_stages
    ] == [8, 8, 8]
    assert [
        projection.projection.out_channels
        for projection in encoder.vit_feature_projections
    ] == [12, 16, 24, 32]
    assert [
        projection.out_channels
        for projection in encoder.spatial_output_projections
    ] == [12, 16, 24, 32]
    assert [tuple(skip.shape[1:]) for skip in skips] == [
        (4, 64, 64, 64),
        (8, 32, 32, 32),
        (12, 16, 16, 16),
        (16, 8, 8, 8),
        (24, 4, 4, 4),
        (32, 2, 2, 2),
    ]


def test_vit_adapter_follows_anisotropic_plan_strides(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder(anisotropic=True, patch_depth=8).eval()
    projection_targets = []
    handles = [
        projection.register_forward_pre_hook(
            lambda _module, inputs: projection_targets.append(tuple(inputs[1]))
        )
        for projection in encoder.vit_feature_projections
    ]

    try:
        with torch.no_grad():
            skips = encoder(torch.randn(1, 1, 64, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    assert projection_targets == [(32, 16, 16), (16, 8, 8), (16, 4, 4), (8, 2, 2)]
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 4, 64, 64, 64),
        (1, 8, 64, 32, 32),
        (1, 12, 32, 16, 16),
        (1, 16, 16, 8, 8),
        (1, 24, 16, 4, 4),
        (1, 32, 8, 2, 2),
    ]


def test_raw_spatial_prior_is_independent_from_high_resolution_stem(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder(spatial_prior_input='raw').eval()
    x = torch.randn(1, 1, 64, 64, 64)
    seen = {}
    handles = [
        encoder.high_resolution_stem.register_forward_hook(
            lambda _module, _inputs, output: seen.update({'p1': output[-1]})
        ),
        encoder.spatial_prior.raw_stem.register_forward_pre_hook(
            lambda _module, inputs: seen.update({'spatial_input': inputs[0]})
        ),
    ]
    try:
        with torch.no_grad():
            encoder(x)
    finally:
        for handle in handles:
            handle.remove()

    assert seen['spatial_input'] is x
    assert seen['spatial_input'].shape[1] == 1
    assert seen['p1'].shape[1] == 8


def test_shared_p1_spatial_prior_consumes_the_refiner_stem_feature(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder(spatial_prior_input='p1').eval()
    seen = {}
    handles = [
        encoder.high_resolution_stem.register_forward_hook(
            lambda _module, _inputs, output: seen.update({'p1': output[-1]})
        ),
        encoder.spatial_prior.register_forward_pre_hook(
            lambda _module, inputs: seen.update({'spatial_input': inputs[0]})
        ),
    ]
    try:
        with torch.no_grad():
            skips = encoder(torch.randn(1, 1, 64, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    assert seen['spatial_input'] is seen['p1']
    assert encoder.spatial_prior.raw_stem is None
    assert len(encoder.spatial_prior.coarse_stages) == 3
    assert len(skips) == 6


def test_raw_spatial_prior_without_refiner_omits_high_resolution_stem(monkeypatch):
    _patch_xformers_attention(monkeypatch)
    encoder = _tiny_encoder(
        spatial_prior_input='raw',
        with_high_resolution_stem=False,
    ).eval()

    with torch.no_grad():
        features = encoder(torch.randn(1, 1, 64, 64, 64))

    assert encoder.high_resolution_stem is None
    assert encoder.output_channels == (12, 16, 24, 32)
    assert [tuple(feature.shape) for feature in features] == [
        (1, 12, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 24, 4, 4, 4),
        (1, 32, 2, 2, 2),
    ]


def test_shared_p1_spatial_prior_requires_high_resolution_stem():
    with pytest.raises(ValueError, match='requires the high-resolution stem'):
        _tiny_encoder(
            spatial_prior_input='p1',
            with_high_resolution_stem=False,
        )


def test_deformable_attention_rejects_low_precision_reference_geometry():
    attention = MultiScaleDeformableAttention3D(
        1,
        1,
        1,
        attention_dim=1,
        num_heads=1,
        num_levels=1,
        num_points=1,
    ).to(dtype=torch.bfloat16)
    query = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
    reference_points = torch.full((1, 1, 1, 3), 0.5, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match='float32 geometry'):
        attention(query, reference_points, query, ((1, 1, 1),))
