"""Plan-aligned encoder built from 3D ViT-Adapter interactions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dynamic_network_architectures.architectures.unet import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)

from pumit.downstream.seg.adapters.deformable_attention import (
    MultiScaleDeformableAttention3D,
    flatten_multiscale_features_3d,
    reference_points_3d,
    unflatten_multiscale_features_3d,
)
from pumit.downstream.seg.adapters.vit_adapter.backbone import InteractiveViTBackbone
from pumit.downstream.seg.adapters.vit_adapter.interaction import (
    InteractionBlock3D,
)
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes


_HIGH_RESOLUTION_STAGES = 2
_PYRAMID_START_STAGE = _HIGH_RESOLUTION_STAGES
SpatialPriorInput = Literal['raw', 'p1']
OutputFusion = Literal['add-then-project', 'project-then-add']
VIT_ADAPTER_CONFIG_KEYS = frozenset({
    'adapter_dim',
    'deform_attention_dim',
    'deform_num_heads',
    'num_points',
    'conv_ffn_hidden_dim',
    'injector_init_values',
    'num_extra_extractors',
    'spatial_prior_stem_dim',
    'spatial_prior_input',
    'with_high_resolution_stem',
})
VIT_ADAPTER_OPTIONAL_CONFIG_KEYS = frozenset({
    'output_fusion',
    'spatial_prior_channels',
})


def vit_adapter_config(embed_dim: int) -> dict[str, object]:
    """Resolve full-width adapter defaults with a shared P1 spatial prior."""
    if embed_dim <= 0 or embed_dim % 64:
        raise ValueError(f'ViT embed_dim must be positive and divisible by 64, got {embed_dim}')
    return {
        'adapter_dim': embed_dim,
        'deform_attention_dim': embed_dim // 2,
        'deform_num_heads': embed_dim // 64,
        'num_points': 4,
        'conv_ffn_hidden_dim': embed_dim // 4,
        'injector_init_values': 0.0,
        'num_extra_extractors': 2,
        'spatial_prior_stem_dim': 64,
        'spatial_prior_input': 'p1',
        'with_high_resolution_stem': True,
    }


def _spatial_stage(
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    plan: EncoderPlan,
) -> nn.Sequential:
    kernel_size = tuple(int(value) for value in kernel_size)
    stride = tuple(int(value) for value in stride)
    layers: list[nn.Module] = [
        plan.conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=tuple(value // 2 for value in kernel_size),
            bias=plan.conv_bias,
        )
    ]
    if plan.norm_op is not None:
        layers.append(plan.norm_op(out_channels, **(plan.norm_op_kwargs or {})))
    if plan.nonlin is not None:
        layers.append(plan.nonlin(**(plan.nonlin_kwargs or {})))
    return nn.Sequential(*layers)


def _feature_norm(channels: int, plan: EncoderPlan) -> nn.Module:
    if plan.norm_op is None:
        return nn.Identity()
    return plan.norm_op(channels, **(plan.norm_op_kwargs or {}))


class SpatialPriorModule3D(nn.Module):
    """Build P2-P5 priors, then project every level to the adapter width."""

    def __init__(
        self,
        plan: EncoderPlan,
        *,
        input_channels: int,
        adapter_dim: int,
        stem_dim: int,
        input_mode: SpatialPriorInput,
        spatial_channels: Sequence[int] | None = None,
    ):
        super().__init__()
        if input_mode not in {'raw', 'p1'}:
            raise ValueError(f'unsupported spatial-prior input: {input_mode!r}')
        if stem_dim <= 0:
            raise ValueError(f'spatial-prior stem_dim must be positive, got {stem_dim}')
        self.input_mode = input_mode
        if spatial_channels is None:
            spatial_channels = plan.output_channels[_PYRAMID_START_STAGE:]
        else:
            expected_levels = len(plan.output_channels) - _PYRAMID_START_STAGE
            spatial_channels = tuple(int(channels) for channels in spatial_channels)
            if (
                len(spatial_channels) != expected_levels
                or any(channels <= 0 for channels in spatial_channels)
            ):
                raise ValueError(
                    f'spatial_prior_channels must contain '
                    f'{expected_levels} positive values, got '
                    f'{spatial_channels}'
                )

        if input_mode == 'raw':
            self.raw_stem = nn.Sequential(
                _spatial_stage(
                    input_channels,
                    stem_dim,
                    plan.kernel_sizes[1],
                    plan.strides[1],
                    plan,
                ),
                _spatial_stage(
                    stem_dim,
                    stem_dim,
                    plan.kernel_sizes[1],
                    (1, 1, 1),
                    plan,
                ),
                _spatial_stage(
                    stem_dim,
                    stem_dim,
                    plan.kernel_sizes[1],
                    (1, 1, 1),
                    plan,
                ),
                nn.MaxPool3d(
                    kernel_size=plan.kernel_sizes[2],
                    stride=plan.strides[2],
                    padding=tuple(value // 2 for value in plan.kernel_sizes[2]),
                ),
            )
            self.p2_stage = _spatial_stage(
                stem_dim,
                spatial_channels[0],
                (1, 1, 1),
                (1, 1, 1),
                plan,
            )
        else:
            self.raw_stem = None
            self.p2_stage = _spatial_stage(
                plan.output_channels[1],
                spatial_channels[0],
                plan.kernel_sizes[2],
                plan.strides[2],
                plan,
            )

        self.coarse_stages = nn.ModuleList(
            [
                _spatial_stage(
                    spatial_channels[stage - _PYRAMID_START_STAGE - 1],
                    spatial_channels[stage - _PYRAMID_START_STAGE],
                    plan.kernel_sizes[stage],
                    plan.strides[stage],
                    plan,
                )
                for stage in range(_PYRAMID_START_STAGE + 1, len(plan.output_channels))
            ]
        )
        self.level_projections = nn.ModuleList(
            nn.Identity()
            if channels == adapter_dim
            else nn.Conv3d(channels, adapter_dim, kernel_size=1, bias=plan.conv_bias)
            for channels in spatial_channels
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if self.raw_stem is not None:
            x = self.raw_stem(x)
        x = self.p2_stage(x)
        features = [x]
        for stage in self.coarse_stages:
            x = stage(x)
            features.append(x)
        if len(features) != len(self.level_projections):
            raise RuntimeError(
                f'expected {len(self.level_projections)} spatial-prior levels, '
                f'got {len(features)}'
            )
        return tuple(
            projection(feature)
            for projection, feature in zip(
                self.level_projections,
                features,
                strict=True,
            )
        )


class PlanAlignedViTProjection3D(nn.Module):
    """Resize one ViT interaction feature to a plan stage and project its channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        conv_bias: bool,
        force_projection: bool = False,
    ):
        super().__init__()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels and not force_projection
            else nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=conv_bias,
            )
        )

    def forward(self, x: Tensor, target_shape: Sequence[int]) -> Tensor:
        target_shape = tuple(int(value) for value in target_shape)
        if len(target_shape) != 3 or any(value <= 0 for value in target_shape):
            raise ValueError(f'invalid 3D target shape: {target_shape}')
        if tuple(x.shape[2:]) != target_shape:
            x = F.interpolate(
                x,
                size=target_shape,
                mode='trilinear',
                align_corners=False,
            )
        return self.projection(x)


class ViTAdapterEncoder3D(nn.Module):
    """Produce a plan-aligned pyramid through repeated 3D spatial/ViT interaction."""

    def __init__(
        self,
        backbone: InteractiveViTBackbone,
        plan: EncoderPlan,
        *,
        input_channels: int,
        adapter_dim: int,
        deform_attention_dim: int,
        deform_num_heads: int,
        num_points: int,
        conv_ffn_hidden_dim: int,
        injector_init_values: float,
        num_extra_extractors: int,
        spatial_prior_stem_dim: int,
        spatial_prior_input: SpatialPriorInput,
        with_high_resolution_stem: bool,
        spatial_prior_channels: Sequence[int] | None = None,
        output_fusion: OutputFusion = 'add-then-project',
    ):
        super().__init__()
        if len(plan.output_channels) < 4:
            raise ValueError(
                f'ViT-Adapter requires at least four nnU-Net encoder stages '
                f'(two high-resolution stages plus a pyramid), got {len(plan.output_channels)}'
            )
        if len(backbone.feature_layers) != 4:
            raise ValueError(
                f'ViT-Adapter requires four interaction depths, got {backbone.feature_layers}'
            )
        if (
            adapter_dim <= 0
            or deform_attention_dim <= 0
            or deform_num_heads <= 0
            or num_points <= 0
        ):
            raise ValueError('adapter dimensions, heads, and sampling points must be positive')
        if deform_attention_dim % deform_num_heads:
            raise ValueError(
                f'deform_attention_dim {deform_attention_dim} must be divisible by '
                f'deform_num_heads {deform_num_heads}'
            )
        if conv_ffn_hidden_dim <= 0:
            raise ValueError('conv_ffn_hidden_dim must be positive')
        if num_extra_extractors < 0:
            raise ValueError('num_extra_extractors must be non-negative')
        if spatial_prior_input not in {'raw', 'p1'}:
            raise ValueError(f'unsupported spatial-prior input: {spatial_prior_input!r}')
        if spatial_prior_input == 'p1' and not with_high_resolution_stem:
            raise ValueError('P1 spatial-prior input requires the high-resolution stem')
        if output_fusion not in {'add-then-project', 'project-then-add'}:
            raise ValueError(f'unsupported ViT-Adapter output fusion: {output_fusion!r}')

        self.backbone = backbone
        self.input_channels = input_channels
        self.spatial_prior_input = spatial_prior_input
        self.with_high_resolution_stem = with_high_resolution_stem
        self.output_fusion = output_fusion
        adopt_plan_contract(self, plan, output_start=0 if with_high_resolution_stem else _PYRAMID_START_STAGE)
        self.high_resolution_stem = (
            ResidualEncoder(
                input_channels=input_channels,
                n_stages=_HIGH_RESOLUTION_STAGES,
                features_per_stage=plan.output_channels[:_HIGH_RESOLUTION_STAGES],
                conv_op=plan.conv_op,
                kernel_sizes=plan.kernel_sizes[:_HIGH_RESOLUTION_STAGES],
                strides=plan.strides[:_HIGH_RESOLUTION_STAGES],
                n_blocks_per_stage=plan.n_blocks_per_stage[:_HIGH_RESOLUTION_STAGES],
                conv_bias=plan.conv_bias,
                norm_op=plan.norm_op,
                norm_op_kwargs=plan.norm_op_kwargs,
                dropout_op=plan.dropout_op,
                dropout_op_kwargs=plan.dropout_op_kwargs,
                nonlin=plan.nonlin,
                nonlin_kwargs=plan.nonlin_kwargs,
                return_skips=True,
            )
            if with_high_resolution_stem
            else None
        )
        self.spatial_prior = SpatialPriorModule3D(
            plan,
            input_channels=input_channels,
            adapter_dim=adapter_dim,
            stem_dim=spatial_prior_stem_dim,
            input_mode=spatial_prior_input,
            spatial_channels=spatial_prior_channels,
        )
        self.level_embeddings = nn.Parameter(
            torch.empty(len(plan.output_channels) - _PYRAMID_START_STAGE - 1, adapter_dim)
        )
        self.interactions = nn.ModuleList(
            InteractionBlock3D(
                backbone.embed_dim,
                adapter_dim,
                attention_dim=deform_attention_dim,
                num_heads=deform_num_heads,
                num_points=num_points,
                num_spatial_levels=len(plan.output_channels) - _PYRAMID_START_STAGE - 1,
                conv_ffn_hidden_dim=conv_ffn_hidden_dim,
                injector_init_values=injector_init_values,
                num_extra_extractors=(
                    num_extra_extractors
                    if interaction_index == len(backbone.feature_layers) - 1
                    else 0
                ),
                gradient_checkpointing=backbone.gradient_checkpointing,
            )
            for interaction_index in range(len(backbone.feature_layers))
        )
        self.vit_feature_projections = nn.ModuleList(
            PlanAlignedViTProjection3D(
                backbone.embed_dim,
                (
                    adapter_dim
                    if output_fusion == 'add-then-project'
                    else out_channels
                ),
                conv_bias=plan.conv_bias,
                force_projection=output_fusion == 'project-then-add',
            )
            for out_channels in plan.output_channels[_PYRAMID_START_STAGE:]
        )
        c2_to_c1_stride = tuple(int(value) for value in plan.strides[_PYRAMID_START_STAGE + 1])
        self.c1_from_spatial = nn.ConvTranspose3d(
            adapter_dim,
            adapter_dim,
            kernel_size=c2_to_c1_stride,
            stride=c2_to_c1_stride,
            bias=plan.conv_bias,
        )
        self.spatial_output_projections = (
            nn.ModuleList(
                nn.Conv3d(
                    adapter_dim,
                    out_channels,
                    kernel_size=1,
                    bias=plan.conv_bias,
                )
                for out_channels in plan.output_channels[_PYRAMID_START_STAGE:]
            )
            if output_fusion == 'project-then-add'
            else None
        )
        self.output_projections = (
            nn.ModuleList(
                nn.Identity()
                if adapter_dim == out_channels
                else nn.Conv3d(
                    adapter_dim,
                    out_channels,
                    kernel_size=1,
                    bias=plan.conv_bias,
                )
                for out_channels in plan.output_channels[_PYRAMID_START_STAGE:]
            )
            if output_fusion == 'add-then-project'
            else None
        )
        self.output_norms = nn.ModuleList(
            _feature_norm(channels, plan)
            for channels in plan.output_channels[_PYRAMID_START_STAGE:]
        )


        initializer = InitWeights_He(1e-2)
        if self.high_resolution_stem is not None:
            self.high_resolution_stem.apply(initializer)
            self.high_resolution_stem.apply(init_last_bn_before_add_to_0)
        self.spatial_prior.apply(initializer)
        self.vit_feature_projections.apply(initializer)
        self.c1_from_spatial.apply(initializer)
        if self.spatial_output_projections is not None:
            self.spatial_output_projections.apply(initializer)
        if self.output_projections is not None:
            self.output_projections.apply(initializer)
        self.interactions.apply(self._init_transformer_module)
        for module in self.interactions.modules():
            if isinstance(module, MultiScaleDeformableAttention3D):
                module.reset_parameters()
        nn.init.normal_(self.level_embeddings, std=0.02)

    @staticmethod
    def _init_transformer_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects '
                f'{self.input_channels}'
            )

        input_shape = tuple(int(value) for value in x.shape[2:])
        stage_shapes = compute_stage_shapes(input_shape, self.cumulative_strides)
        high_resolution_skips = (
            tuple(self.high_resolution_stem(x))
            if self.high_resolution_stem is not None
            else ()
        )
        spatial_prior_input = (
            x
            if self.spatial_prior_input == 'raw'
            else high_resolution_skips[-1]
        )
        spatial_features = self.spatial_prior(spatial_prior_input)
        spatial_tokens, spatial_shapes = flatten_multiscale_features_3d(
            spatial_features[1:],
            self.level_embeddings,
        )

        tokens, patch_shape, block_context = self.backbone.prepare_tokens(x)
        patch_reference_points = reference_points_3d(
            (patch_shape,),
            device=x.device,
        ).expand(-1, -1, len(spatial_shapes), -1)
        spatial_reference_points = reference_points_3d(
            spatial_shapes,
            device=x.device,
        )

        vit_features = []
        block_start = 0
        for interaction, block_end in zip(
            self.interactions,
            self.backbone.feature_layers,
            strict=True,
        ):
            prefix_tokens = tokens[:, :self.backbone.num_prefix_tokens]
            patch_tokens = tokens[:, self.backbone.num_prefix_tokens:]
            patch_tokens = interaction.injector(
                patch_tokens,
                patch_reference_points,
                spatial_tokens,
                spatial_shapes,
            )
            tokens = torch.cat((prefix_tokens, patch_tokens), dim=1)
            tokens = self.backbone.run_blocks(
                tokens,
                block_start,
                block_end,
                block_context,
            )
            patch_tokens = tokens[:, self.backbone.num_prefix_tokens:]
            spatial_tokens = interaction.extractor(
                spatial_tokens,
                spatial_reference_points,
                patch_tokens,
                spatial_shapes,
                patch_shape,
            )
            for extractor in interaction.extra_extractors:
                spatial_tokens = extractor(
                    spatial_tokens,
                    spatial_reference_points,
                    patch_tokens,
                    spatial_shapes,
                    patch_shape,
                )
            normalized_patch_tokens = self.backbone.normalize_patch_tokens(tokens)
            vit_features.append(
                normalized_patch_tokens.transpose(1, 2).reshape(
                    x.shape[0],
                    self.backbone.embed_dim,
                    *patch_shape,
                )
            )
            block_start = block_end

        updated_spatial_features = unflatten_multiscale_features_3d(
            spatial_tokens,
            spatial_shapes,
        )
        spatial_features = (
            spatial_features[0],
            *updated_spatial_features,
        )
        spatial_features = (
            spatial_features[0] + self.c1_from_spatial(spatial_features[1]),
            *spatial_features[1:],
        )
        # Interactions always run at four ViT depths; on plans shallower than six stages only the
        # deepest interaction outputs feed the pyramid (the shallow-to-fine mapping is preserved).
        pyramid_vit_features = vit_features[len(vit_features) - len(self.vit_feature_projections):]
        vit_pyramid = [
            projection(feature, target_shape)
            for projection, feature, target_shape in zip(
                self.vit_feature_projections,
                pyramid_vit_features,
                stage_shapes[_PYRAMID_START_STAGE:],
                strict=True,
            )
        ]
        if self.output_fusion == 'add-then-project':
            if self.output_projections is None:
                raise RuntimeError('add-then-project output projections are missing')
            pyramid = [
                norm(projection(adapter_feature + vit_feature))
                for norm, projection, adapter_feature, vit_feature in zip(
                    self.output_norms,
                    self.output_projections,
                    spatial_features,
                    vit_pyramid,
                    strict=True,
                )
            ]
        else:
            if self.spatial_output_projections is None:
                raise RuntimeError('project-then-add spatial projections are missing')
            pyramid = [
                norm(spatial_projection(adapter_feature) + vit_feature)
                for norm, spatial_projection, adapter_feature, vit_feature in zip(
                    self.output_norms,
                    self.spatial_output_projections,
                    spatial_features,
                    vit_pyramid,
                    strict=True,
                )
            ]
        skips = [
            *high_resolution_skips,
            *pyramid,
        ]
        target_shapes = (
            stage_shapes
            if self.with_high_resolution_stem
            else stage_shapes[_PYRAMID_START_STAGE:]
        )
        for stage, (skip, target_shape) in enumerate(
            zip(skips, target_shapes, strict=True)
        ):
            if tuple(skip.shape[2:]) != target_shape:
                raise RuntimeError(
                    f'encoder stage {stage} shape {tuple(skip.shape[2:])} does not match '
                    f'nnU-Net target shape {target_shape}'
                )
        return skips


def build_vit_adapter_encoder(
    backbone: InteractiveViTBackbone,
    plan: EncoderPlan,
    *,
    input_channels: int,
    config: Mapping[str, object],
) -> ViTAdapterEncoder3D:
    """Build the shared adapter after a model-specific backbone has been validated."""
    missing = VIT_ADAPTER_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f'ViT-Adapter config is missing keys: {sorted(missing)}')
    spatial_prior_input = config['spatial_prior_input']
    if spatial_prior_input not in {'raw', 'p1'}:
        raise ValueError(f'unsupported spatial-prior input: {spatial_prior_input!r}')
    with_high_resolution_stem = config['with_high_resolution_stem']
    if not isinstance(with_high_resolution_stem, bool):
        raise TypeError(
            f'with_high_resolution_stem must be bool, '
            f'got {type(with_high_resolution_stem).__name__}'
        )
    return ViTAdapterEncoder3D(
        backbone,
        plan,
        input_channels=input_channels,
        adapter_dim=int(config['adapter_dim']),
        deform_attention_dim=int(config['deform_attention_dim']),
        deform_num_heads=int(config['deform_num_heads']),
        num_points=int(config['num_points']),
        conv_ffn_hidden_dim=int(config['conv_ffn_hidden_dim']),
        injector_init_values=float(config['injector_init_values']),
        num_extra_extractors=int(config['num_extra_extractors']),
        spatial_prior_stem_dim=int(config['spatial_prior_stem_dim']),
        spatial_prior_input=cast(SpatialPriorInput, spatial_prior_input),
        with_high_resolution_stem=with_high_resolution_stem,
        spatial_prior_channels=config.get('spatial_prior_channels'),
        output_fusion=cast(
            OutputFusion,
            config.get('output_fusion', 'add-then-project'),
        ),
    )
