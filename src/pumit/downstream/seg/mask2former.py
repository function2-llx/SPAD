"""Fixed semantic-query 3D Mask2Former for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pumit.ucpt.seg.pos_embed import sine_pos_embed_3d

from .adapters.deformable_attention import (
    MultiScaleDeformableAttention3D,
    flatten_multiscale_features_3d,
    reference_points_3d,
    unflatten_multiscale_features_3d,
)
from .network import EncoderPretrainedMixin
from .plan import encoder_plan_from_architecture_kwargs


_HIDDEN_DIM = 256
_MASK_DIM = 256
_NUM_HEADS = 8
_NUM_DECODER_LAYERS = 9
_DECODER_FFN_DIM = 2048
_PIXEL_ENCODER_LAYERS = 6
_PIXEL_ENCODER_FFN_DIM = 1024
_NUM_FEATURE_LEVELS = 3
_NUM_DEFORMABLE_POINTS = 4
MASK2FORMER_NUM_OUTPUTS = _NUM_DECODER_LAYERS + 1
MaskAttentionMode = Literal['classes', 'regions']
QueryFeatureSchedule = Literal['concat', 'cycle']


def _interpolate(x: Tensor, shape: Sequence[int]) -> Tensor:
    target_shape = tuple(int(value) for value in shape)
    if tuple(x.shape[2:]) == target_shape:
        return x
    return F.interpolate(
        x,
        size=target_shape,
        mode='trilinear',
        align_corners=False,
    )


def position_encoding_3d(
    x: Tensor,
    dim: int,
) -> Tensor:
    """Build the project's additive-depth sine encoding over a ``(D, H, W)`` grid."""
    if x.ndim != 5:
        raise ValueError(f'expected [B, C, D, H, W] feature, got {tuple(x.shape)}')
    depth, height, width = x.shape[2:]
    position = sine_pos_embed_3d(
        depth,
        height,
        width,
        dim,
        x.device,
    )
    return (
        position.reshape(depth, height, width, dim)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .expand(x.shape[0], -1, -1, -1, -1)
        .to(x.dtype)
    )


def _group_norm(channels: int) -> nn.GroupNorm:
    if channels % 32:
        raise ValueError(f'Mask2Former feature width must be divisible by 32, got {channels}')
    return nn.GroupNorm(32, channels)


class DeformablePixelEncoderLayer3D(nn.Module):
    """One post-norm multi-scale deformable self-attention encoder layer."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        num_levels: int,
        num_points: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.self_attention = MultiScaleDeformableAttention3D(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            attention_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        memory: Tensor,
        position: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        update = self.self_attention(
            memory + position,
            reference_points,
            memory,
            spatial_shapes,
        )
        memory = self.attention_norm(memory + update)
        return self.ffn_norm(memory + self.ffn(memory))


class Mask2FormerPixelDecoder3D(nn.Module):
    """Fuse P2-P5 and expose P2 mask features plus encoded P5-P3 levels."""

    def __init__(
        self,
        input_channels: Sequence[int],
        *,
        hidden_dim: int = _HIDDEN_DIM,
        mask_dim: int = _MASK_DIM,
        num_heads: int = _NUM_HEADS,
        num_encoder_layers: int = _PIXEL_ENCODER_LAYERS,
        ffn_dim: int = _PIXEL_ENCODER_FFN_DIM,
        num_points: int = _NUM_DEFORMABLE_POINTS,
    ):
        super().__init__()
        input_channels = tuple(int(value) for value in input_channels)
        if len(input_channels) != 4:
            raise ValueError(f'Mask2Former pixel decoder requires P2-P5, got {input_channels}')
        if num_encoder_layers <= 0:
            raise ValueError('pixel encoder must contain at least one layer')

        # The deformable encoder consumes low-to-high resolution P5, P4, P3.
        self.input_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(channels, hidden_dim, kernel_size=1),
                _group_norm(hidden_dim),
            )
            for channels in reversed(input_channels[1:])
        )
        self.level_embeddings = nn.Parameter(
            torch.empty(_NUM_FEATURE_LEVELS, hidden_dim)
        )
        self.encoder_layers = nn.ModuleList(
            DeformablePixelEncoderLayer3D(
                hidden_dim,
                num_heads=num_heads,
                num_levels=_NUM_FEATURE_LEVELS,
                num_points=num_points,
                ffn_dim=ffn_dim,
            )
            for _ in range(num_encoder_layers)
        )
        self.p2_lateral = nn.Sequential(
            nn.Conv3d(input_channels[0], hidden_dim, kernel_size=1),
            _group_norm(hidden_dim),
        )
        self.p2_output = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            _group_norm(hidden_dim),
            nn.ReLU(),
        )
        self.mask_projection = nn.Conv3d(hidden_dim, mask_dim, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embeddings, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Conv3d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm | nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, MultiScaleDeformableAttention3D):
                module.reset_parameters()

    def forward(
        self,
        features: Sequence[Tensor],
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        features = tuple(features)
        if len(features) != 4:
            raise ValueError(f'Mask2Former pixel decoder requires four features, got {len(features)}')

        projected = tuple(
            projection(feature)
            for projection, feature in zip(
                self.input_projections,
                reversed(features[1:]),
                strict=True,
            )
        )
        positions = []
        for feature in projected:
            position = position_encoding_3d(feature, feature.shape[1])
            positions.append(position)
        memory, spatial_shapes = flatten_multiscale_features_3d(projected)
        position, _ = flatten_multiscale_features_3d(
            positions,
            self.level_embeddings,
        )
        reference_points = reference_points_3d(
            spatial_shapes,
            device=memory.device,
        ).expand(-1, -1, _NUM_FEATURE_LEVELS, -1)
        for layer in self.encoder_layers:
            memory = layer(
                memory,
                position,
                reference_points,
                spatial_shapes,
            )

        multi_scale_features = unflatten_multiscale_features_3d(
            memory,
            spatial_shapes,
        )

        p2 = self.p2_lateral(features[0])
        p2 = self.p2_output(p2 + _interpolate(multi_scale_features[-1], p2.shape[2:]))
        return self.mask_projection(p2), tuple(multi_scale_features)


class Mask2FormerDecoderLayer3D(nn.Module):
    """One post-norm masked cross-attention, self-attention, and FFN layer."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: Tensor,
        query_positions: Tensor,
        memory: Tensor,
        memory_positions: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        cross_update = self.cross_attention(
            queries + query_positions,
            memory + memory_positions,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        queries = self.cross_norm(queries + cross_update)
        self_update = self.self_attention(
            queries + query_positions,
            queries + query_positions,
            queries,
            need_weights=False,
        )[0]
        queries = self.self_norm(queries + self_update)
        return self.ffn_norm(queries + self.ffn(queries))


class MaskEmbeddingMLP(nn.Module):
    def __init__(self, hidden_dim: int, mask_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, mask_dim),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class FixedSemanticMask2FormerDecoder3D(nn.Module):
    """Predict class-aligned masks with Mask2Former's multi-scale masked attention."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_classes: int,
        hidden_dim: int = _HIDDEN_DIM,
        mask_dim: int = _MASK_DIM,
        num_heads: int = _NUM_HEADS,
        num_layers: int = _NUM_DECODER_LAYERS,
        ffn_dim: int = _DECODER_FFN_DIM,
        mask_attention_mode: MaskAttentionMode = 'regions',
        query_feature_schedule: QueryFeatureSchedule = 'concat',
        deep_supervision: bool,
    ):
        super().__init__()
        if len(encoder.output_channels) != 4:
            raise ValueError(
                f'Mask2Former requires P2-P5, '
                f'got {len(encoder.output_channels)} levels'
            )
        if num_classes <= 0 or num_layers <= 0:
            raise ValueError('classes and decoder layers must be positive')
        if hidden_dim % num_heads:
            raise ValueError(f'hidden_dim {hidden_dim} must be divisible by {num_heads}')
        if mask_attention_mode not in {'classes', 'regions'}:
            raise ValueError(f'unsupported mask attention mode: {mask_attention_mode!r}')
        if query_feature_schedule not in {'concat', 'cycle'}:
            raise ValueError(f'unsupported query feature schedule: {query_feature_schedule!r}')
        self.deep_supervision = deep_supervision
        self.num_heads = num_heads
        self.mask_attention_mode = mask_attention_mode
        self.query_feature_schedule = query_feature_schedule
        self.pixel_decoder = Mask2FormerPixelDecoder3D(
            encoder.output_channels,
            hidden_dim=hidden_dim,
            mask_dim=mask_dim,
            num_heads=num_heads,
        )
        self.level_embeddings = nn.Parameter(
            torch.empty(_NUM_FEATURE_LEVELS, hidden_dim)
        )
        self.query_features = nn.Embedding(num_classes, hidden_dim)
        self.query_positions = nn.Embedding(num_classes, hidden_dim)
        self.layers = nn.ModuleList(
            Mask2FormerDecoderLayer3D(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mask_embedding = MaskEmbeddingMLP(hidden_dim, mask_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embeddings, std=0.02)
        nn.init.normal_(self.query_features.weight, std=0.02)
        nn.init.normal_(self.query_positions.weight, std=0.02)
        for module in (
            self.layers,
            self.output_norm,
            self.mask_embedding,
        ):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.LayerNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)

    def _predict_masks(self, queries: Tensor, mask_features: Tensor) -> Tensor:
        mask_embeddings = self.mask_embedding(self.output_norm(queries))
        return torch.einsum('bqc,bcdhw->bqdhw', mask_embeddings, mask_features)

    def _attention_mask(
        self,
        mask_logits: Tensor,
        target_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        attention_mask = torch.cat(
            tuple(
                self._attention_mask_at_shape(mask_logits, target_shape)
                for target_shape in target_shapes
            ),
            dim=-1,
        )
        attention_mask = attention_mask.detach().repeat_interleave(
            self.num_heads,
            dim=0,
        )
        fully_masked = attention_mask.all(dim=-1)
        attention_mask[fully_masked] = False
        return attention_mask

    def _attention_mask_at_shape(
        self,
        mask_logits: Tensor,
        target_shape: Sequence[int],
    ) -> Tensor:
        resized = _interpolate(mask_logits, target_shape)
        if self.mask_attention_mode == 'regions':
            return resized.sigmoid().flatten(2) < 0.5

        winning_query = resized.argmax(dim=1, keepdim=True)
        query_index = torch.arange(
            resized.shape[1],
            device=resized.device,
        ).view(1, -1, 1, 1, 1)
        return (query_index != winning_query).flatten(2)

    def forward(
        self,
        skips: Sequence[Tensor],
        output_shape: Sequence[int],
    ) -> Tensor | list[Tensor]:
        skips = tuple(skips)
        if len(skips) != 4:
            raise ValueError(f'Mask2Former decoder requires P2-P5, got {len(skips)} levels')
        mask_features, multi_scale_features = self.pixel_decoder(skips)
        memories = []
        memory_positions = []
        memory_shapes = []
        for level, feature in enumerate(multi_scale_features):
            memory_shapes.append(tuple(int(value) for value in feature.shape[2:]))
            memories.append(
                feature.flatten(2).transpose(1, 2)
                + self.level_embeddings[level][None, None]
            )
            position = position_encoding_3d(feature, feature.shape[1])
            memory_positions.append(position.flatten(2).transpose(1, 2))
        if self.query_feature_schedule == 'concat':
            memory = torch.cat(memories, dim=1)
            memory_position = torch.cat(memory_positions, dim=1)

        batch_size = mask_features.shape[0]
        queries = self.query_features.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_positions = self.query_positions.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        predictions = [self._predict_masks(queries, mask_features)]
        for layer_index, layer in enumerate(self.layers):
            if self.query_feature_schedule == 'cycle':
                level = layer_index % _NUM_FEATURE_LEVELS
                layer_memory = memories[level]
                layer_memory_position = memory_positions[level]
                target_shapes = (memory_shapes[level],)
            else:
                layer_memory = memory
                layer_memory_position = memory_position
                target_shapes = memory_shapes
            queries = layer(
                queries,
                query_positions,
                layer_memory,
                layer_memory_position,
                self._attention_mask(predictions[-1], target_shapes),
            )
            predictions.append(self._predict_masks(queries, mask_features))

        if not self.deep_supervision:
            return _interpolate(predictions[-1], output_shape)
        return [
            _interpolate(prediction, output_shape)
            for prediction in reversed(predictions)
        ]


class PlanAlignedMask2FormerSegmentationNetwork(EncoderPretrainedMixin, nn.Module):
    """Native nnU-Net entry point for an encoder plus fixed-query 3D Mask2Former."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        backbone_name: str,
        backbone_config: Mapping[str, object],
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        n_conv_per_stage_decoder: Sequence[int],
        conv_op: type[nn.Module],
        conv_bias: bool,
        norm_op: type[nn.Module] | None,
        norm_op_kwargs: Mapping[str, object] | None,
        dropout_op: type[nn.Module] | None,
        dropout_op_kwargs: Mapping[str, object] | None,
        nonlin: type[nn.Module] | None,
        nonlin_kwargs: Mapping[str, object] | None,
        mask_attention_mode: MaskAttentionMode = 'regions',
        query_feature_schedule: QueryFeatureSchedule = 'concat',
        deep_supervision: bool,
    ):
        super().__init__()
        from .registry import build_encoder

        plan = encoder_plan_from_architecture_kwargs(
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.encoder = build_encoder(backbone_name, plan, input_channels, backbone_config)
        self.backbone_name = backbone_name
        self.decoder = FixedSemanticMask2FormerDecoder3D(
            self.encoder,
            num_classes=num_classes,
            mask_attention_mode=mask_attention_mode,
            query_feature_schedule=query_feature_schedule,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        return self.decoder(self.encoder(x), x.shape[2:])
