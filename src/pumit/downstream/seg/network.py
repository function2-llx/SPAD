"""Complete plan-aligned segmentation networks using nnU-Net's stock decoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch import Tensor, nn
from torch.nn import functional as F

from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from .plan import (
    compute_stage_shapes,
    encoder_plan_from_architecture_kwargs,
)


class UNetDecoderWithoutHighResolutionRefiner(UNetDecoder):
    """Decode the encoder's levels from the coarsest down to P2, then resize the P2 prediction to P0.

    The encoder exposes plan stages P2 onward (four levels on a six-stage plan, three on a five-stage plan); the
    deep-supervision outputs at P0 and P1 are resized copies of the P2 prediction.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        n_conv_per_stage: Sequence[int],
        deep_supervision: bool,
    ):
        n_levels = len(encoder.output_channels)
        if n_levels < 2:
            raise ValueError(
                f'U-Net without its high-resolution refiner requires at least P2 and P3, got {n_levels} levels'
            )
        super().__init__(
            encoder,
            num_classes,
            tuple(n_conv_per_stage)[:n_levels - 1],
            deep_supervision,
        )
        self.n_levels = n_levels

    @staticmethod
    def _resize(logits: Tensor, shape: Sequence[int]) -> Tensor:
        return F.interpolate(
            logits,
            size=tuple(int(value) for value in shape),
            mode='trilinear',
            align_corners=False,
        )

    def forward(
        self,
        skips: Sequence[Tensor],
        target_shapes: Sequence[Sequence[int]],
    ) -> Tensor | list[Tensor]:
        skips = tuple(skips)
        if len(skips) != self.n_levels:
            raise ValueError(f'U-Net decoder requires {self.n_levels} levels from P2, got {len(skips)}')
        target_shapes = tuple(tuple(int(value) for value in shape) for shape in target_shapes)
        if len(target_shapes) != self.n_levels + 1:
            raise ValueError(
                f'U-Net deep supervision requires the {self.n_levels + 1} shapes P0-P{self.n_levels}, '
                f'got {target_shapes}'
            )

        predictions = super().forward(skips)
        if not self.deep_supervision:
            return self._resize(predictions, target_shapes[0])
        final = predictions[0]
        return [
            self._resize(final, target_shapes[0]),
            self._resize(final, target_shapes[1]),
            *predictions,
        ]


class DenseSegmentationNetwork(nn.Module):
    """Compose a complete plan-aligned encoder with nnU-Net's stock decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_classes: int,
        n_conv_per_stage_decoder: Sequence[int],
        deep_supervision: bool,
        decoder_class: type[UNetDecoder] = UNetDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder_class(
            encoder,
            num_classes=num_classes,
            n_conv_per_stage=tuple(int(value) for value in n_conv_per_stage_decoder),
            deep_supervision=deep_supervision,
        )

        initializer = InitWeights_He(1e-2)
        self.decoder.stages.apply(initializer)
        self.decoder.transpconvs.apply(initializer)
        self.decoder.seg_layers.apply(initializer)
        if not deep_supervision:
            # The stock decoder builds one segmentation head per stage but only uses the last one without deep
            # supervision. The idle heads keep their place (checkpoint layout) but leave the trainable set, which
            # DDP would otherwise reject for receiving no gradient.
            for head in self.decoder.seg_layers[:-1]:
                head.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        return self.decoder(self.encoder(x))


class EncoderPretrainedMixin:
    """Initialize the registered encoder while retaining the readout's random weights."""

    backbone_name: str
    encoder: nn.Module

    def load_pretrained(self, weights: Path | None) -> None:
        from .registry import load_pretrained

        load_pretrained(self.backbone_name, self.encoder, weights)


class PlanAlignedSegmentationNetwork(EncoderPretrainedMixin, DenseSegmentationNetwork):
    """Native nnU-Net architecture entry point for one registered dense encoder."""

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
        deep_supervision: bool,
    ):
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
        encoder = build_encoder(backbone_name, plan, input_channels, backbone_config)
        if len(encoder.output_channels) != len(plan.output_channels):
            raise ValueError(
                f'full U-Net readout requires one encoder level per plan stage, got '
                f'{len(encoder.output_channels)} levels for {len(plan.output_channels)} stages'
            )
        super().__init__(
            encoder,
            num_classes=num_classes,
            n_conv_per_stage_decoder=plan.n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )
        self.backbone_name = backbone_name


class PlanAlignedUNetWithoutRefinerSegmentationNetwork(EncoderPretrainedMixin, DenseSegmentationNetwork):
    """Plan-aligned encoder with only the coarsest-to-P2 part of the U-Net decoder."""

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
        deep_supervision: bool,
    ):
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
        encoder = build_encoder(backbone_name, plan, input_channels, backbone_config)
        super().__init__(
            encoder,
            num_classes=num_classes,
            n_conv_per_stage_decoder=plan.n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            decoder_class=UNetDecoderWithoutHighResolutionRefiner,
        )
        self.cumulative_strides = plan.cumulative_strides
        self.backbone_name = backbone_name

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        target_shapes = compute_stage_shapes(x.shape[2:], self.cumulative_strides)[:self.decoder.n_levels + 1]
        return self.decoder(self.encoder(x), target_shapes)
