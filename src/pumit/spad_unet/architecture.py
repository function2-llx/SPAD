"""Plan-constructible ResEnc SPAD U-Net with DA-aware spatial execution."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn import functional as F

from pumit.spadop.conv import SPADConv3d_K3S1, SPADConv3d_K3S2, SPADConvTranspose3d_K2S2
from pumit.spad_unet.geometry import (
    SUPPORTED_FGC_STAGES,
    compute_da_schedule,
    compute_stage_shapes,
    compute_stride_schedule,
)
from pumit.spad_unet.task_aware_bottleneck import (
    TAB_ATTENTION_DOWNSAMPLE_RATE,
    TAB_DEPTH,
    TAB_DIM,
    TAB_FOURIER_SCALE,
    TAB_FOURIER_SEED,
    TAB_HEADS,
    TAB_MLP_DIM,
    TAB_TOKENS_PER_DATASET,
    MultiScaleTaskAwareBottleneck,
    TaskAwareBottleneck,
)

SPAD_NETWORK_CLASS_NAME = 'pumit.spad_unet.architecture.SPADResEncUNet'
UNIVERSAL_SPAD_NETWORK_CLASS_NAME = (
    'pumit.spad_unet.architecture.UniversalSPADResEncUNet'
)
_RESENC_NETWORK_CLASS_NAME = (
    'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
)
FGC_RETURN_EARLY = 'early'
FGC_RETURN_LATE = 'late'
FGC_RETURN_MODES = (
    FGC_RETURN_EARLY,
    FGC_RETURN_LATE,
)
FGC_RETURN_DOWNSAMPLE_TRILINEAR = 'trilinear'
FGC_RETURN_DOWNSAMPLE_AREA = 'area'
FGC_RETURN_DOWNSAMPLE_DEFAULT = FGC_RETURN_DOWNSAMPLE_AREA
FGC_RETURN_DOWNSAMPLE_MODES = (
    FGC_RETURN_DOWNSAMPLE_AREA,
    FGC_RETURN_DOWNSAMPLE_TRILINEAR,
)
_LEGACY_FGC_RETURN_MODE = 'post_upconv'
_RUNTIME_FGC_RETURN_MODES = (*FGC_RETURN_MODES, _LEGACY_FGC_RETURN_MODE)


def resolve_fgc_return_mode(
    stage: int | None,
    return_mode: str | None,
) -> str | None:
    """Validate FGC placement and resolve the legacy S2 return mode."""
    if stage is None:
        if return_mode is not None:
            raise ValueError('FGC return mode requires an FGC stage')
        return None
    if stage not in SUPPORTED_FGC_STAGES:
        raise ValueError(
            f'FGC stage must be one of {SUPPORTED_FGC_STAGES}, '
            f'got {stage!r}'
        )
    if return_mode is None:
        # Original FGC@S2 plans predate the explicit decoder-return field.
        if stage != 2:
            raise ValueError('non-legacy FGC placements require an explicit return mode')
        return _LEGACY_FGC_RETURN_MODE
    if return_mode not in _RUNTIME_FGC_RETURN_MODES:
        raise ValueError(
            f'FGC return mode must be one of {_RUNTIME_FGC_RETURN_MODES}, '
            f'got {return_mode!r}'
        )
    if return_mode == _LEGACY_FGC_RETURN_MODE and stage != 2:
        raise ValueError('post_upconv return is supported only for legacy FGC@S2')
    return return_mode


def resolve_fgc_return_downsample_mode(
    stage: int | None,
    mode: str | None,
) -> str | None:
    """Resolve the decoder return resampler."""
    if stage is None:
        if mode is not None:
            raise ValueError('FGC return downsampling requires an FGC stage')
        return None
    if mode is None:
        return FGC_RETURN_DOWNSAMPLE_DEFAULT
    if mode not in FGC_RETURN_DOWNSAMPLE_MODES:
        raise ValueError(
            f'FGC return downsample mode must be one of '
            f'{FGC_RETURN_DOWNSAMPLE_MODES}, got {mode!r}'
        )
    return mode


def resolve_fgc_return_prefilter(
    stage: int | None,
    return_mode: str | None,
    enabled: bool,
) -> bool:
    """Validate the optional learnable filter before native-grid return."""
    if not isinstance(enabled, bool):
        raise TypeError('FGC return prefilter must be boolean')
    if not enabled:
        return False
    if stage is None:
        raise ValueError('FGC return prefilter requires an FGC stage')
    if return_mode != FGC_RETURN_LATE:
        raise ValueError('FGC return prefilter requires late return')
    return True


def align_spad_feature(
    source: torch.Tensor,
    target_shape: Sequence[int],
    *,
    mode: str = 'trilinear',
) -> torch.Tensor:
    """Map a feature tensor to a same-FOV consumer cell grid."""
    target_shape = tuple(int(value) for value in target_shape)
    if source.ndim != 5:
        raise ValueError(f'SPAD feature must be 5D, got shape {tuple(source.shape)}')
    if len(target_shape) != 3 or any(value < 1 for value in target_shape):
        raise ValueError(f'target_shape must contain three positive values, got {target_shape}')
    if tuple(source.shape[2:]) == target_shape:
        return source
    if mode == 'trilinear':
        aligned = F.interpolate(
            source,
            size=target_shape,
            mode='trilinear',
            align_corners=False,
        )
    elif mode == 'area':
        if any(
            target > current
            for target, current in zip(
                target_shape,
                source.shape[2:],
                strict=True,
            )
        ):
            raise ValueError(
                f'area alignment requires downsampling, got source '
                f'{tuple(source.shape[2:])} and target {target_shape}'
            )
        aligned = F.interpolate(source, size=target_shape, mode='area')
    else:
        raise ValueError(f'unsupported SPAD feature alignment mode {mode!r}')
    return aligned.to(dtype=source.dtype)


class FGCReturnPrefilter(nn.Module):
    """Learn a per-channel z-only residual before FGC return downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        nn.init.zeros_(self.depthwise.weight)

    def forward(
        self,
        source: torch.Tensor,
        target_shape: Sequence[int],
    ) -> torch.Tensor:
        """Filter a real downsampling path and keep identity cases visible to DDP."""
        target_shape = tuple(int(value) for value in target_shape)
        source_shape = tuple(source.shape[2:])
        if source_shape == target_shape:
            zero = self.depthwise.weight.sum().to(dtype=source.dtype) * 0
            return source + zero
        if (
            source_shape[1:] != target_shape[1:]
            or target_shape[0] > source_shape[0]
        ):
            raise ValueError(
                f'FGC return prefilter requires z-only downsampling, got '
                f'source {source_shape} and target {target_shape}'
            )
        return source + self.depthwise(source)


def build_spad_architecture(
    source_architecture: Mapping[str, Any],
    *,
    inference_da: int,
    min_bottleneck: int = 4,
    num_canonical_regions: int | None = None,
    learnable_kernel_reduction: bool = False,
    full_kernel_dynamic_stride: bool = False,
) -> dict[str, Any]:
    """Derive a native nnU-Net architecture schema for the matched SPAD network."""
    if source_architecture.get('network_class_name') != _RESENC_NETWORK_CLASS_NAME:
        raise ValueError(
            'SPAD U-Net requires a ResidualEncoderUNet source architecture'
        )
    source_kwargs = source_architecture.get('arch_kwargs')
    if not isinstance(source_kwargs, Mapping):
        raise TypeError('source architecture arch_kwargs must be a mapping')
    required = {
        'n_stages',
        'features_per_stage',
        'conv_op',
        'kernel_sizes',
        'strides',
        'n_blocks_per_stage',
        'n_conv_per_stage_decoder',
        'conv_bias',
        'norm_op',
        'norm_op_kwargs',
        'dropout_op',
        'dropout_op_kwargs',
        'nonlin',
        'nonlin_kwargs',
    }
    missing = required - source_kwargs.keys()
    if missing:
        raise KeyError(
            f'source architecture is missing required kwargs: {sorted(missing)}'
        )
    n_stages = int(source_kwargs['n_stages'])
    if n_stages < 2:
        raise ValueError(f'SPAD U-Net requires at least two stages, got {n_stages}')
    for key in ('features_per_stage', 'n_blocks_per_stage'):
        if len(source_kwargs[key]) != n_stages:
            raise ValueError(f'{key} must contain {n_stages} entries')
    kernel_sizes = [
        tuple(int(value) for value in kernel)
        for kernel in source_kwargs['kernel_sizes']
    ]
    if len(kernel_sizes) != n_stages or any(
        len(kernel) != 3 or any(value not in (1, 3) for value in kernel)
        for kernel in kernel_sizes
    ):
        raise ValueError(f'invalid 3D ResEnc kernel schedule: {kernel_sizes}')
    strides = [
        tuple(int(value) for value in stride) for stride in source_kwargs['strides']
    ]
    if (
        len(strides) != n_stages
        or strides[0] != (1, 1, 1)
        or any(
            len(stride) != 3 or any(value not in (1, 2) for value in stride)
            for stride in strides[1:]
        )
    ):
        raise ValueError(f'invalid 3D ResEnc stride schedule: {strides}')
    expected_values = {
        'conv_op': 'torch.nn.modules.conv.Conv3d',
        'conv_bias': True,
        'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': 'torch.nn.LeakyReLU',
        'nonlin_kwargs': {'inplace': True},
    }
    mismatched = {
        key: {'expected': expected, 'actual': source_kwargs[key]}
        for key, expected in expected_values.items()
        if source_kwargs[key] != expected
    }
    if mismatched:
        raise ValueError(f'SPAD U-Net source architecture mismatch: {mismatched}')
    if list(source_kwargs['n_conv_per_stage_decoder']) != [1] * (n_stages - 1):
        raise ValueError('SPAD U-Net requires one convolution per decoder stage')
    if (
        isinstance(inference_da, bool)
        or not isinstance(inference_da, int)
        or inference_da < 0
    ):
        raise ValueError(
            f'inference_da must be a non-negative integer, got {inference_da!r}'
        )
    if min_bottleneck < 1:
        raise ValueError(f'min_bottleneck must be positive, got {min_bottleneck}')
    if num_canonical_regions is not None and num_canonical_regions < 1:
        raise ValueError('num_canonical_regions must be positive')
    if not isinstance(learnable_kernel_reduction, bool):
        raise TypeError('learnable_kernel_reduction must be boolean')
    if not isinstance(full_kernel_dynamic_stride, bool):
        raise TypeError('full_kernel_dynamic_stride must be boolean')
    if learnable_kernel_reduction and full_kernel_dynamic_stride:
        raise ValueError('FKDS is incompatible with learnable kernel reduction')
    universal_kwargs = (
        {}
        if num_canonical_regions is None
        else {'num_canonical_regions': num_canonical_regions}
    )
    return {
        'network_class_name': (
            SPAD_NETWORK_CLASS_NAME
            if num_canonical_regions is None
            else UNIVERSAL_SPAD_NETWORK_CLASS_NAME
        ),
        'arch_kwargs': {
            key: source_kwargs[key]
            for key in (
                'n_stages',
                'features_per_stage',
                'n_blocks_per_stage',
                'n_conv_per_stage_decoder',
            )
        }
        | {
            'inference_da': inference_da,
            'min_bottleneck': min_bottleneck,
            'learnable_kernel_reduction': learnable_kernel_reduction,
            'full_kernel_dynamic_stride': full_kernel_dynamic_stride,
        }
        | universal_kwargs,
        '_kw_requires_import': [],
    }


UNIVERSAL_SPAD_N_STAGES = 6
UNIVERSAL_SPAD_FEATURES = (32, 64, 128, 256, 320, 320)
UNIVERSAL_SPAD_BLOCKS = (1, 3, 4, 6, 6, 6)
UNIVERSAL_SPAD_DECODER_CONVS = (1, 1, 1, 1, 1)
UNIVERSAL_SPAD_MIN_BOTTLENECK = 4
UNIVERSAL_SPAD_TAB_FEATURE_LEVEL_INDICES = (4, 5)
LEGACY_UNIVERSAL_SPAD_N_STAGES = 7
LEGACY_UNIVERSAL_SPAD_FEATURES = (*UNIVERSAL_SPAD_FEATURES, 320)
LEGACY_UNIVERSAL_SPAD_BLOCKS = (*UNIVERSAL_SPAD_BLOCKS, 6)
LEGACY_UNIVERSAL_SPAD_DECODER_CONVS = (*UNIVERSAL_SPAD_DECODER_CONVS, 1)


def universal_spad_architecture_profile(
    n_stages: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the approved six-stage profile or its legacy seven-stage form."""
    if n_stages == UNIVERSAL_SPAD_N_STAGES:
        return (
            UNIVERSAL_SPAD_FEATURES,
            UNIVERSAL_SPAD_BLOCKS,
            UNIVERSAL_SPAD_DECODER_CONVS,
        )
    if n_stages == LEGACY_UNIVERSAL_SPAD_N_STAGES:
        return (
            LEGACY_UNIVERSAL_SPAD_FEATURES,
            LEGACY_UNIVERSAL_SPAD_BLOCKS,
            LEGACY_UNIVERSAL_SPAD_DECODER_CONVS,
        )
    raise ValueError(
        f'SPAD Universal supports six stages and legacy seven-stage plans, got {n_stages}'
    )


def build_universal_spad_architecture(
    source_architecture: Mapping[str, Any],
    *,
    inference_da: int,
    num_canonical_regions: int,
    num_task_datasets: int,
    num_packed_output_channels: int | None = None,
    learnable_kernel_reduction: bool = False,
    full_kernel_dynamic_stride: bool = False,
    feature_grid_canonicalization_stage: int | None = None,
    feature_grid_canonicalization_return: str | None = None,
    feature_grid_canonicalization_return_downsample_mode: str | None = None,
    feature_grid_canonicalization_return_prefilter: bool = False,
) -> dict[str, Any]:
    """Build a SPAD Universal schema using the source plan's approved topology."""
    if (
        full_kernel_dynamic_stride
        and feature_grid_canonicalization_stage is not None
    ):
        raise ValueError('FKDS does not support FGC')
    architecture = build_spad_architecture(
        source_architecture,
        inference_da=inference_da,
        min_bottleneck=UNIVERSAL_SPAD_MIN_BOTTLENECK,
        num_canonical_regions=num_canonical_regions,
        learnable_kernel_reduction=learnable_kernel_reduction,
        full_kernel_dynamic_stride=full_kernel_dynamic_stride,
    )
    source_kwargs = architecture['arch_kwargs']
    n_stages = int(source_kwargs['n_stages'])
    features, blocks, decoder_convs = universal_spad_architecture_profile(n_stages)
    source_profile = (
        tuple(source_kwargs['features_per_stage']),
        tuple(source_kwargs['n_blocks_per_stage']),
        tuple(source_kwargs['n_conv_per_stage_decoder']),
    )
    expected_profile = (features, blocks, decoder_convs)
    if source_profile != expected_profile:
        raise ValueError(
            f'SPAD Universal source architecture does not match its approved '
            f'{n_stages}-stage profile: expected {expected_profile}, got {source_profile}'
        )
    tab_feature_level_indices = tuple(range(4, n_stages))
    if (
        feature_grid_canonicalization_stage is not None
        and n_stages != UNIVERSAL_SPAD_N_STAGES
    ):
        raise ValueError(
            'the FGC contract requires the six-stage Universal network'
        )
    resolved_fgc_return = resolve_fgc_return_mode(
        feature_grid_canonicalization_stage,
        feature_grid_canonicalization_return,
    )
    resolved_fgc_return_downsample = resolve_fgc_return_downsample_mode(
        feature_grid_canonicalization_stage,
        feature_grid_canonicalization_return_downsample_mode,
    )
    resolved_fgc_return_prefilter = resolve_fgc_return_prefilter(
        feature_grid_canonicalization_stage,
        resolved_fgc_return,
        feature_grid_canonicalization_return_prefilter,
    )
    architecture['arch_kwargs'].update(
        {
            'n_stages': n_stages,
            'features_per_stage': features,
            'n_blocks_per_stage': blocks,
            'n_conv_per_stage_decoder': decoder_convs,
            'num_task_datasets': num_task_datasets,
            'tab_tokens_per_dataset': TAB_TOKENS_PER_DATASET,
            'tab_dim': TAB_DIM,
            'tab_depth': TAB_DEPTH,
            'tab_heads': TAB_HEADS,
            'tab_mlp_dim': TAB_MLP_DIM,
            'tab_attention_downsample_rate': TAB_ATTENTION_DOWNSAMPLE_RATE,
            'tab_fourier_scale': TAB_FOURIER_SCALE,
            'tab_fourier_seed': TAB_FOURIER_SEED,
            'tab_feature_level_indices': tab_feature_level_indices,
        }
    )
    if num_packed_output_channels is not None:
        if num_packed_output_channels < num_canonical_regions:
            raise ValueError(
                'packed output channels cannot be smaller than the canonical '
                'foreground bank'
            )
        architecture['arch_kwargs']['num_packed_output_channels'] = (
            num_packed_output_channels
        )
    if feature_grid_canonicalization_stage is not None:
        architecture['arch_kwargs'].update({
            'feature_grid_canonicalization_stage': (
                feature_grid_canonicalization_stage
            ),
            'feature_grid_canonicalization_return': resolved_fgc_return,
            'feature_grid_canonicalization_return_downsample_mode': (
                resolved_fgc_return_downsample
            ),
            'feature_grid_canonicalization_return_prefilter': (
                resolved_fgc_return_prefilter
            ),
        })
    return architecture


class SPADConvBlock(nn.Module):
    """Conv + InstanceNorm + LeakyReLU with DA-aware stride."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_stride: bool = False,
        bias: bool = True,
        learnable_kernel_reduction: bool = False,
    ):
        super().__init__()
        self.has_stride = has_stride
        if not has_stride:
            self.conv = SPADConv3d_K3S1(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
                learnable_kernel_reduction=learnable_kernel_reduction,
            )
        else:
            self.conv = SPADConv3d_K3S2(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                bias=bias,
                learnable_kernel_reduction=learnable_kernel_reduction,
            )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin = nn.LeakyReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, da: int, stride: tuple[int, int, int] = (1, 1, 1)
    ) -> torch.Tensor:
        if not self.has_stride:
            if stride != (1, 1, 1):
                raise ValueError(f'non-strided block received runtime stride {stride}')
            x = self.conv(x, da)
        elif stride == (1, 1, 1):
            x = self.conv(x, da, stride_override=stride)
        elif stride in ((1, 2, 2), (2, 2, 2)):
            effective_da = 0 if stride == (2, 2, 2) else max(da, 1)
            if effective_da == 0:
                x = F.pad(x, (1, 1, 1, 1, 1, 1))
            else:
                x = F.pad(x, (1, 1, 1, 1, 0, 0))
            x = self.conv(x, effective_da, stride_override=stride)
        else:
            raise ValueError(f'unsupported SPAD U-Net runtime stride {stride}')
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class SPADBasicBlock(nn.Module):
    """ResNet-D style residual block with SPAD convolutions.

    Two convolutions + residual. First conv may have stride for downsampling.
    Skip connection uses stride-aware average pooling + Conv1x1 + InstanceNorm when needed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_stride: bool = False,
        bias: bool = True,
        learnable_kernel_reduction: bool = False,
    ):
        super().__init__()
        self.has_stride = has_stride
        self.conv1 = SPADConvBlock(
            in_channels,
            out_channels,
            has_stride=has_stride,
            bias=bias,
            learnable_kernel_reduction=learnable_kernel_reduction,
        )
        self.conv2_conv = SPADConv3d_K3S1(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            learnable_kernel_reduction=learnable_kernel_reduction,
        )
        self.conv2_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin = nn.LeakyReLU(inplace=True)

        # Skip connection (ResNet-D: AvgPool when stride, Conv1x1+Norm when channels change)
        self.has_skip_pool = has_stride
        self.has_skip_proj = in_channels != out_channels
        if self.has_skip_proj:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        else:
            self.skip = None

    def forward(
        self, x: torch.Tensor, da: int, stride: tuple[int, int, int] = (1, 1, 1)
    ) -> torch.Tensor:
        if self.has_skip_pool or self.skip is not None:
            residual = x
            if self.has_skip_pool:
                if stride != (1, 1, 1):
                    pool_kernel = list(stride)
                    residual = F.avg_pool3d(residual, pool_kernel, pool_kernel)
            if self.skip is not None:
                residual = self.skip(residual)
        else:
            residual = x

        out = self.conv1(x, da, stride)
        out = self.conv2_conv(out, da)
        out = self.conv2_norm(out)
        return self.nonlin(out + residual)


class SPADResidualEncoder(nn.Module):
    """Stem + N stages of stacked residual blocks with DA-aware SPAD convolutions.

    Returns a list of skip features (one per stage).
    """

    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: tuple[int, ...],
        n_blocks_per_stage: tuple[int, ...],
        learnable_kernel_reduction: bool = False,
        feature_grid_canonicalization_stage: int | None = None,
        feature_grid_canonicalization_return: str | None = None,
    ):
        super().__init__()
        assert len(features_per_stage) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        self.feature_grid_canonicalization_stage = (
            feature_grid_canonicalization_stage
        )
        self.feature_grid_canonicalization_return = resolve_fgc_return_mode(
            feature_grid_canonicalization_stage,
            feature_grid_canonicalization_return,
        )

        # Stem: single conv block at stride=1
        self.stem = SPADConvBlock(
            in_channels,
            features_per_stage[0],
            has_stride=False,
            learnable_kernel_reduction=learnable_kernel_reduction,
        )

        # Stages: each is a sequence of residual blocks
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            blocks = []
            has_stride = s > 0
            in_ch = features_per_stage[s - 1] if s > 0 else features_per_stage[0]
            out_ch = features_per_stage[s]
            blocks.append(
                SPADBasicBlock(
                    in_ch,
                    out_ch,
                    has_stride=has_stride,
                    learnable_kernel_reduction=learnable_kernel_reduction,
                )
            )
            for _ in range(n_blocks_per_stage[s] - 1):
                blocks.append(
                    SPADBasicBlock(
                        out_ch,
                        out_ch,
                        has_stride=False,
                        learnable_kernel_reduction=learnable_kernel_reduction,
                    )
                )
            self.stages.append(nn.ModuleList(blocks))

    def forward(
        self,
        x: torch.Tensor,
        da_schedule: list[int],
        stride_schedule: list[tuple[int, int, int]],
        canonical_shape: tuple[int, int, int] | None = None,
    ) -> list[torch.Tensor]:
        if (canonical_shape is None) != (
            self.feature_grid_canonicalization_stage is None
        ):
            raise ValueError(
                'canonical_shape must be provided exactly when FGC is enabled'
            )
        x = self.stem(x, da_schedule[0])
        skips = []
        for s, stage in enumerate(self.stages):
            if s == self.feature_grid_canonicalization_stage:
                assert canonical_shape is not None
                if tuple(x.shape[3:]) != canonical_shape[1:]:
                    raise ValueError(
                        f'FGC@S{s} must preserve H/W: feature {tuple(x.shape[2:])}, '
                        f'target {canonical_shape}'
                    )
                x = align_spad_feature(x, canonical_shape)
                if self.feature_grid_canonicalization_return == FGC_RETURN_LATE:
                    skips[-1] = x
            da = da_schedule[s]
            stride = stride_schedule[s]
            for i, block in enumerate(stage):
                if i == 0:
                    x = block(x, da, stride)
                else:
                    x = block(x, da)
            skips.append(x)
        return skips


class SPADUNetDecoder(nn.Module):
    """DA-aware decoder with nnU-Net-compatible segmentation heads."""

    def __init__(
        self,
        n_stages: int,
        features_per_stage: tuple[int, ...],
        num_classes: int,
        deep_supervision: bool,
        learnable_kernel_reduction: bool = False,
        feature_grid_canonicalization_stage: int | None = None,
        feature_grid_canonicalization_return: str | None = None,
        feature_grid_canonicalization_return_downsample_mode: str | None = None,
        feature_grid_canonicalization_return_prefilter: bool = False,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.deep_supervision = deep_supervision
        self.feature_grid_canonicalization_stage = (
            feature_grid_canonicalization_stage
        )
        self.feature_grid_canonicalization_return = resolve_fgc_return_mode(
            feature_grid_canonicalization_stage,
            feature_grid_canonicalization_return,
        )
        self.feature_grid_canonicalization_return_downsample_mode = (
            resolve_fgc_return_downsample_mode(
                feature_grid_canonicalization_stage,
                feature_grid_canonicalization_return_downsample_mode,
            )
        )
        self.feature_grid_canonicalization_return_prefilter = (
            resolve_fgc_return_prefilter(
                feature_grid_canonicalization_stage,
                self.feature_grid_canonicalization_return,
                feature_grid_canonicalization_return_prefilter,
            )
        )
        self.return_prefilter = (
            FGCReturnPrefilter(
                features_per_stage[feature_grid_canonicalization_stage - 1]
            )
            if self.feature_grid_canonicalization_return_prefilter
            else None
        )
        n_decoder_stages = n_stages - 1

        self.upsample_convs = nn.ModuleList()
        self.post_cat_convs = nn.ModuleList()

        for i in range(n_decoder_stages):
            in_ch = features_per_stage[n_stages - 1 - i]
            skip_ch = features_per_stage[n_stages - 2 - i]
            self.upsample_convs.append(
                SPADConvTranspose3d_K2S2(
                    in_ch,
                    skip_ch,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                    learnable_kernel_reduction=learnable_kernel_reduction,
                )
            )
            self.post_cat_convs.append(
                SPADConvBlock(
                    skip_ch * 2,
                    skip_ch,
                    has_stride=False,
                    learnable_kernel_reduction=learnable_kernel_reduction,
                )
            )
        self.seg_layers = nn.ModuleList(
            [
                nn.Conv3d(channels, num_classes, kernel_size=1)
                for channels in features_per_stage[:-1]
            ]
        )

    def forward(
        self,
        skips: list[torch.Tensor],
        da_schedule: list[int],
        stride_schedule: list[tuple[int, int, int]],
        native_bridge_shape: tuple[int, int, int] | None = None,
    ) -> list[torch.Tensor]:
        """Decode skip features into multi-scale feature outputs.

        Returns decoder features ordered from low-res to full-res.
        """
        n = len(skips)
        if (native_bridge_shape is None) != (
            self.feature_grid_canonicalization_stage is None
        ):
            raise ValueError(
                'native_bridge_shape must be provided exactly when FGC is enabled'
            )
        x = skips[-1]
        features = []
        for i in range(n - 1):
            encoder_stage_idx = n - 1 - i
            stride = stride_schedule[encoder_stage_idx]
            effective_da = 0 if stride == (2, 2, 2) else 1
            at_fgc_boundary = (
                encoder_stage_idx == self.feature_grid_canonicalization_stage
            )
            if (
                at_fgc_boundary
                and self.feature_grid_canonicalization_return == FGC_RETURN_EARLY
            ):
                assert native_bridge_shape is not None
                native_stage_shape = tuple(
                    size // step
                    for size, step in zip(
                        native_bridge_shape,
                        stride,
                        strict=True,
                    )
                )
                if tuple(x.shape[3:]) != native_stage_shape[1:]:
                    raise ValueError(
                        f'FGC@S{encoder_stage_idx} early return must preserve '
                        f'H/W: feature {tuple(x.shape[2:])}, '
                        f'target {native_stage_shape}'
                    )
                assert self.feature_grid_canonicalization_return_downsample_mode
                x = align_spad_feature(
                    x,
                    native_stage_shape,
                    mode=self.feature_grid_canonicalization_return_downsample_mode,
                )
            x = self.upsample_convs[i](
                x,
                effective_da,
                stride_override=stride,
            )

            skip = skips[n - 2 - i]
            if (
                at_fgc_boundary
                and self.feature_grid_canonicalization_return
                == _LEGACY_FGC_RETURN_MODE
            ):
                if tuple(x.shape[3:]) != tuple(skip.shape[3:]):
                    raise ValueError(
                        f'FGC@S{encoder_stage_idx} post-upconv return must '
                        f'preserve H/W: feature '
                        f'{tuple(x.shape[2:])}, skip {tuple(skip.shape[2:])}'
                    )
                assert self.feature_grid_canonicalization_return_downsample_mode
                x = align_spad_feature(
                    x,
                    skip.shape[2:],
                    mode=self.feature_grid_canonicalization_return_downsample_mode,
                )
            if tuple(x.shape[2:]) != tuple(skip.shape[2:]):
                raise ValueError(
                    f'decoder feature {tuple(x.shape[2:])} does not match '
                    f'stage-{encoder_stage_idx - 1} skip {tuple(skip.shape[2:])}'
                )
            x = torch.cat([x, skip], dim=1)
            post_cat_da = da_schedule[n - 2 - i]
            if (
                at_fgc_boundary
                and self.feature_grid_canonicalization_return == FGC_RETURN_LATE
            ):
                # The cached bridge skip is the input grid of the canonical
                # stage, so its decoder fusion uses that stage's DA state.
                post_cat_da = da_schedule[encoder_stage_idx]
            x = self.post_cat_convs[i](x, post_cat_da)
            if (
                at_fgc_boundary
                and self.feature_grid_canonicalization_return == FGC_RETURN_LATE
            ):
                assert native_bridge_shape is not None
                if tuple(x.shape[3:]) != native_bridge_shape[1:]:
                    raise ValueError(
                        f'FGC@S{encoder_stage_idx} late return must preserve '
                        f'H/W: feature {tuple(x.shape[2:])}, '
                        f'target {native_bridge_shape}'
                    )
                assert self.feature_grid_canonicalization_return_downsample_mode
                if self.return_prefilter is not None:
                    x = self.return_prefilter(x, native_bridge_shape)
                x = align_spad_feature(
                    x,
                    native_bridge_shape,
                    mode=self.feature_grid_canonicalization_return_downsample_mode,
                )
            features.append(x)
        return features

    def segment(
        self,
        features: list[torch.Tensor],
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Project full-resolution-first decoder features into segmentation logits."""
        if len(features) != len(self.seg_layers):
            raise ValueError(
                f'expected {len(self.seg_layers)} decoder levels, got {len(features)}'
            )
        if output_rows is None:
            outputs = [
                head(feature)
                for head, feature in zip(self.seg_layers, features, strict=True)
            ]
        else:
            if output_rows.ndim != 1 or output_rows.dtype != torch.long:
                raise ValueError('output_rows must be a 1D LongTensor')
            outputs = [
                F.conv3d(
                    feature,
                    head.weight.index_select(0, output_rows),
                    None
                    if head.bias is None
                    else head.bias.index_select(0, output_rows),
                )
                for head, feature in zip(
                    self.seg_layers,
                    features,
                    strict=True,
                )
            ]
        return outputs if self.deep_supervision else outputs[0]


class TaskAwareSPADDecoder(nn.Module):
    """Apply TAB to an aligned bottleneck before the spatial SPAD decoder."""

    def __init__(
        self,
        spatial_decoder: SPADUNetDecoder,
        bottleneck_channels: int,
        num_task_datasets: int,
        tab_tokens_per_dataset: int,
        tab_dim: int,
        tab_depth: int,
        tab_heads: int,
        tab_mlp_dim: int,
        tab_attention_downsample_rate: int,
        tab_fourier_scale: float,
        tab_fourier_seed: int,
        tab_feature_level_indices: Sequence[int] | None = None,
        feature_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.spatial_decoder = spatial_decoder
        self.tab_feature_level_indices = (
            None
            if tab_feature_level_indices is None
            else tuple(int(index) for index in tab_feature_level_indices)
        )
        common_kwargs = {
            'num_datasets': num_task_datasets,
            'tokens_per_dataset': tab_tokens_per_dataset,
            'embedding_dim': tab_dim,
            'depth': tab_depth,
            'num_heads': tab_heads,
            'mlp_dim': tab_mlp_dim,
            'attention_downsample_rate': tab_attention_downsample_rate,
            'fourier_scale': tab_fourier_scale,
            'fourier_seed': tab_fourier_seed,
        }
        if self.tab_feature_level_indices is None:
            self.task_aware_bottleneck = TaskAwareBottleneck(
                bottleneck_channels=bottleneck_channels,
                **common_kwargs,
            )
        else:
            if feature_channels is None:
                raise ValueError('multi-scale TAB requires feature_channels')
            if (
                not self.tab_feature_level_indices
                or tuple(sorted(set(self.tab_feature_level_indices)))
                != self.tab_feature_level_indices
                or self.tab_feature_level_indices[-1] >= len(feature_channels)
            ):
                raise ValueError(
                    'tab_feature_level_indices must be sorted, unique, and valid'
                )
            selected_channels = tuple(
                feature_channels[index]
                for index in self.tab_feature_level_indices
            )
            self.task_aware_bottleneck = MultiScaleTaskAwareBottleneck(
                feature_channels=selected_channels,
                **common_kwargs,
            )

    @property
    def post_cat_convs(self) -> nn.ModuleList:
        return self.spatial_decoder.post_cat_convs

    @property
    def seg_layers(self) -> nn.ModuleList:
        return self.spatial_decoder.seg_layers

    @property
    def deep_supervision(self) -> bool:
        return self.spatial_decoder.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, value: bool) -> None:
        self.spatial_decoder.deep_supervision = value

    def forward(
        self,
        skips: list[torch.Tensor],
        da_schedule: list[int],
        stride_schedule: list[tuple[int, int, int]],
        dataset_indices: torch.Tensor,
        native_bridge_shape: tuple[int, int, int] | None = None,
    ) -> list[torch.Tensor]:
        if self.tab_feature_level_indices is None:
            conditioned_skips = [
                *skips[:-1],
                self.task_aware_bottleneck(skips[-1], dataset_indices),
            ]
        else:
            conditioned_skips = list(skips)
            selected_features = [
                skips[index] for index in self.tab_feature_level_indices
            ]
            conditioned_features = self.task_aware_bottleneck(
                selected_features,
                dataset_indices,
            )
            for index, feature in zip(
                self.tab_feature_level_indices,
                conditioned_features,
                strict=True,
            ):
                conditioned_skips[index] = feature
        return self.spatial_decoder(
            conditioned_skips,
            da_schedule,
            stride_schedule,
            native_bridge_shape,
        )

    def segment(
        self,
        features: list[torch.Tensor],
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        return self.spatial_decoder.segment(features, output_rows)


class SPADResEncUNet(nn.Module):
    """Plan-constructible ResEnc SPAD U-Net with DA threading throughout."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        n_stages: int = 6,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        n_blocks_per_stage: Sequence[int] = (1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder: Sequence[int] = (1, 1, 1, 1, 1),
        inference_da: int,
        min_bottleneck: int = 4,
        num_canonical_regions: int | None = None,
        num_packed_output_channels: int | None = None,
        learnable_kernel_reduction: bool = False,
        full_kernel_dynamic_stride: bool = False,
        feature_grid_canonicalization_stage: int | None = None,
        feature_grid_canonicalization_return: str | None = None,
        feature_grid_canonicalization_return_downsample_mode: str | None = None,
        feature_grid_canonicalization_return_prefilter: bool = False,
        deep_supervision: bool = True,
    ):
        super().__init__()
        features_per_stage = tuple(int(value) for value in features_per_stage)
        n_blocks_per_stage = tuple(int(value) for value in n_blocks_per_stage)
        n_conv_per_stage_decoder = tuple(
            int(value) for value in n_conv_per_stage_decoder
        )
        if len(features_per_stage) != n_stages:
            raise ValueError('features_per_stage must match n_stages')
        if len(n_blocks_per_stage) != n_stages:
            raise ValueError('n_blocks_per_stage must match n_stages')
        if n_conv_per_stage_decoder != (1,) * (n_stages - 1):
            raise ValueError('SPAD decoder requires one convolution per decoder stage')
        if (
            isinstance(inference_da, bool)
            or not isinstance(inference_da, int)
            or inference_da < 0
        ):
            raise ValueError(
                f'inference_da must be a non-negative integer, got {inference_da!r}'
            )
        if num_canonical_regions is not None and num_canonical_regions < 1:
            raise ValueError('num_canonical_regions must be positive')
        if num_packed_output_channels is not None:
            if num_canonical_regions is None:
                raise ValueError(
                    'packed output channels require a canonical region bank'
                )
            if num_packed_output_channels < num_canonical_regions:
                raise ValueError(
                    'packed output channels cannot be smaller than the '
                    'canonical foreground bank'
                )
        if not isinstance(learnable_kernel_reduction, bool):
            raise TypeError('learnable_kernel_reduction must be boolean')
        if not isinstance(full_kernel_dynamic_stride, bool):
            raise TypeError('full_kernel_dynamic_stride must be boolean')
        if learnable_kernel_reduction and full_kernel_dynamic_stride:
            raise ValueError('FKDS is incompatible with learnable kernel reduction')
        if (
            full_kernel_dynamic_stride
            and feature_grid_canonicalization_stage is not None
        ):
            raise ValueError('FKDS does not support FGC')
        if (
            feature_grid_canonicalization_stage is not None
            and n_stages != UNIVERSAL_SPAD_N_STAGES
        ):
            raise ValueError(
                'the FGC contract requires the six-stage Universal network'
            )
        resolved_fgc_return = resolve_fgc_return_mode(
            feature_grid_canonicalization_stage,
            feature_grid_canonicalization_return,
        )
        resolved_fgc_return_downsample = resolve_fgc_return_downsample_mode(
            feature_grid_canonicalization_stage,
            feature_grid_canonicalization_return_downsample_mode,
        )
        resolved_fgc_return_prefilter = resolve_fgc_return_prefilter(
            feature_grid_canonicalization_stage,
            resolved_fgc_return,
            feature_grid_canonicalization_return_prefilter,
        )
        output_channels = (
            num_classes
            if num_canonical_regions is None
            else (
                num_canonical_regions
                if num_packed_output_channels is None
                else num_packed_output_channels
            )
        )
        self.n_stages = n_stages
        self.inference_da = inference_da
        self.min_bottleneck = min_bottleneck
        self.num_canonical_regions = num_canonical_regions
        self.num_packed_output_channels = num_packed_output_channels
        self.learnable_kernel_reduction = learnable_kernel_reduction
        self.full_kernel_dynamic_stride = full_kernel_dynamic_stride
        self.feature_grid_canonicalization_stage = (
            feature_grid_canonicalization_stage
        )
        self.feature_grid_canonicalization_return = resolved_fgc_return
        self.feature_grid_canonicalization_return_downsample_mode = (
            resolved_fgc_return_downsample
        )
        self.feature_grid_canonicalization_return_prefilter = (
            resolved_fgc_return_prefilter
        )
        self.encoder = SPADResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            n_blocks_per_stage,
            learnable_kernel_reduction=learnable_kernel_reduction,
            feature_grid_canonicalization_stage=(
                feature_grid_canonicalization_stage
            ),
            feature_grid_canonicalization_return=resolved_fgc_return,
        )
        self.decoder = SPADUNetDecoder(
            n_stages,
            features_per_stage,
            output_channels,
            deep_supervision,
            learnable_kernel_reduction=learnable_kernel_reduction,
            feature_grid_canonicalization_stage=(
                feature_grid_canonicalization_stage
            ),
            feature_grid_canonicalization_return=resolved_fgc_return,
            feature_grid_canonicalization_return_downsample_mode=(
                resolved_fgc_return_downsample
            ),
            feature_grid_canonicalization_return_prefilter=(
                resolved_fgc_return_prefilter
            ),
        )

    def _operator_da_schedule(self, sample_da: int) -> list[int]:
        """Keep route strides while optionally forcing every K3 into full mode."""
        schedule = compute_da_schedule(sample_da, self.n_stages)
        if self.full_kernel_dynamic_stride:
            return [min(stage_da, 1) for stage_da in schedule]
        return schedule

    def encode_sample(
        self,
        x: torch.Tensor,
        da: int,
        canonical_shape: tuple[int, int, int] | None = None,
    ) -> list[torch.Tensor]:
        """Encode one sample under its spacing-derived DA schedule."""
        patch_size = (x.shape[2], x.shape[3], x.shape[4])
        da_schedule = self._operator_da_schedule(da)
        stride_schedule = compute_stride_schedule(
            da, self.n_stages, patch_size, self.min_bottleneck
        )
        return self.encoder(
            x,
            da_schedule,
            stride_schedule,
            canonical_shape,
        )

    def decode_sample(
        self,
        skips: list[torch.Tensor],
        da: int,
        input_shape: Sequence[int],
    ) -> list[torch.Tensor]:
        """Decode one encoded hierarchy under one consumer DA schedule."""
        patch_size = tuple(int(value) for value in input_shape)
        if len(patch_size) != 3:
            raise ValueError(f'input_shape must contain three values, got {patch_size}')
        da_schedule = self._operator_da_schedule(da)
        stride_schedule = compute_stride_schedule(
            da, self.n_stages, patch_size, self.min_bottleneck
        )
        consumer_shapes = []
        current_shape = patch_size
        for stride in stride_schedule:
            if any(size % step for size, step in zip(current_shape, stride, strict=True)):
                raise ValueError(
                    f'input shape {patch_size} is not exactly divisible by '
                    f'decoder stride schedule {stride_schedule}'
                )
            current_shape = tuple(
                size // step
                for size, step in zip(current_shape, stride, strict=True)
            )
            consumer_shapes.append(current_shape)
        if len(skips) != len(consumer_shapes):
            raise ValueError(
                f'expected {len(consumer_shapes)} encoder features, got {len(skips)}'
            )
        aligned_skips = [
            align_spad_feature(skip, consumer_shape)
            for skip, consumer_shape in zip(skips, consumer_shapes, strict=True)
        ]
        return self.decoder(aligned_skips, da_schedule, stride_schedule)[::-1]

    def forward_features(
        self,
        x: torch.Tensor,
        da: int,
        da_decoder: int | None = None,
    ) -> list[torch.Tensor]:
        """Return full-resolution-first features for tied or Cross-DA execution."""
        if self.feature_grid_canonicalization_stage is not None:
            raise RuntimeError(
                'FGC is supported only by UniversalSPADResEncUNet'
            )
        if da_decoder is None:
            da_decoder = da
        if abs(da - da_decoder) > 1:
            raise ValueError('encoder and decoder DA must be equal or adjacent')
        skips = self.encode_sample(x, da)
        return self.decode_sample(skips, da_decoder, x.shape[2:])

    def forward_sample(
        self,
        x: torch.Tensor,
        da: int,
        da_decoder: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Segment one sample while remaining inside the nnU-Net network."""
        return self.decoder.segment(self.forward_features(x, da, da_decoder))

    def forward(
        self, x: torch.Tensor, da: int | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        if da is None:
            da = self.inference_da
        return self.forward_sample(x, da)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        """Match nnU-Net's He initialization and zero residual branch endings."""
        InitWeights_He(1e-2)(module)
        if isinstance(module, SPADBasicBlock):
            nn.init.constant_(module.conv2_norm.weight, 0)
            nn.init.constant_(module.conv2_norm.bias, 0)
        elif isinstance(module, FGCReturnPrefilter):
            nn.init.zeros_(module.depthwise.weight)


class UniversalSPADResEncUNet(SPADResEncUNet):
    """Execute a dataset-planned-grid sample sequence in one DDP-visible forward."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        num_task_datasets: int,
        tab_tokens_per_dataset: int,
        tab_dim: int,
        tab_depth: int,
        tab_heads: int,
        tab_mlp_dim: int,
        tab_attention_downsample_rate: int,
        tab_fourier_scale: float,
        tab_fourier_seed: int,
        tab_feature_level_indices: Sequence[int] | None = None,
        **architecture_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels,
            num_classes,
            **architecture_kwargs,
        )
        bottleneck_channels = int(architecture_kwargs['features_per_stage'][-1])
        self.decoder = TaskAwareSPADDecoder(
            self.decoder,
            bottleneck_channels,
            num_task_datasets,
            tab_tokens_per_dataset,
            tab_dim,
            tab_depth,
            tab_heads,
            tab_mlp_dim,
            tab_attention_downsample_rate,
            tab_fourier_scale,
            tab_fourier_seed,
            tab_feature_level_indices,
            architecture_kwargs['features_per_stage'],
        )

    def forward_sample(
        self,
        x: torch.Tensor,
        da: int,
        da_decoder: int,
        dataset_indices: torch.Tensor,
        canonical_shape: tuple[int, int, int] | None = None,
        packed_output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if abs(da - da_decoder) > 1:
            raise ValueError('encoder and decoder DA must be equal or adjacent')
        if (packed_output_rows is None) != (
            self.num_packed_output_channels is None
        ):
            raise ValueError(
                'packed_output_rows must be provided exactly when packed '
                'output channels are configured'
            )
        fgc_enabled = self.feature_grid_canonicalization_stage is not None
        if fgc_enabled:
            if da != da_decoder:
                raise ValueError('FGC requires tied encoder and decoder DA')
            if canonical_shape is None:
                raise ValueError('FGC requires a canonical shape')
        elif canonical_shape is not None:
            raise ValueError('canonical shape requires FGC')

        skips = self.encode_sample(x, da, canonical_shape)
        patch_size = tuple(int(value) for value in x.shape[2:])
        da_schedule = self._operator_da_schedule(da_decoder)
        stride_schedule = compute_stride_schedule(
            da_decoder,
            self.n_stages,
            patch_size,
            self.min_bottleneck,
        )
        if fgc_enabled:
            decoder_skips = skips
            assert self.feature_grid_canonicalization_stage is not None
            native_bridge_shape = compute_stage_shapes(
                patch_size,
                stride_schedule[:self.feature_grid_canonicalization_stage],
                label='native prefix',
            )[-1]
        else:
            native_bridge_shape = None
            consumer_shapes = []
            current_shape = patch_size
            for stride in stride_schedule:
                if any(
                    size % step
                    for size, step in zip(current_shape, stride, strict=True)
                ):
                    raise ValueError(
                        f'input shape {patch_size} is not exactly divisible by '
                        f'decoder stride schedule {stride_schedule}'
                    )
                current_shape = tuple(
                    size // step
                    for size, step in zip(current_shape, stride, strict=True)
                )
                consumer_shapes.append(current_shape)
            decoder_skips = [
                align_spad_feature(skip, consumer_shape)
                for skip, consumer_shape in zip(
                    skips,
                    consumer_shapes,
                    strict=True,
                )
            ]
        features = self.decoder(
            decoder_skips,
            da_schedule,
            stride_schedule,
            dataset_indices,
            native_bridge_shape,
        )[::-1]
        return self.decoder.segment(features, packed_output_rows)

    def forward(
        self,
        samples: tuple[tuple, ...],
    ) -> tuple[torch.Tensor | list[torch.Tensor], ...]:
        if not samples:
            raise ValueError('Universal SPAD forward requires at least one sample')
        outputs = []
        for sample in samples:
            packed_output_rows = None
            if len(sample) == 4:
                data, da_encoder, da_decoder, dataset_indices = sample
                canonical_shape = None
            elif len(sample) == 5:
                (
                    data,
                    da_encoder,
                    da_decoder,
                    dataset_indices,
                    canonical_shape,
                ) = sample
            elif len(sample) == 6:
                (
                    data,
                    da_encoder,
                    da_decoder,
                    dataset_indices,
                    canonical_shape,
                    packed_output_rows,
                ) = sample
            else:
                raise ValueError(
                    f'Universal SPAD sample input requires 4, 5, or 6 items, got '
                    f'{len(sample)}'
                )
            outputs.append(self.forward_sample(
                data,
                da_encoder,
                da_decoder,
                dataset_indices,
                canonical_shape,
                packed_output_rows,
            ))
        return tuple(outputs)
