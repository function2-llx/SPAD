from types import SimpleNamespace

import pytest
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pumit.downstream.seg.adapters.native_pyramid import (
    NativePyramidBackbone,
    build_native_pyramid_encoder,
    native_encoder,
)
from pumit.downstream.seg.adapters.pyramid import (
    PlanAlignedPyramidEncoder3D,
    ResampleFPNBranch3D,
    SimpleFPNBranch3D,
    SimpleFPNEncoder3D,
    nominal_stage,
)
from pumit.downstream.seg.network import (
    DenseSegmentationNetwork,
    PlanAlignedSegmentationNetwork,
    PlanAlignedUNetWithoutRefinerSegmentationNetwork,
)
from pumit.downstream.seg.plan import (
    compute_cumulative_strides,
    compute_stage_shapes,
    encoder_plan_from_architecture_kwargs,
)
from tests.downstream.seg.conftest import architecture_kwargs


def _architecture_kwargs(*, strides: list[list[int]] | None = None, n_stages: int = 6) -> dict:
    return architecture_kwargs(
        [4, 8, 12, 16, 24, 32][:n_stages],
        strides,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6][:n_stages],
    )


_ANISOTROPIC_STRIDES = [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


class _FlatBackbone(nn.Module):
    feature_channels = (16,) * 4
    feature_strides = ((16, 16, 16),) * 4

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv3d(1, self.feature_channels[0], kernel_size=16, stride=16)

    def forward(self, x):
        feature = self.patch_embed(x)
        return tuple(feature + index for index in range(4))


class _FinalFeatureBackbone(nn.Module):
    feature_channels = 16
    feature_stride = (16, 16, 16)

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv3d(1, self.feature_channels, kernel_size=16, stride=16)

    def forward(self, x):
        return self.patch_embed(x)


class _CoarseFlatEncoder(nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.full_encoder = PlanAlignedPyramidEncoder3D(
            _FlatBackbone(),
            plan,
            input_channels=1,
        )
        self.output_channels = self.full_encoder.output_channels[2:]
        self.kernel_sizes = self.full_encoder.kernel_sizes[2:]
        self.strides = self.full_encoder.strides[2:]
        for attribute in (
            'conv_op',
            'conv_bias',
            'norm_op',
            'norm_op_kwargs',
            'dropout_op',
            'dropout_op_kwargs',
            'nonlin',
            'nonlin_kwargs',
        ):
            setattr(self, attribute, getattr(self.full_encoder, attribute))

    def forward(self, x):
        return self.full_encoder(x)[2:]


def _make_network(*, deep_supervision: bool):
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    encoder = PlanAlignedPyramidEncoder3D(
        _FlatBackbone(),
        plan,
        input_channels=1,
    )
    return DenseSegmentationNetwork(
        encoder,
        num_classes=3,
        n_conv_per_stage_decoder=plan.n_conv_per_stage_decoder,
        deep_supervision=deep_supervision,
    )


def test_compute_cumulative_strides():
    assert compute_cumulative_strides(
        (
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        )
    ) == (
        (1, 1, 1),
        (1, 2, 2),
        (2, 4, 4),
        (4, 8, 8),
    )


def test_compute_stage_shapes_uses_cumulative_strides():
    assert compute_stage_shapes(
        (32, 64, 64),
        ((1, 4, 4), (2, 8, 8), (4, 16, 16)),
    ) == ((32, 16, 16), (16, 8, 8), (8, 4, 4))


def test_compute_stage_shapes_rejects_inexact_pyramid():
    with pytest.raises(ValueError, match='not divisible'):
        compute_stage_shapes((31, 64, 64), ((2, 4, 4),))


def test_stock_nnunet_builder_constructs_native_architecture(monkeypatch):
    import pumit.downstream.seg.registry as registry_module

    captured = {}

    def build_encoder(name, plan, input_channels, config):
        captured['encoder'] = (name, input_channels, config)
        return PlanAlignedPyramidEncoder3D(
            _FlatBackbone(),
            plan,
            input_channels=input_channels,
        )

    monkeypatch.setattr(registry_module, 'build_encoder', build_encoder)
    architecture_kwargs = {
        **_architecture_kwargs(),
        'conv_op': 'torch.nn.Conv3d',
        'norm_op': 'torch.nn.InstanceNorm3d',
        'nonlin': 'torch.nn.LeakyReLU',
        'backbone_name': 'test',
        'backbone_config': {'variant': 'tiny'},
    }
    configuration_manager = SimpleNamespace(
        network_arch_class_name=(
            'pumit.downstream.seg.network.PlanAlignedSegmentationNetwork'
        ),
        network_arch_init_kwargs=architecture_kwargs,
        network_arch_init_kwargs_req_import=[
            'conv_op',
            'norm_op',
            'dropout_op',
            'nonlin',
        ],
    )

    network = nnUNetTrainer.build_network_architecture(
        plans_manager=None,
        configuration_manager=configuration_manager,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )

    assert isinstance(network, PlanAlignedSegmentationNetwork)
    assert captured['encoder'] == ('test', 1, {'variant': 'tiny'})
    assert len(network.decoder.stages) == 5


@pytest.mark.parametrize(
    ('target_stride', 'expected_shape'),
    [
        ((1, 4, 4), (16, 8, 8)),
        ((4, 8, 8), (4, 4, 4)),
        ((2, 16, 16), (8, 2, 2)),
        ((4, 32, 32), (4, 1, 1)),
    ],
)
def test_simple_fpn_branch_handles_anisotropic_up_and_downsampling(
    target_stride,
    expected_shape,
):
    branch = SimpleFPNBranch3D(
        in_channels=16,
        out_channels=8,
        source_stride=(2, 16, 16),
        target_stride=target_stride,
        kernel_size=(1, 3, 3),
        conv_bias=True,
    )

    output = branch(torch.randn(1, 16, 8, 2, 2))

    assert tuple(output.shape) == (1, 8, *expected_shape)


def test_encoder_provides_every_nnunet_plan_stage():
    network = _make_network(deep_supervision=True)

    skips = network.encoder(torch.randn(1, 1, 64, 64, 64))

    assert len(network.encoder.high_resolution_stem.stages) == 2
    assert len(network.encoder.simple_fpn) == 4
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 4, 64, 64, 64),
        (1, 8, 32, 32, 32),
        (1, 12, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 24, 4, 4, 4),
        (1, 32, 2, 2, 2),
    ]


def test_encoder_provides_anisotropic_nnunet_plan_stages():
    architecture_kwargs = _architecture_kwargs()
    architecture_kwargs['strides'] = [
        [1, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2],
        [2, 2, 2],
    ]
    encoder = PlanAlignedPyramidEncoder3D(
        _FlatBackbone(),
        encoder_plan_from_architecture_kwargs(**architecture_kwargs),
        input_channels=1,
    )

    skips = encoder(torch.randn(1, 1, 64, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 4, 64, 64, 64),
        (1, 8, 64, 32, 32),
        (1, 12, 32, 16, 16),
        (1, 16, 16, 8, 8),
        (1, 24, 16, 4, 4),
        (1, 32, 8, 2, 2),
    ]


def test_encoder_routes_selected_depths_to_matching_simple_fpn_levels():
    encoder = _make_network(deep_supervision=True).encoder
    observed = {}

    stem_hook = encoder.high_resolution_stem.register_forward_pre_hook(
        lambda module, args: observed.update(stem_input=args[0])
    )
    backbone_hook = encoder.backbone.register_forward_hook(
        lambda module, args, output: observed.update(backbone_output=output)
    )
    fpn_hooks = [
        branch.register_forward_hook(
            lambda module, args, output, level=level: observed.update({
                f'fpn_input_{level}': args[0],
                f'fpn_output_{level}': output,
            })
        )
        for level, branch in enumerate(encoder.simple_fpn)
    ]
    input_tensor = torch.randn(1, 1, 64, 64, 64)

    with torch.no_grad():
        skips = encoder(input_tensor)

    stem_hook.remove()
    backbone_hook.remove()
    for hook in fpn_hooks:
        hook.remove()
    assert observed['stem_input'] is input_tensor
    for level, feature in enumerate(observed['backbone_output']):
        assert observed[f'fpn_input_{level}'] is feature
        assert skips[level + 2] is observed[f'fpn_output_{level}']


def test_final_feature_simple_fpn_routes_one_backbone_grid_to_every_pyramid_level():
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    encoder = SimpleFPNEncoder3D(_FinalFeatureBackbone(), plan, input_channels=1)
    observed = {}
    backbone_hook = encoder.backbone.register_forward_hook(
        lambda module, args, output: observed.update(backbone_output=output)
    )
    fpn_hooks = [
        branch.register_forward_pre_hook(
            lambda module, args, level=level: observed.update({f'fpn_input_{level}': args[0]})
        )
        for level, branch in enumerate(encoder.simple_fpn)
    ]

    skips = encoder(torch.randn(1, 1, 64, 64, 64))

    backbone_hook.remove()
    for hook in fpn_hooks:
        hook.remove()
    assert len(encoder.high_resolution_stem.stages) == 2
    assert len(skips) == 6
    assert all(
        observed[f'fpn_input_{level}'] is observed['backbone_output']
        for level in range(4)
    )


def test_stock_decoder_outputs_native_nnunet_deep_supervision_scales():
    network = _make_network(deep_supervision=True).eval()

    with torch.no_grad():
        outputs = network(torch.randn(1, 1, 64, 64, 64))

    assert isinstance(outputs, list)
    assert [tuple(output.shape) for output in outputs] == [
        (1, 3, 64, 64, 64),
        (1, 3, 32, 32, 32),
        (1, 3, 16, 16, 16),
        (1, 3, 8, 8, 8),
        (1, 3, 4, 4, 4),
    ]
    assert len(network.decoder.stages) == 5


def test_unet_without_refiner_accepts_native_four_level_adapter_encoder(monkeypatch):
    import pumit.downstream.seg.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        'build_encoder',
        lambda name, plan, input_channels, config: _CoarseFlatEncoder(plan),
    )
    network = PlanAlignedUNetWithoutRefinerSegmentationNetwork(
        1,
        3,
        backbone_name='test',
        backbone_config={},
        deep_supervision=True,
        **_architecture_kwargs(),
    ).eval()

    with torch.no_grad():
        outputs = network(torch.randn(1, 1, 64, 64, 64))

    assert len(network.encoder.output_channels) == 4
    assert len(network.decoder.stages) == 3
    assert [tuple(output.shape) for output in outputs] == [
        (1, 3, 64, 64, 64),
        (1, 3, 32, 32, 32),
        (1, 3, 16, 16, 16),
        (1, 3, 8, 8, 8),
        (1, 3, 4, 4, 4),
    ]


def test_loss_backpropagates_through_stem_backbone_simple_fpn_and_decoder():
    network = _make_network(deep_supervision=False).train()
    logits = network(torch.randn(2, 1, 64, 64, 64))

    logits.mean().backward()

    assert network.encoder.backbone.patch_embed.weight.grad is not None
    assert network.encoder.high_resolution_stem.stem.convs[0].conv.weight.grad is not None
    assert network.encoder.simple_fpn[0].proj2.weight.grad is not None
    assert network.decoder.seg_layers[-1].weight.grad is not None


@pytest.mark.parametrize(
    ('encoder_class', 'backbone_class'),
    [(SimpleFPNEncoder3D, _FinalFeatureBackbone), (PlanAlignedPyramidEncoder3D, _FlatBackbone)],
)
def test_encoders_without_the_stem_expose_only_backbone_levels(encoder_class, backbone_class):
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    encoder = encoder_class(backbone_class(), plan, input_channels=1, with_high_resolution_stem=False)

    skips = encoder(torch.randn(1, 1, 64, 64, 64))

    assert encoder.high_resolution_stem is None
    assert tuple(encoder.output_channels) == (12, 16, 24, 32)
    assert len(encoder.strides) == 4
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 12, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 24, 4, 4, 4),
        (1, 32, 2, 2, 2),
    ]


def test_unet_without_refiner_runs_a_stemless_simple_fpn_probe_with_a_single_output(monkeypatch):
    import pumit.downstream.seg.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        'build_encoder',
        lambda name, plan, input_channels, config: SimpleFPNEncoder3D(
            _FinalFeatureBackbone(), plan, input_channels=input_channels, with_high_resolution_stem=False,
        ),
    )
    network = PlanAlignedUNetWithoutRefinerSegmentationNetwork(
        1,
        3,
        backbone_name='test',
        backbone_config={},
        deep_supervision=False,
        **_architecture_kwargs(),
    ).train()

    logits = network(torch.randn(1, 1, 64, 64, 64))
    logits.mean().backward()

    assert tuple(logits.shape) == (1, 3, 64, 64, 64)
    assert network.encoder.backbone.patch_embed.weight.grad is not None
    assert network.decoder.seg_layers[-1].weight.grad is not None
    # Every trainable parameter takes part in the loss (DDP rejects idle parameters); the idle heads stay in place.
    assert len(network.decoder.seg_layers) == 3
    assert all(not p.requires_grad for head in network.decoder.seg_layers[:-1] for p in head.parameters())
    assert all(parameter.grad is not None for parameter in network.parameters() if parameter.requires_grad)


class _StrideBackbone(nn.Module):
    """Flat trunk stub emitting one grid at a configurable stride and width."""

    def __init__(self, channels: int, stride: tuple[int, int, int]):
        super().__init__()
        self.feature_channels = channels
        self.feature_stride = stride
        self.patch_embed = nn.Conv3d(1, channels, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.patch_embed(x)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize(
    ('strides', 'feature_stride', 'input_shape'),
    [
        (None, (16, 16, 16), (64, 64, 64)),
        (_ANISOTROPIC_STRIDES, (8, 16, 16), (32, 64, 64)),
        (_ANISOTROPIC_STRIDES, (16, 16, 16), (32, 64, 64)),
    ],
)
def test_resample_branches_follow_the_plan_stages_not_the_trunk_patch(strides, feature_stride, input_shape):
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs(strides=strides))
    reference = SimpleFPNEncoder3D(
        _StrideBackbone(40, (16, 16, 16)),
        encoder_plan_from_architecture_kwargs(**_architecture_kwargs()),
        input_channels=1,
        with_high_resolution_stem=False,
        pyramid_branch='resample',
    )
    encoder = SimpleFPNEncoder3D(
        _StrideBackbone(40, feature_stride),
        plan,
        input_channels=1,
        with_high_resolution_stem=False,
        pyramid_branch='resample',
    )

    skips = encoder(torch.randn(1, 1, *input_shape))

    assert nominal_stage(feature_stride, plan.cumulative_strides) == 4
    assert [branch.block_stages for branch in encoder.simple_fpn] == [(3, 2), (3,), (4,), (5,)]
    # The same width and schedule give the same branch parameters on every plan and patch shape.
    assert [_parameter_count(branch) for branch in encoder.simple_fpn] == [
        _parameter_count(branch) for branch in reference.simple_fpn
    ]
    assert [tuple(skip.shape[1:]) for skip in skips] == [
        (12, *shape) for shape in compute_stage_shapes(input_shape, plan.cumulative_strides)[2:3]
    ] + [
        (channels, *shape)
        for channels, shape in zip((16, 24, 32), compute_stage_shapes(input_shape, plan.cumulative_strides)[3:])
    ]


def test_resample_branch_resizes_to_each_crossed_stage_grid():
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    branch = ResampleFPNBranch3D(40, 12, 4, 2, plan)
    stage_shapes = compute_stage_shapes((64, 64, 64), plan.cumulative_strides)
    seen = []
    for block in branch.blocks:
        block.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape[2:])))

    output = branch(torch.randn(1, 40, 4, 4, 4), stage_shapes)

    assert seen == [stage_shapes[3], stage_shapes[2]]
    assert tuple(output.shape) == (1, 12, *stage_shapes[2])
    assert isinstance(branch.proj.conv, nn.Conv3d) and branch.proj.conv.kernel_size == (1, 1, 1)
    assert all(isinstance(block.norm, nn.InstanceNorm3d) for block in branch.blocks)


def test_deconv_branch_stays_the_default():
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    encoder = SimpleFPNEncoder3D(_StrideBackbone(40, (16, 16, 16)), plan, input_channels=1)

    assert all(isinstance(branch, SimpleFPNBranch3D) for branch in encoder.simple_fpn)


def test_resample_branch_rejects_a_grid_it_cannot_reach_by_pooling():
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    branch = ResampleFPNBranch3D(40, 32, 4, 5, plan)

    with pytest.raises(ValueError, match='cannot resize'):
        branch(torch.randn(1, 40, 6, 6, 6), [(96,) * 3, (48,) * 3, (24,) * 3, (12,) * 3, (6,) * 3, (4,) * 3])


class _NativeSixLevelEncoder(nn.Module):
    """Stand-in for the CNN pyramid encoders: every plan stage at its own native width."""

    def __init__(self, plan):
        super().__init__()
        self.output_channels = plan.output_channels
        self.cumulative_strides = plan.cumulative_strides
        self.stages = nn.ModuleList(
            nn.Conv3d(1, channels, kernel_size=stride, stride=stride)
            for channels, stride in zip(plan.output_channels, plan.cumulative_strides)
        )
        self.backbone = SimpleNamespace(parameter_layers=lambda: tuple((p,) for p in self.parameters()))

    def forward(self, x):
        return [stage(x) for stage in self.stages]


def test_native_pyramid_encoder_projects_p2_to_p5_onto_the_plan_schedule_without_the_stem():
    plan = encoder_plan_from_architecture_kwargs(**_architecture_kwargs())
    native_channels = (6, 10, 20, 40, 80, 80)
    built = []

    def make_encoder(encoder_plan):
        built.append(_NativeSixLevelEncoder(encoder_plan))
        return built[-1]

    with_stem = build_native_pyramid_encoder(make_encoder, plan, native_channels, input_channels=1, config={})
    assert with_stem is built[0] and tuple(with_stem.output_channels) == (4, 8, 12, 16, 24, 32)

    encoder = build_native_pyramid_encoder(
        make_encoder, plan, native_channels, input_channels=1,
        config={'with_high_resolution_stem': False, 'pyramid_branch': 'resample'},
    )
    inner = built[1]

    skips = encoder(torch.randn(1, 1, 64, 64, 64))

    assert isinstance(encoder, PlanAlignedPyramidEncoder3D)
    assert isinstance(encoder.backbone, NativePyramidBackbone)
    assert native_encoder(encoder) is inner
    assert tuple(inner.output_channels) == native_channels
    assert encoder.backbone.feature_channels == (20, 40, 80, 80)
    assert tuple(encoder.output_channels) == (12, 16, 24, 32)
    assert [branch.block_stages for branch in encoder.simple_fpn] == [(2,), (3,), (4,), (5,)]
    assert [tuple(skip.shape[1:]) for skip in skips] == [
        (12, 16, 16, 16),
        (16, 8, 8, 8),
        (24, 4, 4, 4),
        (32, 2, 2, 2),
    ]


def test_unet_without_refiner_decodes_a_five_stage_plan_from_p2(monkeypatch):
    import pumit.downstream.seg.registry as registry_module

    kwargs = _architecture_kwargs(n_stages=5)
    monkeypatch.setattr(
        registry_module,
        'build_encoder',
        lambda name, plan, input_channels, config: SimpleFPNEncoder3D(
            _StrideBackbone(40, (16, 16, 16)),
            plan,
            input_channels=input_channels,
            with_high_resolution_stem=False,
            pyramid_branch='resample',
        ),
    )
    supervised = PlanAlignedUNetWithoutRefinerSegmentationNetwork(
        1, 3, backbone_name='test', backbone_config={}, deep_supervision=True, **kwargs,
    ).eval()
    single = PlanAlignedUNetWithoutRefinerSegmentationNetwork(
        1, 3, backbone_name='test', backbone_config={}, deep_supervision=False, **kwargs,
    ).eval()

    with torch.no_grad():
        outputs = supervised(torch.randn(1, 1, 64, 64, 64))
        logits = single(torch.randn(1, 1, 64, 64, 64))

    assert len(supervised.encoder.output_channels) == 3
    assert len(supervised.decoder.stages) == 2
    assert [tuple(output.shape) for output in outputs] == [
        (1, 3, 64, 64, 64),
        (1, 3, 32, 32, 32),
        (1, 3, 16, 16, 16),
        (1, 3, 8, 8, 8),
    ]
    assert tuple(logits.shape) == (1, 3, 64, 64, 64)


def test_resample_probe_trains_every_parameter_it_exposes(monkeypatch):
    import pumit.downstream.seg.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        'build_encoder',
        lambda name, plan, input_channels, config: SimpleFPNEncoder3D(
            _StrideBackbone(40, (16, 16, 16)),
            plan,
            input_channels=input_channels,
            with_high_resolution_stem=False,
            pyramid_branch='resample',
        ),
    )
    network = PlanAlignedUNetWithoutRefinerSegmentationNetwork(
        1, 3, backbone_name='test', backbone_config={}, deep_supervision=False, **_architecture_kwargs(),
    ).train()
    for parameter in network.encoder.backbone.parameters():
        parameter.requires_grad_(False)

    network(torch.randn(2, 1, 64, 64, 64)).mean().backward()

    assert all(parameter.grad is not None for parameter in network.parameters() if parameter.requires_grad)
