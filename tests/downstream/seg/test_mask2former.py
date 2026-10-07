from types import SimpleNamespace

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pumit.downstream.seg.mask2former import (
    FixedSemanticMask2FormerDecoder3D,
    Mask2FormerPixelDecoder3D,
    PlanAlignedMask2FormerSegmentationNetwork,
    position_encoding_3d,
)
from pumit.ucpt.seg.pos_embed import sine_pos_embed_3d
from tests.downstream.seg.conftest import architecture_kwargs


def _architecture_kwargs() -> dict:
    return architecture_kwargs(
        [4, 8, 16, 24, 32, 40],
        [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]],
    )


def _skips(batch_size: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, 16, 4, 8, 8),
        torch.randn(batch_size, 24, 2, 4, 4),
        torch.randn(batch_size, 32, 2, 2, 2),
        torch.randn(batch_size, 40, 2, 1, 1),
    ]


class _Encoder(nn.Module):
    output_channels = (16, 24, 32, 40)

    def forward(self, x):
        return _skips(x.shape[0])


def test_position_encoding_varies_along_all_three_axes():
    position = position_encoding_3d(torch.zeros(1, 32, 2, 3, 4), 32)

    assert position.shape == (1, 32, 2, 3, 4)
    assert not torch.equal(position[:, :, 0, 0, 0], position[:, :, 1, 0, 0])
    assert not torch.equal(position[:, :, 0, 0, 0], position[:, :, 0, 1, 0])
    assert not torch.equal(position[:, :, 0, 0, 0], position[:, :, 0, 0, 1])


def test_position_encoding_reuses_project_additive_depth_encoding():
    feature = torch.zeros(2, 32, 2, 3, 4)

    position = position_encoding_3d(feature, 32)
    expected = sine_pos_embed_3d(
        2,
        3,
        4,
        32,
        torch.device('cpu'),
    ).reshape(2, 3, 4, 32).permute(3, 0, 1, 2)

    assert torch.equal(position[0], expected)
    assert torch.equal(position[1], expected)


def test_pixel_decoder_returns_p2_masks_and_low_to_high_attention_levels():
    decoder = Mask2FormerPixelDecoder3D(
        (16, 24, 32, 40),
        hidden_dim=32,
        mask_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        ffn_dim=64,
        num_points=2,
    )

    mask_features, attention_features = decoder(_skips())

    assert mask_features.shape == (2, 16, 4, 8, 8)
    assert [tuple(feature.shape) for feature in attention_features] == [
        (2, 32, 2, 1, 1),
        (2, 32, 2, 2, 2),
        (2, 32, 2, 4, 4),
    ]


def test_decoder_uses_one_ordered_query_per_segmentation_class():
    decoder = FixedSemanticMask2FormerDecoder3D(
        _Encoder(),
        num_classes=3,
        hidden_dim=32,
        mask_dim=16,
        num_heads=4,
        ffn_dim=64,
        deep_supervision=True,
    )

    memory_lengths = []
    handles = [
        layer.cross_attention.register_forward_pre_hook(
            lambda _module, inputs: memory_lengths.append(inputs[1].shape[1])
        )
        for layer in decoder.layers
    ]
    try:
        outputs = decoder(_skips(), output_shape=(8, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    assert decoder.query_features.num_embeddings == 3
    assert not hasattr(decoder, 'class_embedding')
    assert memory_lengths == [42] * 9
    assert [tuple(output.shape) for output in outputs] == [
        (2, 3, 8, 32, 32),
    ] * 10


def test_class_attention_mask_is_invariant_to_softmax_common_shift():
    decoder = FixedSemanticMask2FormerDecoder3D(
        _Encoder(),
        num_classes=3,
        hidden_dim=32,
        mask_dim=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        mask_attention_mode='classes',
        deep_supervision=False,
    )
    logits = torch.randn(2, 3, 2, 3, 4)

    attention_mask = decoder._attention_mask_at_shape(logits, (2, 3, 4))
    shifted_attention_mask = decoder._attention_mask_at_shape(
        logits + 8,
        (2, 3, 4),
    )

    assert torch.equal(attention_mask, shifted_attention_mask)


def test_region_attention_mask_uses_independent_sigmoid_predictions():
    decoder = FixedSemanticMask2FormerDecoder3D(
        _Encoder(),
        num_classes=3,
        hidden_dim=32,
        mask_dim=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        mask_attention_mode='regions',
        deep_supervision=False,
    )
    logits = torch.tensor([[[[[-1.0]]], [[[1.0]]], [[[-2.0]]]]])

    attention_mask = decoder._attention_mask_at_shape(logits, (1, 1, 1))

    assert torch.equal(
        attention_mask,
        torch.tensor([[[True], [False], [True]]]),
    )


def test_cycle_schedule_visits_p5_p4_p3_once_per_three_layers():
    decoder = FixedSemanticMask2FormerDecoder3D(
        _Encoder(),
        num_classes=3,
        hidden_dim=32,
        mask_dim=16,
        num_heads=4,
        ffn_dim=64,
        query_feature_schedule='cycle',
        deep_supervision=True,
    )
    memory_lengths = []
    handles = [
        layer.cross_attention.register_forward_pre_hook(
            lambda _module, inputs: memory_lengths.append(inputs[1].shape[1])
        )
        for layer in decoder.layers
    ]

    try:
        decoder(_skips(), output_shape=(8, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    assert memory_lengths == [2, 8, 32] * 3


def test_dense_loss_backpropagates_through_masked_attention_and_pixel_decoder():
    decoder = FixedSemanticMask2FormerDecoder3D(
        _Encoder(),
        num_classes=3,
        hidden_dim=32,
        mask_dim=16,
        num_heads=4,
        num_layers=4,
        ffn_dim=64,
        deep_supervision=False,
    )

    logits = decoder(_skips(), output_shape=(8, 32, 32))
    logits.square().mean().backward()

    assert decoder.query_features.weight.grad is not None
    assert decoder.layers[-1].cross_attention.in_proj_weight.grad is not None
    assert decoder.pixel_decoder.mask_projection.weight.grad is not None
    assert (
        decoder.pixel_decoder.encoder_layers[0]
        .self_attention.sampling_offsets.weight.grad
        is not None
    )


def test_stock_nnunet_builder_constructs_mask2former(monkeypatch):
    import pumit.downstream.seg.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        'build_encoder',
        lambda name, plan, input_channels, config: _Encoder(),
    )
    architecture_kwargs = {
        **_architecture_kwargs(),
        'conv_op': 'torch.nn.Conv3d',
        'norm_op': 'torch.nn.InstanceNorm3d',
        'nonlin': 'torch.nn.LeakyReLU',
        'backbone_name': 'test',
        'backbone_config': {},
        'mask_attention_mode': 'classes',
        'query_feature_schedule': 'cycle',
    }
    configuration_manager = SimpleNamespace(
        network_arch_class_name=(
            'pumit.downstream.seg.mask2former.'
            'PlanAlignedMask2FormerSegmentationNetwork'
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

    assert isinstance(network, PlanAlignedMask2FormerSegmentationNetwork)
    assert network.decoder.query_features.num_embeddings == 3
    assert network.decoder.mask_attention_mode == 'classes'
    assert network.decoder.query_feature_schedule == 'cycle'
