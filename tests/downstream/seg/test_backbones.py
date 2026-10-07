import dataclasses
from collections.abc import Sequence
from pathlib import Path

import einops
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import pumit.downstream.seg.backbones.pumit_simple_fpn as pumit_simple_fpn
import pumit.downstream.seg.backbones.dinov3 as dinov3
import pumit.downstream.seg.backbones.dinov3_vit_adapter as dinov3_vit_adapter
import pumit.downstream.seg.backbones._pumit as pumit_checkpoint
import pumit.downstream.seg.backbones.pumit_pretrained_neck as pumit_pretrained_neck
import pumit.downstream.seg.backbones.pumit_vit_adapter as pumit_vit_adapter
import pumit.downstream.seg.backbones.random_simple_fpn as random_simple_fpn
import pumit.downstream.seg.backbones.sam_med3d as sam_med3d
import pumit.downstream.seg.backbones.three_dino as three_dino
import pumit.downstream.seg.backbones.three_dino_vit_adapter as three_dino_vit_adapter
from pumit.downstream.cls.backbones._sam_med3d_encoder import Attention
from pumit.downstream.seg.adapters.checkpoint import (
    adapt_input_weight,
    adapt_patch_embed_weight,
    repeat_single_channel_weight,
)
from pumit.downstream.seg.adapters.fixed_patch_vit import (
    FixedPatchViTPyramidBackbone,
    prepare_fixed_patch_inputs,
    replace_with_fixed_patch_embedding,
    validate_fixed_patch_size,
)
from pumit.downstream.seg.adapters.pretrained_neck import FixedGridPretrainedNeck
from pumit.downstream.seg.adapters.native_pyramid import native_encoder
from pumit.downstream.seg.adapters.pyramid import PlanAlignedPyramidEncoder3D, SimpleFPNEncoder3D
from pumit.downstream.seg.adapters.timm_vit import lift_2d_position_embedding
from pumit.downstream.seg.adapters.vit import select_evenly_spaced_layers
from pumit.downstream.seg.backbones.biomedclip import (
    BiomedCLIPFeatureBackbone,
    build_encoder as build_biomedclip,
    load_pretrained as load_biomedclip,
)
from pumit.downstream.seg.backbones.biomedclip_vit_adapter import (
    BiomedCLIPInteractiveBackbone,
)
from pumit.downstream.seg.backbones.eva02 import (
    Eva02LargeFeatureBackbone,
    build_encoder as build_eva02,
    load_pretrained as load_eva02,
)
from pumit.downstream.seg.backbones.eva02_vit_adapter import (
    Eva02LargeInteractiveBackbone,
)
from pumit.downstream.seg.backbones.sat_pro import (
    build_encoder as build_sat_pro,
    collapse_repeated_input_weight,
    load_pretrained as load_sat_pro,
)
from pumit.downstream.seg.backbones.stu_net import (
    build_encoder as build_stunet,
    load_pretrained as load_stunet,
)
from pumit.downstream.seg.backbones.suprem_unet import (
    build_encoder as build_suprem,
    load_pretrained as load_suprem,
)
from pumit.downstream.seg.backbones.sam_med3d_vit_adapter import (
    SAMMed3DInteractiveBackbone,
    build_encoder as build_sam_med3d_adapter,
    load_pretrained as load_sam_med3d_adapter,
    prepare_config as prepare_sam_med3d_adapter,
)
from pumit.downstream.seg.backbones.segvol_vit_adapter import (
    SegVolInteractiveBackbone,
    build_encoder as build_segvol_adapter,
    load_pretrained as load_segvol_adapter,
    prepare_config as prepare_segvol_adapter,
)
from pumit.downstream.seg.backbones.voco import (
    build_encoder as build_voco,
    load_pretrained as load_voco,
)
from pumit.downstream.seg.backbones.voco_l import (
    build_encoder as build_voco_l,
    load_pretrained as load_voco_l,
)
from pumit.downstream.seg.backbones.unimiss import (
    build_encoder as build_unimiss,
    load_pretrained as load_unimiss,
)
from pumit.downstream.seg.backbones.unimiss_plus import (
    build_encoder as build_unimiss_plus,
    load_pretrained as load_unimiss_plus,
)
from pumit.downstream.seg.plan import compute_stage_shapes
from tests.downstream.seg.conftest import make_plan
from pumit.model.vit import ViT, ViTConfig
from pumit.ucpt.seg.neck import SPADNeck

BIOMEDCLIP_WEIGHTS = Path('pretrained/biomedclip/open_clip_pytorch_model.bin')
DINO_V3_WEIGHTS = Path('pretrained/dinov3-vitl16/model.safetensors')
EVA02_WEIGHTS = Path('pretrained/eva02-l/model.safetensors')
THREE_DINO_WEIGHTS = Path('pretrained/3dino/3dino_vit_weights.pth')
UNIMISS_WEIGHTS = Path('pretrained/unimiss/UniMiss_small.pth')
UNIMISS_PLUS_WEIGHTS = Path('pretrained/unimiss_plus/UniMissPlus.pth')
SUPREM_WEIGHTS = Path('pretrained/suprem/supervised_suprem_unet_2100.pth')
STUNET_WEIGHTS = Path('pretrained/stu-net/large_ep4k.model')
VOCO_WEIGHTS = Path('pretrained/voco/VoCo_B_SSL_head.pt')
VOCO_L_WEIGHTS = Path('pretrained/voco/VoCo_L_SSL_head.pt')
SAT_PRO_WEIGHTS = Path('pretrained/sat/SAT_Pro.pth')
SEGVOL_WEIGHTS = Path('pretrained/segvol/pytorch_model.bin')
SEGVOL_SSL_WEIGHTS = Path('pretrained/segvol/vit_pretrain.ckpt')
SAM_MED3D_WEIGHTS = Path('pretrained/sam-med3d/sam_med3d_turbo.pth')


def _assert_parameter_layers_cover(backbone: nn.Module) -> None:
    layers = backbone.parameter_layers()
    assert layers
    assert all(layers)
    parameter_ids = [
        id(parameter)
        for group in layers
        for parameter in group
        if parameter.requires_grad
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {
        id(parameter)
        for parameter in backbone.parameters()
        if parameter.requires_grad
    }


@pytest.mark.parametrize(
    ('builder', 'first_module', 'last_module'),
    [
        (
            build_unimiss,
            lambda model: model.patch_embed3D0,
            lambda model: model.block4[-1],
        ),
        (
            build_unimiss_plus,
            lambda model: model.ConvBlock3D0,
            lambda model: model.block4[-1],
        ),
    ],
    ids=('unimiss', 'unimiss-plus'),
)
def test_unimiss_parameter_layers_are_ordered_and_exhaustive(
    builder,
    first_module,
    last_module,
):
    backbone = builder(_plan(), 1, {}).backbone
    layers = backbone.parameter_layers()

    assert {
        id(parameter)
        for parameter in layers[0]
    } == {
        id(parameter)
        for parameter in first_module(backbone.model).parameters()
    }
    assert {
        id(parameter)
        for parameter in last_module(backbone.model).parameters()
    }.issubset({
        id(parameter)
        for parameter in layers[-1]
    })
    _assert_parameter_layers_cover(backbone)


def _plan():
    return make_plan([32, 64, 128, 256, 320, 320], n_blocks_per_stage=[1, 3, 4, 6, 6, 6])


def test_fixed_patch_size_accepts_only_explicit_power_of_two_depths():
    assert validate_fixed_patch_size((4, 16, 16), in_plane_patch_size=16) == (4, 16, 16)

    with pytest.raises(ValueError, match='D in'):
        validate_fixed_patch_size((6, 16, 16), in_plane_patch_size=16)


def test_fixed_patch_weight_uniformly_inflates_2d_checkpoint():
    source = torch.randn(5, 3, 16, 16)

    adapted = adapt_patch_embed_weight(source, (4, 16, 16))

    assert tuple(adapted.shape) == (5, 3, 4, 16, 16)
    torch.testing.assert_close(adapted.sum(dim=2), source)


def test_fixed_patch_weight_inflation_preserves_mean_2d_projection():
    source = torch.randn(5, 3, 4, 4)
    bias = torch.randn(5)
    x = torch.randn(2, 3, 8, 8, 8)

    inflated = adapt_patch_embed_weight(source, (4, 4, 4))
    actual = F.conv3d(x, inflated, bias, stride=(4, 4, 4))
    per_slice = F.conv2d(
        x.permute(0, 2, 1, 3, 4).flatten(0, 1),
        source,
        bias,
        stride=(4, 4),
    )
    expected = per_slice.reshape(2, 8, 5, 2, 2)
    expected = expected.reshape(2, 2, 4, 5, 2, 2).mean(dim=2).permute(0, 2, 1, 3, 4)

    torch.testing.assert_close(actual, expected)


def test_fixed_patch_weight_reduces_full_3d_checkpoint_by_depth_groups():
    source = torch.randn(5, 3, 16, 16, 16)

    adapted = adapt_patch_embed_weight(source, (4, 16, 16))
    expected = source.reshape(5, 3, 4, 4, 16, 16).sum(dim=3)

    torch.testing.assert_close(adapted, expected)


def test_fixed_patch_weight_reduction_matches_repeat_interleaved_input():
    source = torch.randn(5, 3, 16, 4, 4, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64)
    x = torch.randn(2, 3, 8, 8, 8, dtype=torch.float64)

    reduced = adapt_patch_embed_weight(source, (4, 4, 4))
    actual = F.conv3d(x, reduced, bias, stride=(4, 4, 4))
    expected = F.conv3d(
        x.repeat_interleave(4, dim=2),
        source,
        bias,
        stride=(16, 4, 4),
    )

    torch.testing.assert_close(actual, expected)


def test_fixed_patch_weight_rejects_nondivisible_3d_depth():
    with pytest.raises(ValueError, match='integer multiple'):
        adapt_patch_embed_weight(torch.randn(2, 3, 6, 16, 16), (4, 16, 16))


def test_fixed_patch_size_accepts_power_of_two_in_plane_divisors():
    assert validate_fixed_patch_size((8, 8, 8), in_plane_patch_size=16) == (8, 8, 8)

    with pytest.raises(ValueError, match='power-of-two'):
        validate_fixed_patch_size((8, 12, 12), in_plane_patch_size=16)
    with pytest.raises(ValueError, match='power-of-two'):
        validate_fixed_patch_size((8, 8, 16), in_plane_patch_size=16)


def test_fixed_patch_weight_in_plane_reduction_matches_nn_upsampled_input_2d():
    source = torch.randn(5, 3, 16, 16, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64)
    x = torch.randn(2, 3, 8, 16, 16, dtype=torch.float64)

    reduced = adapt_patch_embed_weight(source, (4, 8, 8))
    assert tuple(reduced.shape) == (5, 3, 4, 8, 8)
    actual = F.conv3d(x, reduced, bias, stride=(4, 8, 8))
    inflated = adapt_patch_embed_weight(source, (4, 16, 16))
    upsampled = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
    expected = F.conv3d(upsampled, inflated, bias, stride=(4, 16, 16))

    torch.testing.assert_close(actual, expected)


def test_fixed_patch_weight_in_plane_reduction_matches_nn_upsampled_input_3d():
    source = torch.randn(5, 3, 16, 16, 16, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64)
    x = torch.randn(2, 3, 16, 16, 16, dtype=torch.float64)

    reduced = adapt_patch_embed_weight(source, (8, 8, 8))
    assert tuple(reduced.shape) == (5, 3, 8, 8, 8)
    actual = F.conv3d(x, reduced, bias, stride=(8, 8, 8))
    upsampled = (
        x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
    )
    expected = F.conv3d(upsampled, source, bias, stride=(16, 16, 16))

    torch.testing.assert_close(actual, expected)


def test_fixed_patch_weight_rejects_nondivisible_in_plane():
    with pytest.raises(ValueError, match='integer multiple'):
        adapt_patch_embed_weight(torch.randn(2, 3, 16, 12, 12), (4, 8, 8))


def test_dinov3_checkpoint_loader_adapts_patch_weight_before_strict_shape_load(monkeypatch):
    config = ViTConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        patch_size=16,
        num_register_tokens=1,
        in_channels=3,
        pos_embed_rescale=None,
    )
    source_state = ViT(config).state_dict()
    source_weight = torch.randn(16, 3, 16, 16)
    source_state[dinov3.PATCH_WEIGHT_KEY] = source_weight
    target = ViT(config)
    replace_with_fixed_patch_embedding(target, (4, 16, 16))
    monkeypatch.setattr(dinov3.st, 'load_file', lambda _path: source_state)

    dinov3.load_vit_pretrained(target, Path('unused.safetensors'))

    loaded_weight = target.embeddings.patch_embeddings.weight
    assert tuple(loaded_weight.shape) == (16, 3, 4, 16, 16)
    torch.testing.assert_close(loaded_weight.sum(dim=2), source_weight)


def test_fixed_depth_16_patch_embedding_keeps_existing_checkpoint_keys_and_shapes():
    config = ViTConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        patch_size=16,
        num_register_tokens=1,
        in_channels=3,
        pos_embed_rescale=None,
    )
    source_state = ViT(config).state_dict()
    target = ViT(config)
    replace_with_fixed_patch_embedding(target, (16, 16, 16))

    target.load_state_dict(source_state, strict=True)

    assert target.state_dict().keys() == source_state.keys()
    assert tuple(target.embeddings.patch_embeddings.weight.shape) == (16, 3, 16, 16, 16)


def test_fixed_patch_vit_pyramid_returns_selected_normalized_depths(monkeypatch):
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
    vit = ViT(
        ViTConfig(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=32,
            patch_size=4,
            num_register_tokens=1,
            in_channels=1,
            pos_embed_rescale=None,
        )
    )
    backbone = FixedPatchViTPyramidBackbone(vit, 1, (1, 2, 3, 4))

    features = backbone(torch.randn(1, 1, 16, 16, 16))

    assert [tuple(feature.shape) for feature in features] == [
        (1, 16, 4, 4, 4),
    ] * 4
    _assert_parameter_layers_cover(backbone)


def test_fixed_patch_vit_pyramid_rejects_nonpositive_feature_depth():
    vit = ViT(
        ViTConfig(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=32,
            patch_size=4,
            num_register_tokens=1,
            in_channels=1,
            pos_embed_rescale=None,
        )
    )

    with pytest.raises(ValueError, match='feature_layers'):
        FixedPatchViTPyramidBackbone(vit, 1, (0, 2, 3, 4))


_PUMIT_VIT_CONFIG = ViTConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128,
    num_register_tokens=2,
    patch_size=16,
    in_channels=3,
    pos_embed_rescale=None,
)


def _pumit_checkpoint(
    tmp_path: Path,
    config: ViTConfig = _PUMIT_VIT_CONFIG,
    seg_hidden_size: int | None = None,
) -> Path:
    """Write a tiny checkpoint mirroring the current UCPT ViT layout."""
    model_state = {
        f'teacher_vit.{key}': value
        for key, value in ViT(config).state_dict().items()
    }
    model_state.update({f'vit.{key}': value for key, value in ViT(config).state_dict().items()})
    model_config = {
        'embed_dim': config.hidden_size,
        'depth': config.num_hidden_layers,
        'num_heads': config.num_attention_heads,
        'mlp_ratio': config.intermediate_size / config.hidden_size,
        'n_register_tokens': config.num_register_tokens,
    }
    if seg_hidden_size is not None:
        model_config['seg_hidden_size'] = seg_hidden_size
        neck = SPADNeck(in_channels=config.hidden_size, hidden_size=seg_hidden_size)
        model_state.update({
            f'ema_seg.neck.{key}': value
            for key, value in neck.state_dict().items()
        })
        # Sibling EMA segmentation components that downstream transfer must leave behind.
        model_state['ema_seg.text_encoder.proj.weight'] = torch.randn(seg_hidden_size, 8)
        model_state['ema_seg.fusion.layers.0.norm.weight'] = torch.randn(seg_hidden_size)
        model_state['ema_seg.head.mask_proj.weight'] = torch.randn(seg_hidden_size, seg_hidden_size)
    checkpoint_path = tmp_path / 'ucpt-checkpoint.pt'
    torch.save(
        {
            'model': model_state,
            'config': {'model': model_config},
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_pumit_vit_adapter_uses_the_dinov3_adapter_config(monkeypatch):
    checkpoint = {
        'config': {
            'model': {
                'embed_dim': 1024,
                'depth': 24,
                'num_heads': 16,
                'mlp_ratio': 4.0,
                'n_register_tokens': 4,
            },
        },
    }
    monkeypatch.setattr(pumit_checkpoint.torch, 'load', lambda *args, **kwargs: checkpoint)

    config = pumit_vit_adapter.prepare_config(Path('ucpt.pt'), 'ucpt', False)

    assert config == dinov3_vit_adapter.prepare_config(Path('dinov3.safetensors'), None, False)


def test_pumit_vit_adapter_loads_only_the_ema_vit_and_reduces_patch_depth(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = _pumit_checkpoint(tmp_path)

    class FakeViTAdapterEncoder3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.vit = ViT(_PUMIT_VIT_CONFIG)
            replace_with_fixed_patch_embedding(self.backbone.vit, (4, 16, 16))

    monkeypatch.setattr(
        pumit_vit_adapter,
        'ViTAdapterEncoder3D',
        FakeViTAdapterEncoder3D,
    )
    encoder = FakeViTAdapterEncoder3D()

    pumit_vit_adapter.load_pretrained(encoder, checkpoint_path)

    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
    for key, value in encoder.backbone.vit.state_dict().items():
        expected = source[f'teacher_vit.{key}']
        if key == 'embeddings.patch_embeddings.weight':
            expected = adapt_patch_embed_weight(expected, (4, 16, 16))
        torch.testing.assert_close(value, expected)
    assert not hasattr(encoder.backbone, 'neck')


def test_pumit_simple_fpn_loads_only_the_ema_vit_and_reduces_patch_depth(tmp_path):
    checkpoint_path = _pumit_checkpoint(
        tmp_path,
        dataclasses.replace(_PUMIT_VIT_CONFIG, num_hidden_layers=24),
    )
    config = pumit_simple_fpn.prepare_config(checkpoint_path, 'ucpt', False)
    assert config['feature_layers'] == [6, 12, 18, 24]
    assert 'checkpoint_format' not in config
    config['vit_patch_size'] = [4, 16, 16]
    encoder = pumit_simple_fpn.build_encoder(_plan(), 1, config)

    pumit_simple_fpn.load_pretrained(encoder, checkpoint_path)

    assert encoder.backbone.feature_layers == (6, 12, 18, 24)
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
    for key, value in encoder.backbone.vit.state_dict().items():
        expected = source[f'teacher_vit.{key}']
        if key == 'embeddings.patch_embeddings.weight':
            expected = adapt_patch_embed_weight(expected, (4, 16, 16))
        torch.testing.assert_close(value, expected)
    assert not hasattr(encoder.backbone, 'neck')
    _assert_parameter_layers_cover(encoder.backbone)


_NECK_HIDDEN_SIZE = 16
_NECK_PATCH_SIZE = [16, 16, 16]
_NECK_INPUT_SHAPE = (32, 32, 32)
_NECK_STRIDES = ([1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2])


def _neck_plan(
    features_per_stage: Sequence[int] = (4, 8, 8, _NECK_HIDDEN_SIZE, 32, _PUMIT_VIT_CONFIG.hidden_size),
    strides: Sequence[Sequence[int]] = _NECK_STRIDES,
):
    """Build an isotropic six-stage plan whose P2-P4 strides match the pretrained neck."""
    return make_plan(features_per_stage, strides, n_blocks_per_stage=[1, 3, 4, 6, 6, 6])


@pytest.fixture
def cpu_attention(monkeypatch):
    """Route the ViT's xformers attention through SDPA so CPU float32 forwards run."""
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


def _pretrained_neck_encoder(tmp_path, plan=None) -> tuple[nn.Module, Path]:
    checkpoint_path = _pumit_checkpoint(tmp_path, seg_hidden_size=_NECK_HIDDEN_SIZE)
    config = pumit_pretrained_neck.prepare_config(checkpoint_path, 'ucpt', False)
    config['vit_patch_size'] = _NECK_PATCH_SIZE
    encoder = pumit_pretrained_neck.build_encoder(plan if plan is not None else _neck_plan(), 1, config)
    return encoder, checkpoint_path


def test_fixed_grid_pretrained_neck_matches_the_ucpt_da0_path():
    source = SPADNeck(in_channels=64, hidden_size=_NECK_HIDDEN_SIZE)
    fixed = FixedGridPretrainedNeck(in_channels=64, hidden_size=_NECK_HIDDEN_SIZE)
    fixed.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 64, 2, 3, 4)

    expected = source(x, 0)
    actual = fixed(x)

    assert source.state_dict().keys() == fixed.state_dict().keys()
    for level in ('1/4', '1/8', '1/16'):
        torch.testing.assert_close(actual[level], expected[level])


def test_pumit_pretrained_neck_loads_the_matched_ema_vit_and_neck_pair(tmp_path):
    checkpoint_path = _pumit_checkpoint(tmp_path, seg_hidden_size=_NECK_HIDDEN_SIZE)
    config = pumit_pretrained_neck.prepare_config(checkpoint_path, 'ucpt', False)
    assert config['neck_hidden_size'] == _NECK_HIDDEN_SIZE
    assert 'feature_layers' not in config
    config['vit_patch_size'] = _NECK_PATCH_SIZE
    encoder = pumit_pretrained_neck.build_encoder(_neck_plan(), 1, config)

    pumit_pretrained_neck.load_pretrained(encoder, checkpoint_path)

    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
    for key, value in encoder.backbone.vit.state_dict().items():
        expected = source[f'teacher_vit.{key}']
        if key == 'embeddings.patch_embeddings.weight':
            expected = adapt_patch_embed_weight(expected, tuple(_NECK_PATCH_SIZE))
        torch.testing.assert_close(value, expected)
    for key, value in encoder.backbone.neck.state_dict().items():
        torch.testing.assert_close(value, source[f'ema_seg.neck.{key}'])
    assert set(encoder.backbone.state_dict()) == {
        *(f'vit.{key}' for key in encoder.backbone.vit.state_dict()),
        *(f'neck.{key}' for key in encoder.backbone.neck.state_dict()),
    }
    _assert_parameter_layers_cover(encoder.backbone)


def test_pumit_pretrained_neck_returns_the_plan_aligned_pyramid(tmp_path, cpu_attention):
    encoder, checkpoint_path = _pretrained_neck_encoder(tmp_path)
    pumit_pretrained_neck.load_pretrained(encoder, checkpoint_path)
    encoder.eval()

    with torch.no_grad():
        skips = encoder(torch.randn(1, 1, *_NECK_INPUT_SHAPE))

    assert [skip.shape[1] for skip in skips] == [4, 8, 8, _NECK_HIDDEN_SIZE, 32, 64]
    assert [tuple(skip.shape[2:]) for skip in skips] == [
        (32, 32, 32),
        (16, 16, 16),
        (8, 8, 8),
        (4, 4, 4),
        (2, 2, 2),
        (1, 1, 1),
    ]
    # P3 needs no projection because the plan stage already matches the pretrained pyramid width.
    assert isinstance(encoder.neck_projections[1], nn.Identity)


def test_pumit_pretrained_neck_pyramid_comes_from_the_normalized_final_vit_feature(tmp_path, cpu_attention):
    encoder, checkpoint_path = _pretrained_neck_encoder(tmp_path)
    pumit_pretrained_neck.load_pretrained(encoder, checkpoint_path)
    encoder.eval()
    x = torch.randn(1, 1, *_NECK_INPUT_SHAPE)

    with torch.no_grad():
        skips = encoder(x)
        vit = encoder.backbone.vit
        tokens, rope, patch_shape = prepare_fixed_patch_inputs(
            vit,
            x.expand(-1, vit.config.in_channels, -1, -1, -1),
        )
        feature = einops.rearrange(
            vit(tokens, rope)[:, vit.n_prefix:],
            'b (d h w) c -> b c d h w',
            d=patch_shape[0],
            h=patch_shape[1],
            w=patch_shape[2],
        )
        levels = encoder.backbone.neck(feature)
        pooled = F.max_pool3d(feature, kernel_size=2, stride=2)

    for stage, level in enumerate(('1/4', '1/8', '1/16')):
        expected = encoder.neck_projections[stage](levels[level])
        torch.testing.assert_close(skips[stage + 2], expected)
    torch.testing.assert_close(skips[5], pooled)


def test_pumit_pretrained_neck_keeps_scratch_adapters_outside_the_frozen_backbone(tmp_path, cpu_attention):
    encoder, checkpoint_path = _pretrained_neck_encoder(tmp_path)
    pumit_pretrained_neck.load_pretrained(encoder, checkpoint_path)
    backbone_ids = {id(parameter) for parameter in encoder.backbone.parameters()}
    assert backbone_ids == {
        id(parameter)
        for module in (encoder.backbone.vit, encoder.backbone.neck)
        for parameter in module.parameters()
    }
    scratch = [
        *encoder.high_resolution_stem.parameters(),
        *encoder.neck_projections.parameters(),
    ]
    assert scratch
    assert not backbone_ids & {id(parameter) for parameter in scratch}

    # Mirror DownstreamSegTrainer.configure_optimizers, which freezes encoder.backbone as one unit.
    for parameter in encoder.backbone.parameters():
        parameter.requires_grad = False
    skips = encoder(torch.randn(1, 1, *_NECK_INPUT_SHAPE))
    sum(skip.square().mean() for skip in skips).backward()

    assert all(parameter.grad is None for parameter in encoder.backbone.parameters())
    assert all(parameter.grad is not None for parameter in scratch)
    # P3 and P5 reach the decoder without any trainable adaptation of their own.
    assert not skips[3].requires_grad
    assert not skips[5].requires_grad
    assert skips[2].requires_grad


def test_pumit_pretrained_neck_rejects_a_plan_whose_pyramid_strides_miss_the_neck(tmp_path):
    checkpoint_path = _pumit_checkpoint(tmp_path, seg_hidden_size=_NECK_HIDDEN_SIZE)
    config = pumit_pretrained_neck.prepare_config(checkpoint_path, 'ucpt', False)
    config['vit_patch_size'] = _NECK_PATCH_SIZE
    plan = _neck_plan(strides=([1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]))

    with pytest.raises(ValueError, match='pretrained neck level'):
        pumit_pretrained_neck.build_encoder(plan, 1, config)


def test_pumit_pretrained_neck_rejects_a_plan_that_rewidens_the_pooled_vit_stage(tmp_path):
    checkpoint_path = _pumit_checkpoint(tmp_path, seg_hidden_size=_NECK_HIDDEN_SIZE)
    config = pumit_pretrained_neck.prepare_config(checkpoint_path, 'ucpt', False)
    config['vit_patch_size'] = _NECK_PATCH_SIZE
    plan = _neck_plan(features_per_stage=(4, 8, 8, _NECK_HIDDEN_SIZE, 32, 96))

    with pytest.raises(ValueError, match='pretrained width'):
        pumit_pretrained_neck.build_encoder(plan, 1, config)


def test_pumit_pretrained_neck_requires_a_ucpt_checkpoint_carrying_the_neck(tmp_path):
    checkpoint_path = _pumit_checkpoint(tmp_path)

    with pytest.raises(KeyError, match='seg_hidden_size'):
        pumit_pretrained_neck.prepare_config(checkpoint_path, 'ucpt', False)


def test_pumit_pretrained_neck_completes_the_stock_unet_readout(tmp_path, cpu_attention):
    from pumit.downstream.seg.network import PlanAlignedSegmentationNetwork

    checkpoint_path = _pumit_checkpoint(tmp_path, seg_hidden_size=_NECK_HIDDEN_SIZE)
    config = pumit_pretrained_neck.prepare_config(checkpoint_path, 'ucpt', False)
    config['vit_patch_size'] = _NECK_PATCH_SIZE
    network = PlanAlignedSegmentationNetwork(
        1,
        3,
        backbone_name='pumit-pretrained-neck',
        backbone_config=config,
        features_per_stage=[4, 8, 8, _NECK_HIDDEN_SIZE, 32, _PUMIT_VIT_CONFIG.hidden_size],
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[list(stride) for stride in _NECK_STRIDES],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder=[1] * 5,
        conv_op=nn.Conv3d,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
    )
    network.eval()

    with torch.no_grad():
        logits = network(torch.randn(1, 1, *_NECK_INPUT_SHAPE))

    assert [tuple(output.shape) for output in logits] == [
        (1, 3, 32, 32, 32),
        (1, 3, 16, 16, 16),
        (1, 3, 8, 8, 8),
        (1, 3, 4, 4, 4),
        (1, 3, 2, 2, 2),
    ]


def test_single_channel_weight_repetition_preserves_equal_channel_response():
    weight = torch.randn(5, 1, 3, 3, 3)

    inflated = repeat_single_channel_weight(weight, 3)

    assert tuple(inflated.shape) == (5, 3, 3, 3, 3)
    torch.testing.assert_close(inflated.sum(dim=1, keepdim=True), weight)


def test_flat_vit_input_adaptation_preserves_equal_channel_response():
    weight = torch.randn(5, 3, 3, 3)

    adapted = adapt_input_weight(weight, 4)

    assert tuple(adapted.shape) == (5, 4, 3, 3)
    torch.testing.assert_close(adapted.sum(dim=1), weight.sum(dim=1))


def test_flat_vit_position_lifting_replicates_interpolated_in_plane_table_along_depth():
    position = torch.randn(1, 5, 8)

    lifted = lift_2d_position_embedding(position, (3, 4, 6), num_prefix_tokens=1)

    assert tuple(lifted.shape) == (1, 73, 8)
    torch.testing.assert_close(lifted[:, :1], position[:, :1])
    spatial = lifted[:, 1:].reshape(1, 3, 4, 6, 8)
    torch.testing.assert_close(spatial[:, 0], spatial[:, 1])
    torch.testing.assert_close(spatial[:, 1], spatial[:, 2])


def test_flat_vit_uses_four_evenly_spaced_transformer_depths():
    assert select_evenly_spaced_layers(24) == (6, 12, 18, 24)
    assert select_evenly_spaced_layers(12) == (3, 6, 9, 12)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not DINO_V3_WEIGHTS.exists(),
    reason='DINOv3 CUDA smoke test requires the released checkpoint',
)
def test_dinov3_loads_real_weights_and_returns_plan_aligned_multilayer_pyramid():
    config = dinov3.prepare_config(DINO_V3_WEIGHTS, None, False)
    config['vit_patch_size'] = [16, 16, 16]
    encoder = dinov3.build_encoder(_plan(), 1, config).cuda().eval()
    dinov3.load_pretrained(encoder, DINO_V3_WEIGHTS)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        skips = encoder(torch.randn(1, 1, 32, 32, 64, device='cuda'))

    assert config['feature_layers'] == [6, 12, 18, 24]
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 32, 64),
        (1, 64, 16, 16, 32),
        (1, 128, 8, 8, 16),
        (1, 256, 4, 4, 8),
        (1, 320, 2, 2, 4),
        (1, 320, 1, 1, 2),
    ]
    _assert_parameter_layers_cover(encoder.backbone)


class _TinySAMMed3DPatchEmbed(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(input_channels, 4, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 4, 1)


class _TinySAMMed3DBlock(nn.Module):
    def __init__(self, rel_pos_length: int):
        super().__init__()
        self.attn = nn.Module()
        self.attn.rel_pos_d = nn.Parameter(torch.randn(rel_pos_length, 1))
        self.attn.rel_pos_h = nn.Parameter(torch.randn(rel_pos_length, 1))
        self.attn.rel_pos_w = nn.Parameter(torch.randn(rel_pos_length, 1))

    def forward(self, x):
        return x


class _TinySAMMed3D(nn.Module):
    def __init__(self, input_channels: int, rel_pos_length: int = 15):
        super().__init__()
        self.patch_embed = _TinySAMMed3DPatchEmbed(input_channels)
        self.pos_embed = nn.Parameter(torch.randn(1, 2, 2, 2, 4))
        self.blocks = nn.ModuleList(
            _TinySAMMed3DBlock(rel_pos_length)
            for _ in range(12)
        )
        self.neck = nn.Conv3d(4, sam_med3d.OUT_CHANS, kernel_size=1)


def test_sam_med3d_uses_official_layer_norm_epsilon():
    encoder = sam_med3d.build_sam_med3d(1)

    assert all(block.norm1.eps == 1e-6 and block.norm2.eps == 1e-6 for block in encoder.blocks)


def test_sam_med3d_relative_position_supports_non_cubic_grids():
    attention = Attention(
        dim=4,
        num_heads=1,
        use_rel_pos=True,
        input_size=(2, 2, 2),
    )

    output = attention(torch.randn(1, 2, 3, 4, 4))

    assert tuple(output.shape) == (1, 2, 3, 4, 4)
    assert torch.isfinite(output).all()


def test_sam_med3d_interpolates_position_without_resizing_input(monkeypatch):
    monkeypatch.setattr(
        sam_med3d,
        'build_sam_med3d',
        lambda input_channels: _TinySAMMed3D(input_channels),
    )
    backbone = sam_med3d.SAMMed3DFeatureBackbone(1, (16, 16, 16))
    x = torch.randn(1, 1, 32, 48, 64)

    features = backbone(x)

    tokens = backbone.model.patch_embed(x)
    position = F.interpolate(
        backbone.model.pos_embed.permute(0, 4, 1, 2, 3),
        size=(2, 3, 4),
        mode='trilinear',
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)
    hidden = (tokens + position).permute(0, 4, 1, 2, 3)
    expected_final = backbone.model.neck(hidden)
    assert tuple(features.shape) == (1, sam_med3d.OUT_CHANS, 2, 3, 4)
    torch.testing.assert_close(features, expected_final)


def test_sam_med3d_loads_only_image_encoder_and_inflates_input_channels(monkeypatch):
    source = _TinySAMMed3D(1).state_dict()
    monkeypatch.setattr(
        sam_med3d,
        'build_sam_med3d',
        lambda input_channels: _TinySAMMed3D(input_channels),
    )
    monkeypatch.setattr(sam_med3d, '_load_state_dict', lambda weights: source)
    config = {'vit_patch_size': [16, 16, 16]}
    encoder = sam_med3d.build_encoder(_plan(), 3, config)

    sam_med3d.load_pretrained(encoder, Path('unused.pth'))

    inflated = encoder.backbone.model.patch_embed.proj.weight
    torch.testing.assert_close(
        inflated.sum(dim=1, keepdim=True),
        source['patch_embed.proj.weight'],
    )


def test_sam_med3d_reduces_checkpoint_to_fixed_depth_patch(monkeypatch):
    source = _TinySAMMed3D(1, rel_pos_length=27).state_dict()
    monkeypatch.setattr(
        sam_med3d,
        'build_sam_med3d',
        lambda input_channels: _TinySAMMed3D(input_channels),
    )
    monkeypatch.setattr(sam_med3d, '_load_state_dict', lambda weights: source)
    config = {'vit_patch_size': [4, 16, 16]}
    encoder = sam_med3d.build_encoder(_plan(), 1, config)

    sam_med3d.load_pretrained(encoder, Path('unused.pth'))

    expected = adapt_patch_embed_weight(source['patch_embed.proj.weight'], (4, 16, 16))
    actual = encoder.backbone.model.patch_embed.proj.weight
    assert encoder.backbone.feature_stride == (4, 16, 16)
    assert tuple(actual.shape[2:]) == (4, 16, 16)
    assert tuple(encoder.backbone.model.blocks[0].attn.rel_pos_d.shape) == (15, 1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not SAM_MED3D_WEIGHTS.exists(),
    reason='released SAM-Med3D checkpoint is not available',
)
def test_sam_med3d_vit_adapter_matches_the_original_full_trunk_path():
    plan = dataclasses.replace(
        _plan(),
        output_channels=(32, 64, 128, 256, 384, 768),
        strides=((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    )
    config = prepare_sam_med3d_adapter(SAM_MED3D_WEIGHTS, None, False)
    encoder = build_sam_med3d_adapter(plan, 1, {**config, 'vit_patch_size': [8, 16, 16]})
    load_sam_med3d_adapter(encoder, SAM_MED3D_WEIGHTS)
    encoder.eval()
    backbone = encoder.backbone
    assert isinstance(backbone, SAMMed3DInteractiveBackbone)
    assert not hasattr(backbone.model, 'neck')
    _assert_parameter_layers_cover(backbone)

    x = torch.randn(1, 1, 32, 64, 64)
    with torch.inference_mode():
        skips = encoder(x)
        assert [tuple(skip.shape) for skip in skips] == [
            (1, 32, 32, 64, 64),
            (1, 64, 32, 32, 32),
            (1, 128, 16, 16, 16),
            (1, 256, 8, 8, 8),
            (1, 384, 4, 4, 4),
            (1, 768, 2, 2, 2),
        ]

        # The interactive block path must reproduce the released trunk computation exactly.
        tokens, grid, context = backbone.prepare_tokens(x)
        interactive = backbone.run_blocks(tokens, 0, 12, context).view(1, *grid, -1)
        reference = sam_med3d.SAMMed3DFeatureBackbone(1, [8, 16, 16])
        state_dict = sam_med3d.adapt_state_dict(
            dict(sam_med3d._load_state_dict(SAM_MED3D_WEIGHTS)),
            reference.model,
        )
        state_dict['patch_embed.proj.weight'] = adapt_patch_embed_weight(
            state_dict['patch_embed.proj.weight'],
            (8, 16, 16),
        )
        reference.model.load_state_dict(state_dict, strict=True)
        reference.eval()
        features = reference.model.patch_embed(x)
        position = reference.model.pos_embed
        if position.shape[1:4] != features.shape[1:4]:
            position = F.interpolate(
                position.permute(0, 4, 1, 2, 3),
                size=features.shape[1:4],
                mode='trilinear',
                align_corners=False,
            ).permute(0, 2, 3, 4, 1)
        features = features + position
        for block in reference.model.blocks:
            features = block(features)
    torch.testing.assert_close(interactive, features, rtol=0, atol=0)


@pytest.mark.skipif(
    not UNIMISS_PLUS_WEIGHTS.exists(),
    reason='released UniMiSS+ checkpoint is not available',
)
def test_unimiss_plus_loads_real_weights_and_returns_plan_aligned_pyramid():
    encoder = build_unimiss_plus(_plan(), 1, {})
    load_unimiss_plus(encoder, UNIMISS_PLUS_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 32, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 32, 64),
        (1, 64, 16, 16, 32),
        (1, 128, 8, 8, 16),
        (1, 256, 4, 4, 8),
        (1, 320, 2, 2, 4),
        (1, 320, 1, 1, 2),
    ]
    assert not hasattr(encoder.backbone.model, 'head_new')
    model = encoder.backbone.model
    assert model.ConvBlock3D1[0].conv.stride == (2, 2, 2)
    assert [
        patch_embed.proj.conv.stride
        for patch_embed in (
            model.patch_embed3D1,
            model.patch_embed3D2,
            model.patch_embed3D3,
            model.patch_embed3D4,
        )
    ] == [(2, 2, 2)] * 4
    assert model.patch_embed3D1.grid_size == (8, 24, 24)
    assert model.patch_embed3D4.grid_size == (4, 3, 3)


@pytest.mark.skipif(
    not UNIMISS_PLUS_WEIGHTS.exists(),
    reason='released UniMiSS+ checkpoint is not available',
)
def test_unimiss_plus_inflates_its_pretrained_stem_for_multiple_channels():
    single_channel = build_unimiss_plus(_plan(), 1, {})
    load_unimiss_plus(single_channel, UNIMISS_PLUS_WEIGHTS)
    multi_channel = build_unimiss_plus(_plan(), 3, {})
    load_unimiss_plus(multi_channel, UNIMISS_PLUS_WEIGHTS)

    source = single_channel.backbone.model.ConvBlock3D0[0].conv.weight
    inflated = multi_channel.backbone.model.ConvBlock3D0[0].conv.weight

    assert tuple(inflated.shape) == (32, 3, 3, 3, 3)
    torch.testing.assert_close(inflated.sum(dim=1, keepdim=True), source)


def _sat_pro_plan():
    return make_plan(
        [128, 128, 256, 512, 1024, 1536],
        [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        n_blocks_per_stage=[3] * 6,
        n_conv_per_stage_decoder=[3] * 5,
    )


def test_sat_pro_requires_the_native_six_stage_plan():
    with pytest.raises(ValueError, match='requires plan channels'):
        build_sat_pro(_plan(), 1, {})

    strided_stem = dataclasses.replace(
        _sat_pro_plan(),
        strides=((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
    )
    with pytest.raises(ValueError, match='starting at stride'):
        build_sat_pro(strided_stem, 1, {})


def test_sat_pro_strided_convs_follow_anisotropic_plan_strides():
    encoder = build_sat_pro(_sat_pro_plan(), 1, {})
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 16, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 128, 16, 64, 64),
        (1, 128, 8, 32, 32),
        (1, 256, 4, 16, 16),
        (1, 512, 2, 8, 8),
        (1, 1024, 1, 4, 4),
        (1, 1536, 1, 2, 2),
    ]


def test_sat_pro_encoder_is_entirely_pretrained():
    encoder = build_sat_pro(_sat_pro_plan(), 1, {})
    assert {id(parameter) for parameter in encoder.parameters()} == {
        id(parameter) for parameter in encoder.backbone.parameters()
    }
    layers = encoder.backbone.parameter_layers()
    assert len(layers) == 6
    _assert_parameter_layers_cover(encoder.backbone)


def test_sat_pro_stem_collapse_matches_channel_repeated_input():
    conv = nn.Conv3d(3, 5, 3, padding=1)
    x = torch.randn(1, 1, 6, 6, 6)

    collapsed = nn.Conv3d(1, 5, 3, padding=1)
    with torch.no_grad():
        collapsed.weight.copy_(collapse_repeated_input_weight(conv.weight))
        collapsed.bias.copy_(conv.bias)
        torch.testing.assert_close(collapsed(x), conv(x.expand(-1, 3, -1, -1, -1)))


@pytest.mark.skipif(
    not SAT_PRO_WEIGHTS.exists(),
    reason='released SAT-Pro checkpoint is not available',
)
def test_sat_pro_loads_real_weights_strictly_and_returns_plan_aligned_pyramid():
    encoder = build_sat_pro(_sat_pro_plan(), 1, {})
    load_sat_pro(encoder, SAT_PRO_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 16, 64, 64))

    assert all(torch.isfinite(skip).all() for skip in skips)
    assert [skip.shape[1] for skip in skips] == [128, 128, 256, 512, 1024, 1536]


def _stunet_plan():
    return make_plan(
        [64, 128, 256, 512, 1024, 1024],
        [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        n_blocks_per_stage=[2] * 6,
        n_conv_per_stage_decoder=[2] * 5,
    )


def test_stunet_requires_the_native_six_stage_plan():
    with pytest.raises(ValueError, match='requires plan channels'):
        build_stunet(_plan(), 1, {})

    strided_stem = dataclasses.replace(
        _stunet_plan(),
        strides=((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
    )
    with pytest.raises(ValueError, match='starting at stride'):
        build_stunet(strided_stem, 1, {})


def test_stunet_strided_convs_follow_anisotropic_plan_strides():
    encoder = build_stunet(_stunet_plan(), 1, {})

    with torch.no_grad():
        skips = encoder(torch.randn(1, 1, 16, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 64, 16, 64, 64),
        (1, 128, 8, 32, 32),
        (1, 256, 4, 16, 16),
        (1, 512, 2, 8, 8),
        (1, 1024, 1, 4, 4),
        (1, 1024, 1, 2, 2),
    ]


def test_stunet_encoder_is_entirely_pretrained():
    encoder = build_stunet(_stunet_plan(), 1, {})
    assert {id(parameter) for parameter in encoder.parameters()} == {
        id(parameter) for parameter in encoder.backbone.parameters()
    }
    layers = encoder.backbone.parameter_layers()
    assert len(layers) == 6
    assert {
        id(parameter) for parameter in layers[0]
    } == {
        id(parameter) for parameter in encoder.backbone.conv_blocks_context[0].parameters()
    }
    _assert_parameter_layers_cover(encoder.backbone)


@pytest.mark.skipif(
    not STUNET_WEIGHTS.exists(),
    reason='released STU-Net-L checkpoint is not available',
)
def test_stunet_loads_real_weights_and_returns_plan_aligned_pyramid():
    encoder = build_stunet(_stunet_plan(), 1, {})
    load_stunet(encoder, STUNET_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 16, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 64, 16, 64, 64),
        (1, 128, 8, 32, 32),
        (1, 256, 4, 16, 16),
        (1, 512, 2, 8, 8),
        (1, 1024, 1, 4, 4),
        (1, 1024, 1, 2, 2),
    ]


@pytest.mark.skipif(
    not STUNET_WEIGHTS.exists(),
    reason='released STU-Net-L checkpoint is not available',
)
def test_stunet_inflates_its_pretrained_stem_for_multiple_channels():
    single_channel = build_stunet(_stunet_plan(), 1, {})
    load_stunet(single_channel, STUNET_WEIGHTS)
    multi_channel = build_stunet(_stunet_plan(), 3, {})
    load_stunet(multi_channel, STUNET_WEIGHTS)

    source = single_channel.backbone.conv_blocks_context[0][0].conv1.weight
    inflated = multi_channel.backbone.conv_blocks_context[0][0].conv1.weight

    assert tuple(inflated.shape) == (64, 3, 3, 3, 3)
    torch.testing.assert_close(inflated.sum(dim=1, keepdim=True), source)


def _segvol_plan():
    return dataclasses.replace(
        _plan(),
        output_channels=(32, 64, 128, 256, 384, 768),
        strides=((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    )


def test_segvol_requires_its_native_patch_geometry():
    config = prepare_segvol_adapter(Path('dummy.bin'), None, False)
    with pytest.raises(ValueError, match='native patch'):
        build_segvol_adapter(_segvol_plan(), 1, {**config, 'vit_patch_size': [8, 16, 16]})


def test_segvol_backbone_parameter_layers_cover():
    backbone = SegVolInteractiveBackbone(1, (4, 16, 16), gradient_checkpointing=False)
    _assert_parameter_layers_cover(backbone)


@pytest.mark.skipif(
    not SEGVOL_WEIGHTS.exists(),
    reason='released SegVol checkpoint is not available',
)
def test_segvol_vit_adapter_loads_real_weights_and_returns_plan_aligned_pyramid():
    config = prepare_segvol_adapter(SEGVOL_WEIGHTS, None, False)
    encoder = build_segvol_adapter(_segvol_plan(), 1, {**config, 'vit_patch_size': [4, 16, 16]})
    load_segvol_adapter(encoder, SEGVOL_WEIGHTS)
    encoder.eval()
    assert isinstance(encoder.backbone, SegVolInteractiveBackbone)

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 64, 64),
        (1, 64, 32, 32, 32),
        (1, 128, 16, 16, 16),
        (1, 256, 8, 8, 8),
        (1, 384, 4, 4, 4),
        (1, 768, 2, 2, 2),
    ]


@pytest.mark.skipif(
    not SEGVOL_WEIGHTS.exists(),
    reason='released SegVol checkpoint is not available',
)
def test_segvol_embedding_matches_the_released_rearrange_semantics():
    config = prepare_segvol_adapter(SEGVOL_WEIGHTS, None, False)
    encoder = build_segvol_adapter(_segvol_plan(), 1, {**config, 'vit_patch_size': [4, 16, 16]})
    load_segvol_adapter(encoder, SEGVOL_WEIGHTS)
    encoder.eval()

    full = torch.load(SEGVOL_WEIGHTS, map_location='cpu', weights_only=True)
    prefix = 'model.image_encoder.patch_embedding.'
    x = torch.randn(1, 1, 32, 256, 256)
    patches = einops.rearrange(
        x, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16
    )
    reference = F.linear(
        patches,
        full[f'{prefix}patch_embeddings.1.weight'],
        full[f'{prefix}patch_embeddings.1.bias'],
    ) + full[f'{prefix}position_embeddings']

    with torch.inference_mode():
        tokens, grid = encoder.backbone.model.embed_tokens(x)

    assert grid == (8, 16, 16)
    torch.testing.assert_close(tokens, reference, atol=1e-4, rtol=1e-5)


@pytest.mark.skipif(
    not (SEGVOL_WEIGHTS.exists() and SEGVOL_SSL_WEIGHTS.exists()),
    reason='released SegVol checkpoints are not available',
)
def test_segvol_loads_both_released_formats_and_they_differ():
    config = prepare_segvol_adapter(SEGVOL_WEIGHTS, None, False)

    sft = build_segvol_adapter(_segvol_plan(), 1, {**config, 'vit_patch_size': [4, 16, 16]})
    load_segvol_adapter(sft, SEGVOL_WEIGHTS)
    ssl = build_segvol_adapter(_segvol_plan(), 1, {**config, 'vit_patch_size': [4, 16, 16]})
    load_segvol_adapter(ssl, SEGVOL_SSL_WEIGHTS)

    sft_weight = sft.backbone.model.blocks[0].attn.qkv.weight
    ssl_weight = ssl.backbone.model.blocks[0].attn.qkv.weight
    assert not torch.equal(sft_weight, ssl_weight)


def _voco_plan():
    return make_plan(
        [48, 48, 96, 192, 384, 768],
        [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    )


def test_voco_requires_the_native_six_stage_plan():
    with pytest.raises(ValueError, match='requires plan channels'):
        build_voco(_plan(), 1, {})

    strided_stem = dataclasses.replace(
        _voco_plan(),
        strides=((2, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
    )
    with pytest.raises(ValueError, match='starting at stride'):
        build_voco(strided_stem, 1, {})


def test_voco_resamples_native_isotropic_levels_onto_anisotropic_plan_shapes():
    encoder = build_voco(_voco_plan(), 1, {})
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 8, 64, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 48, 8, 64, 64),
        (1, 48, 8, 32, 32),
        (1, 96, 4, 16, 16),
        (1, 192, 2, 8, 8),
        (1, 384, 1, 4, 4),
        (1, 768, 1, 2, 2),
    ]


def test_voco_encoder_is_entirely_pretrained():
    encoder = build_voco(_voco_plan(), 1, {})
    assert {id(parameter) for parameter in encoder.parameters()} == {
        id(parameter) for parameter in encoder.backbone.parameters()
    }
    layers = encoder.backbone.parameter_layers()
    assert len(layers) == 6
    assert {
        id(parameter) for parameter in layers[0]
    } == {
        id(parameter) for parameter in encoder.backbone.encoder1.parameters()
    }
    _assert_parameter_layers_cover(encoder.backbone)


def _truncated_plan(plan, num_stages: int):
    return dataclasses.replace(
        plan,
        output_channels=plan.output_channels[:num_stages],
        kernel_sizes=plan.kernel_sizes[:num_stages],
        strides=plan.strides[:num_stages],
        n_blocks_per_stage=plan.n_blocks_per_stage[:num_stages],
        n_conv_per_stage_decoder=plan.n_conv_per_stage_decoder[:num_stages - 1],
    )


def test_voco_five_stage_plan_consumes_the_shallowest_levels():
    encoder = build_voco(_truncated_plan(_voco_plan(), 5), 1, {})
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 8, 64, 64))

    assert [skip.shape[1] for skip in skips] == [48, 48, 96, 192, 384]


def test_stu_net_five_stage_plan_builds_the_shallowest_stages():
    encoder = build_stunet(_truncated_plan(_stunet_plan(), 5), 1, {})
    assert len(encoder.backbone.conv_blocks_context) == 5
    assert tuple(encoder.output_channels) == (64, 128, 256, 512, 1024)


def test_sat_pro_five_stage_plan_builds_the_shallowest_stages():
    encoder = build_sat_pro(_truncated_plan(_sat_pro_plan(), 5), 1, {})
    assert len(encoder.backbone.encoder.stages) == 5
    assert tuple(encoder.output_channels) == (128, 128, 256, 512, 1024)


@pytest.mark.skipif(
    not VOCO_WEIGHTS.exists(),
    reason='released VoCo-B checkpoint is not available',
)
def test_voco_loads_real_weights_strictly_and_returns_plan_aligned_pyramid():
    encoder = build_voco(_voco_plan(), 1, {})
    load_voco(encoder, VOCO_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 8, 64, 64))

    assert all(torch.isfinite(skip).all() for skip in skips)
    assert [skip.shape[1] for skip in skips] == [48, 48, 96, 192, 384, 768]


@pytest.mark.skipif(
    not VOCO_WEIGHTS.exists(),
    reason='released VoCo-B checkpoint is not available',
)
def test_voco_inflates_its_pretrained_input_convs_for_multiple_channels():
    single_channel = build_voco(_voco_plan(), 1, {})
    load_voco(single_channel, VOCO_WEIGHTS)
    multi_channel = build_voco(_voco_plan(), 3, {})
    load_voco(multi_channel, VOCO_WEIGHTS)

    source = single_channel.backbone.swinViT.patch_embed.proj.weight
    inflated = multi_channel.backbone.swinViT.patch_embed.proj.weight

    assert tuple(inflated.shape) == (48, 3, 2, 2, 2)
    torch.testing.assert_close(inflated.sum(dim=1, keepdim=True), source)


def _voco_l_plan():
    return dataclasses.replace(
        _voco_plan(),
        output_channels=(96, 96, 192, 384, 768, 1536),
    )


def test_voco_l_requires_its_own_native_channels():
    with pytest.raises(ValueError, match='feature size 96 requires plan channels'):
        build_voco_l(_voco_plan(), 1, {})


@pytest.mark.skipif(
    not VOCO_L_WEIGHTS.exists(),
    reason='released VoCo-L checkpoint is not available',
)
def test_voco_l_loads_real_weights_strictly_and_returns_plan_aligned_pyramid():
    encoder = build_voco_l(_voco_l_plan(), 1, {})
    load_voco_l(encoder, VOCO_L_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 8, 64, 64))

    assert all(torch.isfinite(skip).all() for skip in skips)
    assert [skip.shape[1] for skip in skips] == [96, 96, 192, 384, 768, 1536]


def _suprem_plan():
    return make_plan([64, 128, 256, 512], n_blocks_per_stage=[1, 3, 4, 6], n_conv_per_stage_decoder=[1] * 3)


def test_suprem_unet_requires_the_native_four_stage_plan():
    with pytest.raises(ValueError, match='requires plan channels'):
        build_suprem(_plan(), 1, {})

    narrow = dataclasses.replace(_suprem_plan(), output_channels=(32, 64, 128, 256))
    with pytest.raises(ValueError, match='requires plan channels'):
        build_suprem(narrow, 1, {})

    strided_stem = dataclasses.replace(
        _suprem_plan(),
        strides=((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    )
    with pytest.raises(ValueError, match='starting at stride'):
        build_suprem(strided_stem, 1, {})


def test_suprem_unet_pools_follow_anisotropic_plan_strides():
    anisotropic = dataclasses.replace(
        _suprem_plan(),
        strides=((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2)),
    )
    encoder = build_suprem(anisotropic, 1, {})
    skips = encoder(torch.randn(1, 1, 8, 32, 32))
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 64, 8, 32, 32),
        (1, 128, 8, 16, 16),
        (1, 256, 4, 8, 8),
        (1, 512, 2, 4, 4),
    ]


def test_suprem_unet_encoder_is_entirely_pretrained():
    encoder = build_suprem(_suprem_plan(), 1, {})
    assert {id(parameter) for parameter in encoder.parameters()} == {
        id(parameter) for parameter in encoder.backbone.parameters()
    }
    layers = encoder.backbone.parameter_layers()
    assert len(layers) == 4
    assert {
        id(parameter) for parameter in layers[0]
    } == {
        id(parameter) for parameter in encoder.backbone.down_tr64.parameters()
    }
    assert {
        id(parameter) for parameter in layers[-1]
    } == {
        id(parameter) for parameter in encoder.backbone.down_tr512.parameters()
    }
    _assert_parameter_layers_cover(encoder.backbone)


@pytest.mark.skipif(
    not SUPREM_WEIGHTS.exists(),
    reason='released SuPreM U-Net checkpoint is not available',
)
def test_suprem_unet_loads_real_weights_and_returns_plan_aligned_pyramid():
    encoder = build_suprem(_suprem_plan(), 1, {})
    load_suprem(encoder, SUPREM_WEIGHTS)

    with torch.no_grad():
        skips = encoder(torch.randn(2, 1, 16, 16, 32))

    assert [tuple(skip.shape) for skip in skips] == [
        (2, 64, 16, 16, 32),
        (2, 128, 8, 8, 16),
        (2, 256, 4, 4, 8),
        (2, 512, 2, 2, 4),
    ]


@pytest.mark.skipif(
    not UNIMISS_WEIGHTS.exists(),
    reason='released UniMiSS checkpoint is not available',
)
def test_unimiss_loads_real_weights_and_returns_plan_aligned_pyramid():
    encoder = build_unimiss(_plan(), 1, {})
    load_unimiss(encoder, UNIMISS_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 32, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 32, 64),
        (1, 64, 16, 16, 32),
        (1, 128, 8, 8, 16),
        (1, 256, 4, 4, 8),
        (1, 320, 2, 2, 4),
        (1, 320, 1, 1, 2),
    ]
    model = encoder.backbone.model
    assert not hasattr(model, 'norm_new')
    assert not hasattr(model, 'head_new')
    assert model.patch_embed3D0.conv.stride == (2, 2, 2)
    assert [
        patch_embed.proj.conv.stride
        for patch_embed in (
            model.patch_embed3D1,
            model.patch_embed3D2,
            model.patch_embed3D3,
            model.patch_embed3D4,
        )
    ] == [(2, 2, 2)] * 4


@pytest.mark.skipif(
    not BIOMEDCLIP_WEIGHTS.exists(),
    reason='released BiomedCLIP checkpoint is not available',
)
def test_biomedclip_loads_real_visual_weights_and_returns_plan_aligned_pyramid():
    config = {'gradient_checkpointing': False}
    encoder = build_biomedclip(_plan(), 1, config)
    load_biomedclip(encoder, BIOMEDCLIP_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 32, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 32, 64),
        (1, 64, 16, 16, 32),
        (1, 128, 8, 8, 16),
        (1, 256, 4, 4, 8),
        (1, 320, 2, 2, 4),
        (1, 320, 1, 1, 2),
    ]
    assert tuple(encoder.backbone.trunk.patch_embed.proj.weight.shape) == (768, 1, 16, 16, 16)
    assert tuple(encoder.backbone.projection.weight.shape) == (512, 768)
    _assert_parameter_layers_cover(encoder.backbone)


def test_biomedclip_vit_adapter_matches_the_original_full_trunk_path():
    original = BiomedCLIPFeatureBackbone(
        1,
        gradient_checkpointing=False,
    ).eval()
    interactive = BiomedCLIPInteractiveBackbone(
        1,
        (16, 16, 16),
        gradient_checkpointing=False,
    ).eval()
    interactive.trunk.load_state_dict(original.trunk.state_dict(), strict=True)
    x = torch.randn(1, 1, 32, 48, 64)

    with torch.inference_mode():
        expected = original(x)
        tokens, patch_shape, context = interactive.prepare_tokens(x)
        tokens = interactive.run_blocks(tokens, 0, 12, context)
        patch_tokens = interactive.normalize_patch_tokens(tokens)
        actual = original.projection(patch_tokens).transpose(1, 2).reshape(
            1,
            512,
            *patch_shape,
        )

    assert context == {}
    assert patch_shape == (2, 3, 4)
    assert next(iter(interactive.state_dict())).startswith('trunk.')
    torch.testing.assert_close(actual, expected)
    _assert_parameter_layers_cover(interactive)


@pytest.mark.skipif(
    not EVA02_WEIGHTS.exists(),
    reason='released EVA-02-L checkpoint is not available',
)
def test_eva02_loads_real_weights_and_returns_plan_aligned_pyramid():
    config = {'gradient_checkpointing': False}
    encoder = build_eva02(_plan(), 1, config)
    load_eva02(encoder, EVA02_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 32, 64))

    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 32, 64),
        (1, 64, 16, 16, 32),
        (1, 128, 8, 8, 16),
        (1, 256, 4, 4, 8),
        (1, 320, 2, 2, 4),
        (1, 320, 1, 1, 2),
    ]
    model = encoder.backbone.model
    assert tuple(model.patch_embed.proj.weight.shape) == (1024, 1, 16, 16, 16)
    assert not any(parameter.requires_grad for parameter in model.fc_norm.parameters())
    _assert_parameter_layers_cover(encoder.backbone)


def test_eva02_vit_adapter_matches_the_original_full_trunk_path():
    original = Eva02LargeFeatureBackbone(
        1,
        gradient_checkpointing=False,
    ).eval()
    interactive = Eva02LargeInteractiveBackbone(
        1,
        (16, 16, 16),
        gradient_checkpointing=False,
    ).eval()
    interactive.model.load_state_dict(original.model.state_dict(), strict=True)
    x = torch.randn(1, 1, 32, 48, 64)

    with torch.inference_mode():
        expected = original(x)
        tokens, patch_shape, context = interactive.prepare_tokens(x)
        tokens = interactive.run_blocks(tokens, 0, 24, context)
        patch_tokens = interactive.normalize_patch_tokens(tokens)
        actual = patch_tokens.transpose(1, 2).reshape(
            1,
            1024,
            *patch_shape,
        )

    assert context.keys() == {'rope'}
    assert patch_shape == (2, 3, 4)
    assert next(iter(interactive.state_dict())).startswith('model.')
    torch.testing.assert_close(actual, expected)
    _assert_parameter_layers_cover(interactive)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not THREE_DINO_WEIGHTS.exists(),
    reason='3DINO CUDA smoke test requires the released checkpoint',
)
def test_3dino_loads_real_weights_and_returns_plan_aligned_pyramid():
    config = three_dino.prepare_config(THREE_DINO_WEIGHTS, None, False)
    encoder = three_dino.build_encoder(_plan(), 1, config).cuda().eval()
    three_dino.load_pretrained(encoder, THREE_DINO_WEIGHTS)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        skips = encoder(torch.randn(1, 1, 32, 32, 64, device='cuda'))

    assert config == {}
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 32, 32, 32, 64),
        (1, 64, 16, 16, 32),
        (1, 128, 8, 8, 16),
        (1, 256, 4, 4, 8),
        (1, 320, 2, 2, 4),
        (1, 320, 1, 1, 2),
    ]
    _assert_parameter_layers_cover(encoder.backbone)


class _TinyThreeDinoPatchEmbed(nn.Module):
    def __init__(self, input_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = (16, 16, 16)
        self.proj = nn.Conv3d(
            input_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _TinyThreeDino(nn.Module):
    embed_dim = 4
    n_blocks = 24
    chunked_blocks = False

    def __init__(self, input_channels: int):
        super().__init__()
        self.patch_embed = _TinyThreeDinoPatchEmbed(input_channels, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 2, self.embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, self.embed_dim))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(self.n_blocks)])
        self.norm = nn.LayerNorm(self.embed_dim)

    def interpolate_pos_encoding(self, tokens, height, width, depth):
        self.seen_position_input_shape = (height, width, depth)
        return torch.arange(
            tokens.shape[1] * tokens.shape[2],
            device=tokens.device,
            dtype=tokens.dtype,
        ).reshape(1, tokens.shape[1], tokens.shape[2])

    def forward_features(self, x):
        self.seen_shape = tuple(x.shape)
        feature = self.patch_embed.proj(x)
        tokens = feature.permute(0, 2, 3, 4, 1).flatten(1, 3)
        return {'x_norm_patchtokens': tokens}


def test_3dino_uses_native_hwd_order_and_returns_nnunet_dhw(monkeypatch):
    model = _TinyThreeDino(1)
    monkeypatch.setattr(three_dino, 'build_three_dino', lambda input_channels: model)
    backbone = three_dino.ThreeDinoFeatureBackbone(1)
    x = torch.randn(1, 1, 32, 48, 64)

    features = backbone(x)

    native_input = x.permute(0, 1, 3, 4, 2)
    native_feature = model.patch_embed.proj(native_input)
    assert model.seen_shape == (1, 1, 48, 64, 32)
    assert tuple(features.shape) == (1, model.embed_dim, 2, 3, 4)
    torch.testing.assert_close(features, native_feature.permute(0, 1, 4, 2, 3))


def test_3dino_inflates_pretrained_patch_embed_and_freezes_unused_mask_token(monkeypatch):
    source = _TinyThreeDino(1).state_dict()
    monkeypatch.setattr(
        three_dino,
        'build_three_dino',
        lambda input_channels: _TinyThreeDino(input_channels),
    )
    monkeypatch.setattr(three_dino, '_load_state_dict', lambda weights: source)
    config = {}
    encoder = three_dino.build_encoder(_plan(), 3, config)

    three_dino.load_pretrained(encoder, Path('unused.pth'))

    inflated = encoder.backbone.model.patch_embed.proj.weight
    torch.testing.assert_close(
        inflated.sum(dim=1, keepdim=True),
        source['patch_embed.proj.weight'],
    )
    assert not encoder.backbone.model.mask_token.requires_grad


def test_3dino_vit_adapter_preserves_native_positions_before_dhw_reordering(monkeypatch):
    model = _TinyThreeDino(1)
    monkeypatch.setattr(
        three_dino_vit_adapter,
        'build_three_dino',
        lambda input_channels, *, drop_path_rate=0.0: model,
    )
    backbone = three_dino_vit_adapter.ThreeDinoInteractiveBackbone(
        1,
        (16, 16, 16),
        gradient_checkpointing=False,
    )
    x = torch.randn(1, 1, 32, 48, 64)

    tokens, patch_shape, context = backbone.prepare_tokens(x)

    native_input = x.permute(0, 1, 3, 4, 2)
    native_patch_tokens = model.patch_embed(native_input)
    native_tokens = torch.cat((model.cls_token, native_patch_tokens), dim=1)
    native_tokens = native_tokens + model.interpolate_pos_encoding(
        native_tokens,
        *native_input.shape[2:],
    )
    expected = torch.cat(
        (
            native_tokens[:, :1],
            native_tokens[:, 1:].reshape(1, 3, 4, 2, 4).permute(0, 3, 1, 2, 4).flatten(1, 3),
        ),
        dim=1,
    )
    assert patch_shape == (2, 3, 4)
    assert context is None
    assert model.seen_position_input_shape == (48, 64, 32)
    torch.testing.assert_close(tokens, expected)


def test_3dino_vit_adapter_reduces_depth_and_repeats_input_channels(monkeypatch):
    source = _TinyThreeDino(1).state_dict()
    monkeypatch.setattr(
        three_dino_vit_adapter,
        'build_three_dino',
        lambda input_channels, *, drop_path_rate=0.0: _TinyThreeDino(input_channels),
    )
    monkeypatch.setattr(
        three_dino_vit_adapter,
        '_load_state_dict',
        lambda weights: source,
    )
    config = three_dino_vit_adapter.prepare_config(Path('unused.pth'), None, False)
    config['vit_patch_size'] = [4, 16, 16]
    encoder = three_dino_vit_adapter.build_encoder(_plan(), 3, config)

    three_dino_vit_adapter.load_pretrained(encoder, Path('unused.pth'))

    source_dhw = repeat_single_channel_weight(
        source['patch_embed.proj.weight'],
        3,
    ).permute(0, 1, 4, 2, 3)
    expected = adapt_patch_embed_weight(source_dhw, (4, 16, 16)).permute(0, 1, 3, 4, 2)
    torch.testing.assert_close(
        encoder.backbone.model.patch_embed.proj.weight,
        expected,
    )
    assert not encoder.backbone.model.mask_token.requires_grad


@pytest.mark.parametrize('drop_path_rate', [None, 0.0, 0.1])
def test_3dino_vit_adapter_uses_native_depth_linear_drop_path(monkeypatch, drop_path_rate):
    import pumit.downstream.cls.backbones.three_dino as native_three_dino
    from pumit.downstream.cls.backbones._3dino.models.vision_transformer import DinoVisionTransformer3d

    def small_vit(**kwargs):
        kwargs['img_size'] = 16
        return DinoVisionTransformer3d(embed_dim=32, depth=24, num_heads=4, **kwargs)

    monkeypatch.setattr(native_three_dino, 'vit_large_3d', small_vit)
    monkeypatch.setattr(
        three_dino_vit_adapter,
        'build_vit_adapter_encoder',
        lambda backbone, *args, **kwargs: backbone,
    )
    config = three_dino_vit_adapter.prepare_config(Path('unused.pth'), None, False)
    config['vit_patch_size'] = [16, 16, 16]
    if drop_path_rate is None:
        del config['architecture']
    else:
        config['architecture']['drop_path_rate'] = drop_path_rate

    backbone = three_dino_vit_adapter.build_encoder(_plan(), 1, config)
    blocks = three_dino_vit_adapter._transformer_blocks(backbone.model)
    expected = torch.linspace(0, drop_path_rate or 0.0, 24, dtype=torch.float64).tolist()
    assert [block.sample_drop_ratio for block in blocks] == expected
    assert blocks[-1].sample_drop_ratio == (drop_path_rate or 0.0)
    for block, probability in zip(blocks, expected, strict=True):
        for drop_path in (block.drop_path1, block.drop_path2):
            if probability == 0:
                assert isinstance(drop_path, nn.Identity)
            else:
                assert drop_path.drop_prob == probability


def test_pumit_final_layer_simple_fpn_probe_drops_the_stem_and_loads_the_ema_vit(tmp_path, cpu_attention):
    vit_config = dataclasses.replace(_PUMIT_VIT_CONFIG, num_hidden_layers=4)
    checkpoint_path = _pumit_checkpoint(tmp_path, vit_config)
    config = pumit_simple_fpn.prepare_config(checkpoint_path, 'ucpt', False)
    config['feature_layers'] = [vit_config.num_hidden_layers]
    config['vit_patch_size'] = [16, 16, 16]
    config['with_high_resolution_stem'] = False
    encoder = pumit_simple_fpn.build_encoder(_plan(), 1, config)
    pumit_simple_fpn.load_pretrained(encoder, checkpoint_path)

    assert isinstance(encoder, SimpleFPNEncoder3D)
    assert encoder.high_resolution_stem is None
    assert tuple(encoder.output_channels) == (128, 256, 320, 320)
    skips = encoder(torch.randn(1, 1, 32, 32, 32))
    assert [tuple(skip.shape) for skip in skips] == [
        (1, 128, 8, 8, 8),
        (1, 256, 4, 4, 4),
        (1, 320, 2, 2, 2),
        (1, 320, 1, 1, 1),
    ]
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
    torch.testing.assert_close(
        encoder.backbone.vit.layer[0].attention.q_proj.weight,
        source['teacher_vit.layer.0.attention.q_proj.weight'],
    )
    _assert_parameter_layers_cover(encoder.backbone)


def test_dinov3_and_random_simple_fpn_share_the_final_layer_pyramid():
    config = random_simple_fpn.prepare_config(None, None, False)
    config['feature_layers'] = [24]
    config['vit_patch_size'] = [16, 16, 16]
    encoder = random_simple_fpn.build_encoder(_plan(), 1, config)

    assert random_simple_fpn.build_encoder is dinov3.build_encoder
    assert isinstance(encoder, SimpleFPNEncoder3D)
    assert encoder.high_resolution_stem is not None
    random_simple_fpn.load_pretrained(encoder, None)
    with pytest.raises(ValueError, match='takes no weights'):
        random_simple_fpn.load_pretrained(encoder, Path('weights.pt'))
    with pytest.raises(ValueError, match='feature_layers must be'):
        dinov3.build_encoder(_plan(), 1, {**config, 'feature_layers': [12, 24]})


_PROBE_CONFIG = {'with_high_resolution_stem': False, 'pyramid_branch': 'resample'}


@pytest.mark.parametrize(
    'builder',
    [pumit_simple_fpn.build_encoder, dinov3.build_encoder, random_simple_fpn.build_encoder],
)
@pytest.mark.parametrize('num_stages', [5, 6])
def test_flat_vit_probe_matches_the_plan_level_count(builder, num_stages, cpu_attention):
    plan = make_plan([4, 8, 16, 24, 32, 32][:num_stages])
    config = {
        'architecture': dataclasses.asdict(
            ViTConfig(
                hidden_size=32, num_hidden_layers=24, num_attention_heads=4,
                intermediate_size=64, pos_embed_rescale=None,
            ),
        ),
        'feature_layers': list(select_evenly_spaced_layers(24, num_stages - 2)),
        'vit_patch_size': [16, 16, 16],
        **_PROBE_CONFIG,
    }
    encoder = builder(plan, 1, config).eval()
    encoder.backbone.requires_grad_(False)
    assert encoder.high_resolution_stem is None
    assert encoder.backbone.feature_layers == ((8, 16, 24) if num_stages == 5 else (6, 12, 18, 24))
    skips = encoder(torch.randn(1, 1, 32, 32, 64))
    shapes = compute_stage_shapes((32, 32, 64), plan.cumulative_strides)
    assert [tuple(skip.shape) for skip in skips] == [
        (1, channels, *shape)
        for channels, shape in zip(plan.output_channels[2:], shapes[2:], strict=True)
    ]
    sum(skip.square().mean() for skip in skips).backward()
    assert all(parameter.grad is None for parameter in encoder.backbone.parameters())
    assert all(parameter.grad is not None for parameter in encoder.simple_fpn.parameters())


@pytest.mark.parametrize(
    ('build', 'native_channels', 'input_shape'),
    [
        (build_stunet, (64, 128, 256, 512, 1024, 1024), (32, 64, 64)),
        (build_sat_pro, (128, 128, 256, 512, 1024, 1536), (32, 64, 64)),
        (build_unimiss_plus, (32, 64, 128, 256, 320, 320), (32, 32, 64)),
        (build_voco, (48, 48, 96, 192, 384, 768), (32, 64, 64)),
    ],
)
def test_pyramid_wrappers_project_p2_to_p5_onto_the_plan_schedule_without_the_stem(
    build, native_channels, input_shape,
):
    plan = _plan()
    encoder = build(plan, 1, _PROBE_CONFIG)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, *input_shape))

    inner = native_encoder(encoder)
    assert inner is not encoder
    assert tuple(inner.output_channels) == native_channels
    assert tuple(encoder.output_channels) == (128, 256, 320, 320)
    assert encoder.high_resolution_stem is None
    assert [branch.block_stages for branch in encoder.simple_fpn] == [(2,), (3,), (4,), (5,)]
    stage_shapes = compute_stage_shapes(input_shape, plan.cumulative_strides)
    assert [tuple(skip.shape) for skip in skips] == [
        (1, channels, *shape) for channels, shape in zip((128, 256, 320, 320), stage_shapes[2:])
    ]
    # The trunk is the whole native encoder; only the branches are new parameters.
    assert {id(p) for p in encoder.backbone.parameters()} == {id(p) for p in inner.parameters()}


@pytest.mark.skipif(
    not STUNET_WEIGHTS.exists(),
    reason='released STU-Net-L checkpoint is not available',
)
def test_stunet_loads_its_weights_through_the_projected_readout():
    encoder = build_stunet(_plan(), 1, _PROBE_CONFIG)
    load_stunet(encoder, STUNET_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 64, 64))

    assert all(torch.isfinite(skip).all() for skip in skips)
    assert [skip.shape[1] for skip in skips] == [128, 256, 320, 320]


@pytest.mark.skipif(
    not VOCO_WEIGHTS.exists(),
    reason='released VoCo-B checkpoint is not available',
)
def test_voco_loads_its_weights_through_the_projected_readout():
    encoder = build_voco(_plan(), 1, _PROBE_CONFIG)
    load_voco(encoder, VOCO_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 64, 64))

    assert all(torch.isfinite(skip).all() for skip in skips)
    assert [skip.shape[1] for skip in skips] == [128, 256, 320, 320]



@pytest.mark.parametrize(
    ('module', 'weights', 'depth'),
    [(three_dino, THREE_DINO_WEIGHTS, 24), (sam_med3d, SAM_MED3D_WEIGHTS, 12)],
)
def test_flat_baselines_default_to_four_evenly_spaced_depths(module, weights, depth):
    config = module.prepare_config(weights, None, False)

    assert config['feature_layers'] == list(select_evenly_spaced_layers(depth))


def test_biomedclip_defaults_to_four_evenly_spaced_depths():
    import pumit.downstream.seg.backbones.biomedclip as biomedclip

    config = biomedclip.prepare_config(BIOMEDCLIP_WEIGHTS, None, False)

    assert config['feature_layers'] == [3, 6, 9, 12]


class _TinySAMMed3DCountingBlock(_TinySAMMed3DBlock):
    """Adds one per block so the depth a level was read at is visible in its values."""

    def forward(self, x):
        return x + 1


class _TinySAMMed3DCounting(_TinySAMMed3D):
    def __init__(self, input_channels: int):
        super().__init__(input_channels)
        self.blocks = nn.ModuleList(_TinySAMMed3DCountingBlock(15) for _ in range(12))


def test_sam_med3d_multi_depth_readout_reads_four_blocks_and_drops_the_neck(monkeypatch):
    source = _TinySAMMed3DCounting(1).state_dict()
    monkeypatch.setattr(sam_med3d, 'build_sam_med3d', lambda input_channels: _TinySAMMed3DCounting(input_channels))
    monkeypatch.setattr(sam_med3d, '_load_state_dict', lambda weights: source)
    plan = _plan()
    config = {'vit_patch_size': [16, 16, 16], 'feature_layers': [3, 6, 9, 12], **_PROBE_CONFIG}
    encoder = sam_med3d.build_encoder(plan, 1, config)
    sam_med3d.load_pretrained(encoder, Path('unused.pth'))
    encoder.eval()

    x = torch.randn(1, 1, 32, 32, 64)
    with torch.inference_mode():
        features = encoder.backbone(x)
        skips = encoder(x)

    assert isinstance(encoder, PlanAlignedPyramidEncoder3D)
    assert not hasattr(encoder.backbone.model, 'neck')
    assert encoder.backbone.feature_channels == (4, 4, 4, 4)
    assert [tuple(feature.shape) for feature in features] == [(1, 4, 2, 2, 4)] * 4
    # Shallow to fine: P2 reads block 3, each following level three blocks deeper.
    for level in range(1, 4):
        torch.testing.assert_close(features[level], features[0] + 3 * level)
    assert [skip.shape[1] for skip in skips] == [128, 256, 320, 320]
    assert [branch.block_stages for branch in encoder.simple_fpn] == [(3, 2), (3,), (4,), (5,)]
    with pytest.raises(ValueError, match='feature_layers must be'):
        sam_med3d.build_encoder(plan, 1, {**config, 'feature_layers': [12]})


def _record_block_outputs(blocks, depths: tuple[int, ...]) -> tuple[list, list]:
    recorded = []
    handles = [
        blocks[depth - 1].register_forward_hook(lambda module, inputs, output: recorded.append(output))
        for depth in depths
    ]
    return recorded, handles


def test_3dino_multi_depth_readout_takes_the_real_intermediate_layers_through_the_final_norm():
    torch.manual_seed(0)
    legacy = three_dino.ThreeDinoFeatureBackbone(1).eval()
    multi = three_dino.ThreeDinoFeatureBackbone(1, (6, 12, 18, 24)).eval()
    multi.model.load_state_dict(legacy.model.state_dict())
    x = torch.randn(1, 1, 32, 48, 64)
    blocks = three_dino._transformer_blocks(multi.model)
    recorded, handles = _record_block_outputs(blocks, (6, 12, 18, 24))

    with torch.inference_mode():
        final = legacy(x)
        features = multi(x)
        expected = [
            einops.rearrange(multi.model.norm(hidden)[:, 1:], 'b (h w d) c -> b c d h w', h=3, w=4, d=2)
            for hidden in recorded
        ]
    for handle in handles:
        handle.remove()

    assert multi.feature_channels == (multi.model.embed_dim,) * 4
    assert [tuple(feature.shape) for feature in features] == [(1, multi.model.embed_dim, 2, 3, 4)] * 4
    torch.testing.assert_close(features[3], final)
    for feature, hidden in zip(features, expected, strict=True):
        torch.testing.assert_close(feature, hidden)
    _assert_parameter_layers_cover(multi)


def test_biomedclip_multi_depth_readout_reads_four_normed_trunk_depths_without_the_projection():
    torch.manual_seed(0)
    legacy = BiomedCLIPFeatureBackbone(1, gradient_checkpointing=False).eval()
    multi = BiomedCLIPFeatureBackbone(1, gradient_checkpointing=False, feature_layers=(3, 6, 9, 12)).eval()
    multi.trunk.load_state_dict(legacy.trunk.state_dict())
    x = torch.randn(1, 1, 32, 32, 64)
    recorded, handles = _record_block_outputs(multi.trunk.blocks, (3, 6, 9, 12))

    with torch.inference_mode():
        projected = legacy(x)
        features = multi(x)
        expected = [
            einops.rearrange(multi.trunk.norm(hidden)[:, 1:], 'b (d h w) c -> b c d h w', d=2, h=2, w=4)
            for hidden in recorded
        ]
        # The deepest level is the legacy map before the CLIP projection.
        reprojected = einops.rearrange(
            legacy.projection(einops.rearrange(features[3], 'b c d h w -> b (d h w) c')),
            'b (d h w) c -> b c d h w', d=2, h=2, w=4,
        )
    for handle in handles:
        handle.remove()

    assert not hasattr(multi, 'projection')
    assert multi.feature_channels == (768,) * 4
    assert [tuple(feature.shape) for feature in features] == [(1, 768, 2, 2, 4)] * 4
    for feature, hidden in zip(features, expected, strict=True):
        torch.testing.assert_close(feature, hidden)
    torch.testing.assert_close(reprojected, projected)
    _assert_parameter_layers_cover(multi)


@pytest.mark.skipif(
    not BIOMEDCLIP_WEIGHTS.exists(),
    reason='released BiomedCLIP checkpoint is not available',
)
def test_biomedclip_multi_depth_readout_loads_the_released_trunk():
    import pumit.downstream.seg.backbones.biomedclip as biomedclip

    config = {**biomedclip.prepare_config(BIOMEDCLIP_WEIGHTS, None, False), **_PROBE_CONFIG}
    encoder = build_biomedclip(_plan(), 1, config)
    load_biomedclip(encoder, BIOMEDCLIP_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 32, 64))

    assert isinstance(encoder, PlanAlignedPyramidEncoder3D)
    assert [skip.shape[1] for skip in skips] == [128, 256, 320, 320]
    assert all(torch.isfinite(skip).all() for skip in skips)


def test_eva02_defaults_to_four_evenly_spaced_depths_and_reads_them_through_the_final_norm():
    import pumit.downstream.seg.backbones.eva02 as eva02

    assert eva02.prepare_config(EVA02_WEIGHTS, None, False)['feature_layers'] == [6, 12, 18, 24]
    torch.manual_seed(0)
    legacy = Eva02LargeFeatureBackbone(1, gradient_checkpointing=False).eval()
    multi = Eva02LargeFeatureBackbone(1, gradient_checkpointing=False, feature_layers=(6, 12, 18, 24)).eval()
    multi.model.load_state_dict(legacy.model.state_dict())
    x = torch.randn(1, 1, 32, 48, 64)
    recorded, handles = _record_block_outputs(multi.model.blocks, (6, 12, 18, 24))

    with torch.inference_mode():
        final = legacy(x)
        features = multi(x)
        expected = [
            einops.rearrange(multi.model.norm(hidden)[:, 1:], 'b (d h w) c -> b c d h w', d=2, h=3, w=4)
            for hidden in recorded
        ]
    for handle in handles:
        handle.remove()

    assert multi.feature_channels == (1024,) * 4
    assert [tuple(feature.shape) for feature in features] == [(1, 1024, 2, 3, 4)] * 4
    torch.testing.assert_close(features[3], final)
    for feature, hidden in zip(features, expected, strict=True):
        torch.testing.assert_close(feature, hidden)
    _assert_parameter_layers_cover(multi)


@pytest.mark.skipif(
    not EVA02_WEIGHTS.exists(),
    reason='released EVA-02-L checkpoint is not available',
)
def test_eva02_multi_depth_readout_loads_the_released_trunk():
    import pumit.downstream.seg.backbones.eva02 as eva02

    config = {**eva02.prepare_config(EVA02_WEIGHTS, None, False), **_PROBE_CONFIG}
    encoder = build_eva02(_plan(), 1, config)
    load_eva02(encoder, EVA02_WEIGHTS)
    encoder.eval()

    with torch.inference_mode():
        skips = encoder(torch.randn(1, 1, 32, 32, 64))

    assert isinstance(encoder, PlanAlignedPyramidEncoder3D)
    assert [skip.shape[1] for skip in skips] == [128, 256, 320, 320]
    assert all(torch.isfinite(skip).all() for skip in skips)
