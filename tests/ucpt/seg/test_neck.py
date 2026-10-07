import torch

from pumit.ucpt.seg.neck import SPADNeck


def _vit_map(d=8, h=24, w=24):
    return torch.randn(1, 1024, d, h, w)


def test_three_levels_shapes_da0():
    neck = SPADNeck(in_channels=1024, hidden_size=256)
    x = _vit_map(d=8)
    levels = neck(x, 0)
    l4, l8, l16 = levels['1/4'], levels['1/8'], levels['1/16']
    assert l16.shape == (1, 256, 8, 24, 24)
    assert l8.shape == (1, 256, 16, 48, 48)
    assert l4.shape == (1, 256, 32, 96, 96)


def test_depth_frozen_high_da():
    neck = SPADNeck(in_channels=1024, hidden_size=256)
    x = _vit_map(d=12)
    levels = neck(x, 3)
    assert levels['1/16'].shape[2] == 12
    assert levels['1/8'].shape[2] == 24
    assert levels['1/4'].shape[2] == 24


def test_consistent_depth_at_1_4():
    neck = SPADNeck(in_channels=1024, hidden_size=256)
    x = _vit_map(d=8)
    levels = neck(x, 0)
    assert levels['1/4'].shape[2] == 32


def test_pointwise_projections_use_channels_last_3d_weight_layout():
    neck = SPADNeck(in_channels=32, hidden_size=8)
    expected_strides = {
        'proj40_1': (8, 1, 8, 8, 8),
        'proj20_1': (16, 1, 16, 16, 16),
        'proj10_1': (32, 1, 32, 32, 32),
    }
    for name, stride in expected_strides.items():
        assert getattr(neck, name).weight.stride() == stride

    restored = SPADNeck(in_channels=32, hidden_size=8)
    restored.load_state_dict(neck.state_dict())
    for name in expected_strides:
        assert getattr(restored, name).weight.stride() == getattr(neck, name).weight.stride()
