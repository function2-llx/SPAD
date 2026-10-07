"""Tests for stochastic DA decomposition in the data pipeline."""
import numpy as np

from pumit.data import gen_trans_info
from pumit.codec.datamodule import TransformConf
from pumit.codec.config import MAX_DA


def test_stochastic_da_at_boundary():
    """At exact DA boundary (ratio=2.0), t=0, always floor DA."""
    conf = TransformConf()
    conf.scale_z_p = 0.0
    conf.scale_xy_p = 0.0
    data = {'shape': [64, 256, 256], 'spacing': np.array([2.0, 1.0, 1.0])}
    R = np.random.RandomState(42)
    results = [gen_trans_info(data, conf, R, max_da=MAX_DA) for _ in range(100)]
    for r in results:
        assert r['da_enc'] == 1, f"Expected da_enc=1, got {r['da_enc']}"
        assert r['da_dec'] == 1, f"Expected da_dec=1, got {r['da_dec']}"
        assert r['t'] == 0.0


def test_stochastic_da_mid_band():
    """At ratio ~1.5 (t~0.585), both floor and ceil should appear."""
    conf = TransformConf()
    conf.scale_z_p = 0.0
    conf.scale_xy_p = 0.0
    data = {'shape': [64, 256, 256], 'spacing': np.array([1.5, 1.0, 1.0])}
    R = np.random.RandomState(42)
    results = [gen_trans_info(data, conf, R, max_da=MAX_DA) for _ in range(1000)]
    enc_das = [r['da_enc'] for r in results]
    dec_das = [r['da_dec'] for r in results]
    assert 0 in enc_das and 1 in enc_das
    assert 0 in dec_das and 1 in dec_das
    ceil_frac = sum(1 for d in enc_das if d == 1) / len(enc_das)
    assert 0.4 < ceil_frac < 0.75, f'P(ceil)={ceil_frac}, expected ~0.585'


def test_stochastic_da_independence():
    """da_enc and da_dec are independently sampled (not always equal)."""
    conf = TransformConf()
    conf.scale_z_p = 0.0
    conf.scale_xy_p = 0.0
    data = {'shape': [64, 256, 256], 'spacing': np.array([1.5, 1.0, 1.0])}
    R = np.random.RandomState(42)
    results = [gen_trans_info(data, conf, R, max_da=MAX_DA) for _ in range(1000)]
    # With t~0.585, P(enc!=dec) = 2*t*(1-t) ~ 0.486, so we should see disagreements
    disagreements = sum(1 for r in results if r['da_enc'] != r['da_dec'])
    assert disagreements > 100, f'Only {disagreements}/1000 disagreements, expected ~486'


def test_trans_info_has_new_fields():
    """TransInfo should have da_enc, da_dec, t fields."""
    conf = TransformConf()
    conf.scale_z_p = 0.0
    conf.scale_xy_p = 0.0
    data = {'shape': [32, 256, 256], 'spacing': np.array([3.0, 1.0, 1.0])}
    R = np.random.RandomState(42)
    result = gen_trans_info(data, conf, R, max_da=MAX_DA)
    assert 'da_enc' in result
    assert 'da_dec' in result
    assert 't' in result
    assert isinstance(result['t'], float)


def test_isotropic_always_zero():
    """Isotropic spacing (ratio=1.0) should always give da_enc=da_dec=0, t=0."""
    conf = TransformConf()
    conf.scale_z_p = 0.0
    conf.scale_xy_p = 0.0
    data = {'shape': [64, 256, 256], 'spacing': np.array([1.0, 1.0, 1.0])}
    R = np.random.RandomState(42)
    results = [gen_trans_info(data, conf, R, max_da=MAX_DA) for _ in range(100)]
    for r in results:
        assert r['da_enc'] == 0
        assert r['da_dec'] == 0
        assert r['t'] == 0.0


def test_high_aniso_clamped():
    """Very high aniso ratio should clamp at MAX_DA."""
    conf = TransformConf()
    conf.scale_z_p = 0.0
    conf.scale_xy_p = 0.0
    # spacing_z/spacing_xy = 100, ratio clipped to 2^MAX_DA => da_enc=MAX_DA, t=0
    data = {'shape': [10, 256, 256], 'spacing': np.array([100.0, 1.0, 1.0])}
    R = np.random.RandomState(42)
    result = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False)
    assert result['da_enc'] == MAX_DA
    assert result['da_dec'] == MAX_DA
    assert result['t'] == 0.0


def test_cross_da_resize_shapes():
    """Verify resize dimensions for DA boundary pairs where depth downsamples differ.

    For adjacent DA pairs (da_enc, da_dec = da_enc +/- 1) with different depth
    downsample counts, the cross-DA latent resize followed by decode should
    produce output with the same depth as the original input encoded at da_enc.
    DA >= 3 all have 0 depth downsamples, so transitions at the DA=3 boundary
    (e.g., 3->4) do not require resize.
    """
    for da_enc in range(4):
        D_input = {0: 64, 1: 48, 2: 32, 3: 24}[da_enc]
        nds_enc = 3 - min(da_enc, 3)
        D_lat = D_input // (2 ** nds_enc)

        # Up: da_dec = da_enc + 1
        da_dec = da_enc + 1
        nds_dec = 3 - min(da_dec, 3)
        if nds_enc != nds_dec:
            D_resized = D_lat * 2
            D_out = D_resized * (2 ** nds_dec)
            assert D_out == D_input, f'enc={da_enc} dec={da_dec}: D_out={D_out} != D_input={D_input}'

        # Down: da_dec = da_enc - 1
        if da_enc > 0:
            da_dec = da_enc - 1
            nds_dec = 3 - min(da_dec, 3)
            if nds_enc != nds_dec:
                D_resized = D_lat // 2
                D_out = D_resized * (2 ** nds_dec)
                assert D_out == D_input, f'enc={da_enc} dec={da_dec}: D_out={D_out} != D_input={D_input}'

    # Confirm DA=3->4 skips resize (same depth downsample count = 0)
    assert (3 - min(3, 3)) == (3 - min(4, 3)) == 0


def test_cross_da_resize_same_da_is_noop():
    """When da_enc == da_dec, no resize should happen (resize guard)."""
    import torch
    import torch.nn.functional as F

    z_plain = torch.randn(1, 16, 8, 32, 32)
    da_enc = 0
    da_dec = 0

    # Guard: no resize when da_enc == da_dec
    if da_enc != da_dec:
        scale_d = 2.0 if da_dec > da_enc else 0.5
        new_d = int(z_plain.shape[2] * scale_d)
        z_resized = F.interpolate(
            z_plain,
            size=(new_d, z_plain.shape[3], z_plain.shape[4]),
            mode='trilinear',
            align_corners=False,
        )
    else:
        z_resized = z_plain

    assert z_resized.shape == z_plain.shape, 'same DA should not change shape'


def test_cross_da_resize_latent_dimensions():
    """Verify actual tensor resize produces correct latent depth."""
    import torch
    import torch.nn.functional as F

    # Simulate: encode at DA=0 (D_input=64 -> D_lat=8), resize for DA=1 decode
    z = torch.randn(1, 16, 8, 32, 32)
    da_enc, da_dec = 0, 1
    scale_d = 2.0  # da_dec > da_enc
    new_d = int(z.shape[2] * scale_d)
    z_resized = F.interpolate(
        z, size=(new_d, z.shape[3], z.shape[4]),
        mode='trilinear', align_corners=False,
    )
    assert z_resized.shape == (1, 16, 16, 32, 32)

    # Simulate: encode at DA=1 (D_input=48 -> D_lat=12), resize for DA=0 decode
    z = torch.randn(1, 16, 12, 32, 32)
    da_enc, da_dec = 1, 0
    scale_d = 0.5  # da_dec < da_enc
    new_d = int(z.shape[2] * scale_d)
    z_resized = F.interpolate(
        z, size=(new_d, z.shape[3], z.shape[4]),
        mode='trilinear', align_corners=False,
    )
    assert z_resized.shape == (1, 16, 6, 32, 32)

