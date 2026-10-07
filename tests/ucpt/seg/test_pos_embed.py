import importlib.util
import math
from pathlib import Path

import pytest
import torch

from pumit.ucpt.seg.pos_embed import sine_pos_embed_3d

SAM3_PE_PATH = Path('refs/repos/sam3/sam3/model/position_encoding.py')


def _load_sam3_position_encoding():
    """Load SAM 3's position_encoding.py standalone (the sam3 package __init__ pulls heavy deps)."""
    spec = importlib.util.spec_from_file_location('sam3_pe', SAM3_PE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shape_matches_grid():
    pe = sine_pos_embed_3d(d=8, h=24, w=24, dim=256, device=torch.device('cpu'))
    assert pe.shape == (8 * 24 * 24, 256)


def test_deterministic():
    a = sine_pos_embed_3d(4, 6, 6, 256, torch.device('cpu'))
    b = sine_pos_embed_3d(4, 6, 6, 256, torch.device('cpu'))
    assert torch.equal(a, b)


def test_dim_must_be_divisible_by_4():
    for dim in (4, 250):
        with pytest.raises(ValueError):
            sine_pos_embed_3d(4, 6, 6, dim, torch.device('cpu'))


def test_depth_temperature_must_differ():
    with pytest.raises(ValueError):
        sine_pos_embed_3d(4, 6, 6, 256, torch.device('cpu'),
                          temperature=10000.0, depth_temperature=10000.0)


def test_fp32_rounding_equal_families_rejected():
    with pytest.raises(ValueError):
        sine_pos_embed_3d(4, 6, 6, 256, torch.device('cpu'),
                          temperature=10000.0, depth_temperature=10000.0 * (1 + 1e-9))


def test_depth_temperature_structural_ends_rejected():
    for bad in (1.0, 0.0, -100.0, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            sine_pos_embed_3d(4, 6, 6, 256, torch.device('cpu'), depth_temperature=bad)


@pytest.mark.skipif(not SAM3_PE_PATH.exists(), reason='SAM3 reference repo not checked out')
@pytest.mark.parametrize('h,w', [(24, 24), (8, 16)])
def test_d1_bit_exact_vs_sam3(h, w):
    """d=1 must reproduce SAM 3's PositionEmbeddingSine bit-for-bit (the warm-start property: pretrained
    fusion Q/K read the positional signal they were trained on). Square and non-square grids (the coordinate
    normalization is per-axis)."""
    dim = 256
    sam3_pe = _load_sam3_position_encoding()
    ref_mod = sam3_pe.PositionEmbeddingSine(num_pos_feats=dim, normalize=True)
    ref = ref_mod(torch.zeros(1, 1, h, w))[0].permute(1, 2, 0).reshape(h * w, dim)
    ours = sine_pos_embed_3d(1, h, w, dim, torch.device('cpu'))
    assert torch.equal(ours, ref), \
        f'd=1 not bit-exact vs SAM3: max|diff|={(ours - ref).abs().max():.3e}'


def test_d1_matches_inline_sam3_form():
    """Inline restatement of SAM 3's convention (normalized (0, 2*pi] coords, paired freqs, interleaved
    sin/cos, y-block then x-block) so the contract is pinned even without the reference repo."""
    h, w, dim = 6, 6, 256
    pe = sine_pos_embed_3d(1, h, w, dim, torch.device('cpu'))

    npf = dim // 2
    k = torch.arange(npf, dtype=torch.float32)
    dim_t = 10000.0 ** (2 * torch.div(k, 2, rounding_mode='floor') / npf)
    eps = 1e-6
    yy = torch.arange(1, h + 1, dtype=torch.float32) / (h + eps) * (2 * math.pi)
    xx = torch.arange(1, w + 1, dtype=torch.float32) / (w + eps) * (2 * math.pi)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    gy, gx = gy.reshape(-1), gx.reshape(-1)

    def block(pos):
        ang = pos[:, None] / dim_t
        return torch.stack((ang[:, 0::2].sin(), ang[:, 1::2].cos()), dim=2).flatten(1)

    expected = torch.cat([block(gy), block(gx)], dim=1)
    assert torch.allclose(pe, expected, atol=1e-6)


def test_depth_actually_shifts_code():
    """Different depth at the same (h, w) must give a different code (depth is encoded)."""
    pe = sine_pos_embed_3d(3, 4, 4, 256, torch.device('cpu')).reshape(3, 4, 4, 256)
    assert not torch.allclose(pe[0, 2, 2], pe[1, 2, 2], atol=1e-4)
    assert not torch.allclose(pe[0, 2, 2], pe[2, 2, 2], atol=1e-4)


def _shared_family_pe(d: int, h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """Reference sine PE with depth folded in using the SAME period family as in-plane (the aliasing bug).

    Built explicitly rather than by passing depth_temperature ~= temperature to sine_pos_embed_3d, which
    would rely on fp32-rounding to make the families equal (and is now rejected by the family-level
    validation). Same coordinate maps and layout as the shipped function; only the depth period differs.
    """
    axis_dim = dim // 2
    eps = 1e-6
    scale = 2 * math.pi
    hh = torch.arange(1, h + 1, dtype=torch.float32, device=device)
    ww = torch.arange(1, w + 1, dtype=torch.float32, device=device)
    hh = hh / (hh[-1] + eps) * scale
    ww = ww / (ww[-1] + eps) * scale
    dd = ((2 * (torch.arange(d, dtype=torch.float32, device=device) + 0.5) / d) - 1) * scale
    k = torch.arange(axis_dim, dtype=torch.float32, device=device)
    period = 10000.0 ** (2 * torch.div(k, 2, rounding_mode='floor') / axis_dim)  # depth uses the SAME period
    gh, gw = torch.meshgrid(hh, ww, indexing='ij')
    inplane_h = gh.reshape(-1)[:, None] / period
    inplane_w = gw.reshape(-1)[:, None] / period
    depth_angles = dd[:, None] / period
    pos_h = (inplane_h[None] + depth_angles[:, None]).reshape(-1, axis_dim)
    pos_w = (inplane_w[None] + depth_angles[:, None]).reshape(-1, axis_dim)
    def block(angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((angles[:, 0::2].sin(), angles[:, 1::2].cos()), dim=2).flatten(1)

    return torch.cat([block(pos_h), block(pos_w)], dim=1)


def test_no_diagonal_aliasing():
    """The distinct depth frequency family must keep diagonal moves distinct: with a shared family,
    (h, w, d) and (h+c', w+c', d-c) collide wherever the step sizes are commensurate.

    On a 4 x 6 x 6 grid the centered depth step is exactly 3x the in-plane step, so the shared-family
    collision pair is (d, h, w) = (2, 1, 1) vs (1, 4, 4). The test first proves that pair DOES collide
    under a shared family (guarding against picking a pair the bug never confused), then that the shipped
    distinct family separates it."""
    dim = 256
    shared = _shared_family_pe(4, 6, 6, dim, torch.device('cpu')).reshape(4, 6, 6, dim)
    assert torch.allclose(shared[2, 1, 1], shared[1, 4, 4], atol=1e-4), \
        'setup error: chosen pair does not collide under the shared family; test would be vacuous'

    pe = sine_pos_embed_3d(4, 6, 6, dim, torch.device('cpu')).reshape(4, 6, 6, dim)
    assert not torch.allclose(pe[2, 1, 1], pe[1, 4, 4], atol=1e-3), \
        'diagonal aliasing: depth family not distinct from in-plane'
