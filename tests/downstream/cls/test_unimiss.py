"""UniMiSS (ECCV 2022 predecessor) adapter tests (cls env, uses the downloaded GDrive checkpoint).

The vendored VisionTransformer is real code; tests load the actual released UniMiss_small.pth
student checkpoint. Native contract mirrors unimiss_plus: 3D-only, single-channel, min-max [0,1],
readout = the official pre-head CLS/GAP average (512-dim for the `small` variant).
"""
import os

import numpy as np
import pytest
import torch

import pumit.downstream.cls.backbones.unimiss as unimiss

WEIGHTS = 'pretrained/unimiss/UniMiss_small.pth'
pytestmark = pytest.mark.skipif(not os.path.exists(WEIGHTS),
                                reason=f'{WEIGHTS} not downloaded (GDrive; see README)')


def _encoder():
    return unimiss.UniMissEncoder(weights=WEIGHTS)


def test_loads_student_checkpoint_keys():
    sd = unimiss._load_encoder_state_dict(WEIGHTS)
    assert 'module.backbone.transformer.' not in ''.join(sd)   # student prefix stripped
    enc = unimiss._build_encoder()
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not any(k in ('cls_tokens1', 'pos_embed3D1', 'patch_embed3D1.proj.conv.weight',
                         'patch_embed3D0.conv.weight', 'block1.0.mlp.fc1.weight') for k in missing)
    _2d = ('ConvBlock2D', 'patch_embed2D', 'pos_embed2D', 'cls_tokens2D', 'block2D',
           'Decblock2D', 'DecEmbed2D', 'DecPosEmbed2D', 'TransposeConv2D', 'DeConvBlock2D', 'recon_conv2D')
    assert all(k.startswith(_2d) or '.sr2D.' in k for k in unexpected)


def test_forward_matches_native_readout():
    enc = _encoder().eval()
    x = torch.randn(1, 1, 96, 96, 96)          # cubic: the adapter's isotropic contract
    with torch.no_grad():
        global_features, patch_tokens = enc.model.forward3d_features(x)
        manual_seq = enc.model._forward_stages_3d(x)
        manual = enc.model.norm_new(manual_seq)
        expected_global = 0.5 * (manual[:, 0] + manual[:, 1:].mean(dim=1))
    torch.testing.assert_close(global_features, expected_global)
    torch.testing.assert_close(patch_tokens, manual[:, 1:])
    assert global_features.shape == (1, 512)
    assert enc.embed_dim == 512


def test_adapter_forward_resamples_to_an_isotropic_cube():
    """MedMNIST3D is isotropic 64^3, so the input is a cube, not the anisotropic training crop."""
    enc = _encoder().eval()                                  # img_size=96 by default here
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        g, p = enc(x)
    assert g.shape == (2, 512)
    assert p.shape == (2, 27, 512)   # 96^3 -> uniform /32 = 3^3 stage-4 tokens


def test_img_size_sets_the_stage_grid():
    """192^3 -> /16 stage is 12^3 (the aligned grid); the stage-4 readout is one deeper, 6^3."""
    enc = unimiss.UniMissEncoder(weights=WEIGHTS, img_size=192).eval()
    with torch.no_grad():
        g, p = enc(torch.randn(1, 1, 64, 64, 64))
    assert p.shape == (1, 216, 512)
    assert g.shape == (1, 512)


def test_state_dict_strict_roundtrip():
    a = _encoder()
    b = _encoder()
    b.load_state_dict(a.state_dict(), strict=True)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_factory_contract():
    frozen = unimiss.unimiss(dims=3, device='cpu', weights=WEIGHTS, img_size=96, trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = unimiss.unimiss(dims=3, device='cpu', weights=WEIGHTS, img_size=96, trainable=True)
    assert trainable.training
    with pytest.raises(ValueError, match='3D-only'):
        unimiss.unimiss(dims=2, device='cpu', weights=WEIGHTS, img_size=96)


def test_transform_batch_normalization():
    images = np.stack([np.zeros((64, 64, 64), dtype=np.uint8),
                       np.full((64, 64, 64), 255, dtype=np.uint8)])
    out = unimiss.transform_batch(images, is_3d=True)['x']
    assert out.shape == (2, 1, 64, 64, 64)
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]))
    torch.testing.assert_close(out[1], torch.ones_like(out[1]))
    with pytest.raises(ValueError, match='3D-only'):
        unimiss.transform_batch(np.zeros((2, 64, 64), dtype=np.uint8), is_3d=False)


def test_official_forward_equivalence():
    """head_new(forward3d_features global) == model.forward() logits (linearity of head_new)."""
    enc = _encoder().eval()
    x = torch.randn(2, 1, 96, 96, 96)          # cubic: the adapter's isotropic contract
    with torch.no_grad():
        global_feat, _ = enc.model.forward3d_features(x)
        ours = enc.model.head_new(global_feat)
        official = enc.model.forward(x)
    torch.testing.assert_close(ours, official, atol=1e-5, rtol=1e-5)
