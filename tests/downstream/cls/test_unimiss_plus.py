"""UniMiSS+ adapter tests (cls env, uses the downloaded GDrive checkpoint).

The vendored downstream MiT_encoder is real code; tests load the actual released
UniMissPlus.pth student checkpoint (pretrained/unimiss_plus/UniMissPlus.pth). Native contract:
3D-only, single-channel, min-max [0,1] (MedMNIST3D is uint8-scaled; the encoder's first block
is InstanceNorm, which absorbs input scale), readout = the official pre-head CLS/GAP average
`0.5*(norm_new(CLS) + mean(norm_new(patches)))` (320-dim), so the harness head reproduces the
official `0.5*[head_new(norm(CLS)) + head_new(mean(norm(patches)))]` logits exactly.
"""
import os

import numpy as np
import pytest
import torch

import pumit.downstream.cls.backbones.unimiss_plus as unimiss_plus

WEIGHTS = 'pretrained/unimiss_plus/UniMissPlus.pth'
pytestmark = pytest.mark.skipif(not os.path.exists(WEIGHTS),
                                reason=f'{WEIGHTS} not downloaded (GDrive; see README)')


def _encoder():
    return unimiss_plus.UniMissPlusEncoder(weights=WEIGHTS)


def test_loads_student_checkpoint_keys():
    sd = unimiss_plus._load_encoder_state_dict(WEIGHTS)
    assert 'module.backbone.transformer.' not in ''.join(sd)   # student prefix stripped
    enc = unimiss_plus._build_encoder()
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    # every transformer-block / patch-embed / pos-embed / cls / conv-block key must load
    assert not any(k in ('cls_tokens1', 'pos_embed3D1', 'patch_embed3D1.proj.conv.weight',
                         'ConvBlock3D0.0.conv.weight', 'block1.0.mlp.fc1.weight') for k in missing)
    # the only unexpected keys are the pretrained MM model's 2D branch (incl. shared-block sr2D convs)
    _2d = ('ConvBlock2D', 'patch_embed2D', 'pos_embed2D', 'cls_tokens2D', 'block2D',
           'Decblock2D', 'DecEmbed2D', 'DecPosEmbed2D', 'TransposeConv2D', 'DeConvBlock2D', 'recon_conv2D')
    assert all(k.startswith(_2d) or '.sr2D.' in k for k in unexpected)


def test_forward_matches_native_readout():
    enc = _encoder().eval()
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        global_features, patch_tokens = enc.model.forward3d_features(x)
        manual_seq = enc.model._forward_stages_3d(x)
        manual = enc.model.norm_new(manual_seq)
        # official readout: 0.5 * (norm_new(CLS) + mean(norm_new(patches)))
        expected_global = 0.5 * (manual[:, 0] + manual[:, 1:].mean(dim=1))
    torch.testing.assert_close(global_features, expected_global)
    torch.testing.assert_close(patch_tokens, manual[:, 1:])
    assert global_features.shape == (1, 320)
    assert patch_tokens.shape == (1, 27, 320)   # 96^3 -> uniform /32 = 3^3
    assert enc.embed_dim == 320


def test_adapter_forward_resamples_to_an_isotropic_cube():
    enc = _encoder().eval()
    x = torch.randn(2, 1, 64, 64, 64)                       # MedMNIST3D native cube
    with torch.no_grad():
        g, p = enc(x)
    assert g.shape == (2, 320)
    assert p.shape == (2, 27, 320)                          # 96^3 -> uniform /32 = 3^3


def test_state_dict_strict_roundtrip():
    a = _encoder()
    b = _encoder()
    b.load_state_dict(a.state_dict(), strict=True)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_factory_contract():
    frozen = unimiss_plus.unimiss_plus(dims=3, device='cpu', weights=WEIGHTS, img_size=96, trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = unimiss_plus.unimiss_plus(dims=3, device='cpu', weights=WEIGHTS, img_size=96, trainable=True)
    assert trainable.training
    assert any(p.requires_grad for p in trainable.parameters())
    with pytest.raises(ValueError, match='3D-only'):
        unimiss_plus.unimiss_plus(dims=2, device='cpu', weights=WEIGHTS, img_size=96)


def test_transform_batch_normalization():
    images = np.stack([np.zeros((64, 64, 64), dtype=np.uint8),
                       np.full((64, 64, 64), 255, dtype=np.uint8)])
    out = unimiss_plus.transform_batch(images, is_3d=True)['x']
    assert out.shape == (2, 1, 64, 64, 64)
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]))
    torch.testing.assert_close(out[1], torch.ones_like(out[1]))
    with pytest.raises(ValueError, match='3D-only'):
        unimiss_plus.transform_batch(np.zeros((2, 64, 64), dtype=np.uint8), is_3d=False)


def test_official_forward_equivalence():
    """head_new(forward3d_features global) == model.forward() logits (linearity of head_new)."""
    enc = _encoder().eval()
    x = torch.randn(2, 1, 96, 96, 96)
    with torch.no_grad():
        global_feat, _ = enc.model.forward3d_features(x)
        ours = enc.model.head_new(global_feat)
        official = enc.model.forward(x)
    torch.testing.assert_close(ours, official, atol=1e-5, rtol=1e-5)


def test_img_size_sets_the_stage_grid():
    """192^3 -> the /16 stage is 12^3, matching the flat ViTs; the readout is one deeper, 6^3."""
    enc = unimiss_plus.UniMissPlusEncoder(weights=WEIGHTS, img_size=192).eval()
    with torch.no_grad():
        global_features, patch_tokens = enc(torch.randn(1, 1, 64, 64, 64))
    assert patch_tokens.shape == (1, 216, 320)
    assert global_features.shape == (1, 320)


def test_recompute_stem_preserves_weights_and_output():
    """Recomputing the stem in backward is transparent: same state dict, same forward."""
    plain = unimiss_plus.UniMissPlusEncoder(weights=WEIGHTS, recompute_stem=False).eval()
    wrapped = unimiss_plus.UniMissPlusEncoder(weights=WEIGHTS, recompute_stem=True).eval()
    assert {k.replace('.block.', '.') for k in wrapped.state_dict()} == set(plain.state_dict())
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        torch.testing.assert_close(plain(x)[0], wrapped(x)[0])
