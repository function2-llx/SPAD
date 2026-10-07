"""SAM-Med3D adapter tests (cls env, Hub-independent).

The vendored ImageEncoderViT3D is real code, so tests use the real architecture with random
init; the HF loader seam (_load_state_dict) is monkeypatched. Native contract: 3D-only,
single-channel, transform_batch only converts uint8->float (no norm); forward resizes 64^3->128^3
THEN full-volume Z-normalizes (official order); a constant volume raises; readout = GAP over the
post-neck image embedding (no cls token exists in SAM-Med3D).
"""
import pytest
import torch
from torch import nn

import pumit.downstream.cls.backbones.sam_med3d as sam_med3d

try:   # the two real-weight tests below need the pinned HF checkpoint cached
    from huggingface_hub import hf_hub_download
    _REAL_WEIGHTS = hf_hub_download(sam_med3d.SAM_MED3D_REPO,
                                    sam_med3d.SAM_MED3D_WEIGHTS,
                                    revision=sam_med3d.SAM_MED3D_REVISION)
except Exception:
    _REAL_WEIGHTS = None
_needs_weights = pytest.mark.skipif(_REAL_WEIGHTS is None,
                                    reason='pinned SAM-Med3D weights not cached')


def _random_state_dict() -> dict:
    """Full Sam3D-style state dict (encoder + prompt/mask junk) around a random encoder."""
    enc = sam_med3d.build_encoder()
    sd = {f'image_encoder.{k}': v for k, v in enc.state_dict().items()}
    sd['prompt_encoder.junk'] = torch.zeros(3)
    sd['mask_decoder.junk'] = torch.zeros(3)
    return {'model_state_dict': sd}


@pytest.fixture
def random_weights(monkeypatch):
    """Adapter loads a fixed random encoder state dict, Hub-free (stripped of the image_encoder prefix)."""
    sd = _random_state_dict()
    encoder_sd = {k.removeprefix('image_encoder.'): v
                  for k, v in sd['model_state_dict'].items() if k.startswith('image_encoder.')}
    monkeypatch.setattr(sam_med3d, '_load_state_dict', lambda: encoder_sd)
    return encoder_sd


def test_forward_matches_official_image_embedding(random_weights):
    enc = sam_med3d.SAMMed3DEncoder().eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
        resized = torch.nn.functional.interpolate(x, size=(128, 128, 128), mode='trilinear',
                                                  align_corners=False)
        embedding = enc.encoder(sam_med3d._znormalize(resized))   # resize -> Z-norm -> encoder
    assert embedding.shape == (1, 384, 8, 8, 8)
    torch.testing.assert_close(global_features, embedding.mean(dim=(2, 3, 4)))
    torch.testing.assert_close(
        patch_tokens, embedding.flatten(2).transpose(1, 2))
    assert patch_tokens.shape == (1, 512, 384)
    assert enc.embed_dim == 384
    assert not hasattr(enc, 'n_prefix')


def test_loads_only_image_encoder_keys_strict(tmp_path, monkeypatch):
    sd = _random_state_dict()
    ckpt = tmp_path / 'fake.pth'
    torch.save(sd, ckpt)
    calls = {}

    def fake_download(repo_id, filename, *, revision):
        calls['args'] = (repo_id, filename, revision)
        return str(ckpt)

    monkeypatch.setattr(sam_med3d, 'hf_hub_download', fake_download)
    enc = sam_med3d.SAMMed3DEncoder()
    reference = {k.removeprefix('image_encoder.'): v
                 for k, v in sd['model_state_dict'].items() if k.startswith('image_encoder.')}
    for key, value in enc.encoder.state_dict().items():
        assert torch.equal(value, reference[key]), key
    assert calls['args'] == (sam_med3d.SAM_MED3D_REPO, sam_med3d.SAM_MED3D_WEIGHTS,
                             sam_med3d.SAM_MED3D_REVISION)


def test_img_size_sets_the_token_grid_and_loads_strictly():
    """192^3 -> 12^3 = 1728 tokens, with every checkpoint tensor resampled onto the new shapes."""
    enc = sam_med3d.SAMMed3DEncoder(img_size=192).eval()   # real HF checkpoint, strict load
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
    assert patch_tokens.shape == (1, 1728, 384)
    assert global_features.shape == (1, 384)
    # window clamped to the grid, so window_partition3D pads nothing
    assert enc.encoder.blocks[0].window_size == 12
    assert enc.encoder.blocks[2].window_size == 0            # global block, unchanged


def test_adapt_state_dict_resamples_only_position_tables():
    """pos_embed and rel_pos move to the declared shapes; every other tensor is untouched."""
    model = sam_med3d.build_encoder(img_size=192)
    source = sam_med3d.build_encoder(img_size=sam_med3d.NATIVE_IMG_SIZE).state_dict()
    adapted = sam_med3d.adapt_state_dict(source, model)
    model.load_state_dict(adapted, strict=True)              # shapes all match
    assert adapted['pos_embed'].shape == (1, 12, 12, 12, 768)
    for key, value in source.items():
        if 'pos_embed' in key or 'rel_pos' in key:
            continue
        assert torch.equal(adapted[key], value), key


def test_state_dict_strict_roundtrip(random_weights):
    a = sam_med3d.SAMMed3DEncoder()
    b = sam_med3d.SAMMed3DEncoder()
    b.load_state_dict(a.state_dict(), strict=True)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_factory_contract(random_weights):
    frozen = sam_med3d.sam_med3d(dims=3, device='cpu', img_size=128, trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = sam_med3d.sam_med3d(dims=3, device='cpu', img_size=128, trainable=True)
    assert trainable.training
    assert any(p.requires_grad for p in trainable.parameters())
    with pytest.raises(ValueError, match='3D-only'):
        sam_med3d.sam_med3d(dims=2, device='cpu', img_size=128)


def test_transform_batch_is_raw_float_no_normalization():
    import numpy as np
    # transform_batch only converts uint8 -> float; normalization is deferred to forward (post-resize)
    spread = np.linspace(0, 255, 64 * 64 * 64, dtype=np.uint8).reshape(64, 64, 64)
    out = sam_med3d.transform_batch(np.stack([spread]), is_3d=True)['x']
    assert out.shape == (1, 1, 64, 64, 64)
    assert out.dtype.is_floating_point
    assert out.min().item() == pytest.approx(0.0)
    assert out.max().item() == pytest.approx(255.0)
    with pytest.raises(ValueError, match='3D-only'):
        sam_med3d.transform_batch(np.zeros((2, 64, 64), dtype=np.uint8), is_3d=False)


@_needs_weights
def test_forward_znormalizes_after_resize():
    """forward resizes 64³->128³ THEN Z-normalizes, so the encoder receives unit-std input."""
    import numpy as np
    enc = sam_med3d.SAMMed3DEncoder().eval()  # loads the real HF checkpoint
    x = torch.from_numpy(np.linspace(0, 255, 64 * 64 * 64, dtype=np.uint8).reshape(1, 1, 64, 64, 64))
    captured = {}
    orig_forward = enc.encoder.forward
    def capture(inp, *a, **k):
        captured['x'] = inp
        return orig_forward(inp, *a, **k)
    enc.encoder.forward = capture
    with torch.no_grad():
        enc(x.float())
    z = captured['x']
    assert z.shape[-3:] == (128, 128, 128)            # resized first
    assert z.mean().item() == pytest.approx(0.0, abs=1e-3)
    assert z.std().item() == pytest.approx(1.0, abs=1e-3)   # Z-normed after resize


@_needs_weights
def test_znormalize_raises_on_constant_volume():
    import numpy as np
    enc = sam_med3d.SAMMed3DEncoder().eval()
    const = torch.full((1, 1, 64, 64, 64), 200.0)
    with pytest.raises(ValueError, match='constant'):
        enc(const)
