"""3DINO adapter tests (cls env, Hub-independent).

The vendored DinoVisionTransformer3d is real code, so tests use the real architecture with
random init; the gated-HF loader seam (_load_state_dict) is monkeypatched. Native contract:
3D-only, single-channel, the official per-volume 0.05/99.95 percentile -> [-1,1] scaling (a
zero-range volume follows MONAI and maps to -1), readout = the normed CLS token (1024-dim, the
native forward_head pre-logits), and the EMA teacher is required (not a bare state_dict). A
real-weights test guards the init_values=1e-5 / block_chunks=4 contract against the gated
checkpoint; `img_size` is a per-run argument, with the pos_embed parameter kept at the native 7^3.
"""
import pytest
import torch

import pumit.downstream.cls.backbones.three_dino as three_dino

_REAL_WEIGHTS = None
try:
    from huggingface_hub import hf_hub_download
    _REAL_WEIGHTS = hf_hub_download(three_dino.THREEDINO_REPO, three_dino.THREEDINO_WEIGHTS,
                                    revision=three_dino.THREEDINO_REVISION)
except Exception:
    _REAL_WEIGHTS = None


def _random_encoder_sd() -> dict:
    return three_dino.build_encoder().state_dict()


@pytest.fixture
def random_weights(monkeypatch):
    """Adapter loads a fixed random encoder state dict, Hub-free."""
    sd = _random_encoder_sd()
    monkeypatch.setattr(three_dino, '_load_state_dict', lambda: sd)
    return sd


def test_forward_matches_native_cls_readout(random_weights):
    enc = three_dino.ThreeDinoEncoder().eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
        feat = enc.model.forward_features(enc._resize(x))
    torch.testing.assert_close(global_features, feat['x_norm_clstoken'])
    torch.testing.assert_close(patch_tokens, feat['x_norm_patchtokens'])
    assert global_features.shape == (1, 1024)
    assert patch_tokens.shape == (1, 343, 1024)   # 112^3 -> (112/16)^3 = 7^3 patches
    assert enc.embed_dim == 1024
    assert not hasattr(enc, 'n_prefix')


def test_loads_teacher_checkpoint_keys(monkeypatch, tmp_path):
    """The gated checkpoint is a {'teacher': {'backbone.<k>': v}} dump; the loader strips prefixes."""
    encoder = three_dino.build_encoder()
    teacher_sd = {f'backbone.{k}': v for k, v in encoder.state_dict().items()}
    teacher_sd['backbone.head.weight'] = torch.zeros(3, 1024)   # extra classifier key, must be ignored
    ckpt = tmp_path / 'fake.pth'
    torch.save({'teacher': teacher_sd, 'epoch': 99}, ckpt)
    calls = {}

    def fake_download(repo_id, filename, *, revision, token):
        calls['args'] = (repo_id, filename, revision, token)
        return str(ckpt)

    monkeypatch.setattr(three_dino, 'hf_hub_download', fake_download)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    sd = three_dino._load_state_dict()
    assert calls['args'][:3] == (three_dino.THREEDINO_REPO, three_dino.THREEDINO_WEIGHTS,
                                 three_dino.THREEDINO_REVISION)
    assert 'backbone.' not in ''.join(sd)           # backbone. prefix stripped
    assert 'module.' not in ''.join(sd)             # module. prefix stripped
    assert 'epoch' not in sd                        # top-level non-teacher keys dropped
    # bare encoder keys load with no missing; the extra classifier head is the only unexpected
    enc = three_dino.build_encoder()
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not missing
    assert all('head' in k for k in unexpected)


def test_state_dict_strict_roundtrip(random_weights):
    a = three_dino.ThreeDinoEncoder()
    b = three_dino.ThreeDinoEncoder()
    b.load_state_dict(a.state_dict(), strict=True)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_wrapper_unexpected_key_gate(random_weights, monkeypatch):
    """The production gate allowlists only dino_head./ibot_head. unexpected keys."""
    base = three_dino.build_encoder().state_dict()
    # allowed: a released DINO head key is accepted (no raise)
    allowed = dict(base); allowed['dino_head.last_layer.weight'] = torch.zeros(4)
    monkeypatch.setattr(three_dino, '_load_state_dict', lambda: allowed)
    three_dino.ThreeDinoEncoder()  # must not raise
    # rejected: a non-head, non-encoder key fails the gate
    rejected = dict(base); rejected['rogue.layer.weight'] = torch.zeros(4)
    monkeypatch.setattr(three_dino, '_load_state_dict', lambda: rejected)
    with pytest.raises(RuntimeError, match='unexpected non-head keys'):
        three_dino.ThreeDinoEncoder()


def test_loader_requires_teacher_branch(monkeypatch, tmp_path):
    """A bare state_dict (no `teacher` branch) is rejected, not silently accepted."""
    bare = three_dino.build_encoder().state_dict()
    ckpt = tmp_path / 'bare.pth'
    torch.save(bare, ckpt)
    monkeypatch.setattr(three_dino, 'hf_hub_download',
                        lambda *a, **k: str(ckpt))
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='teacher'):
        three_dino._load_state_dict()


def test_factory_contract(random_weights):
    frozen = three_dino.three_dino(dims=3, device='cpu', img_size=112, trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = three_dino.three_dino(dims=3, device='cpu', img_size=112, trainable=True)
    assert trainable.training
    assert any(p.requires_grad for p in trainable.parameters())
    with pytest.raises(ValueError, match='3D-only'):
        three_dino.three_dino(dims=2, device='cpu', img_size=112)


def test_transform_batch_normalization():
    import numpy as np
    # a volume with a real intensity spread: the 0.05th/99.95th percentile range maps to [-1,1]
    spread = np.linspace(0, 255, 64 * 64 * 64, dtype=np.uint8).reshape(64, 64, 64)
    out = three_dino.transform_batch(np.stack([spread]), is_3d=True)['x']
    assert out.shape == (1, 1, 64, 64, 64)
    assert out.min().item() == pytest.approx(-1.0, abs=1e-4)
    assert out.max().item() == pytest.approx(1.0, abs=1e-4)
    # zero percentile range follows MONAI ScaleIntensityRange: `img - a_min + b_min`, so a
    # constant volume maps to b_min = -1 rather than failing the run
    degenerate = three_dino.transform_batch(
        np.stack([np.zeros((64, 64, 64), dtype=np.uint8),
                  np.full((64, 64, 64), 255, dtype=np.uint8)]), is_3d=True)['x']
    torch.testing.assert_close(degenerate, torch.full_like(degenerate, -1.0))


def test_transform_batch_sparse_volume_does_not_collapse_the_batch():
    """A volume too sparse for its 99.95th percentile to clear zero must not break its neighbours.

    vesselmnist3d has one such training sample (76/262144 nonzero); it previously raised and took
    the whole run with it.
    """
    import numpy as np
    sparse = np.zeros((64, 64, 64), dtype=np.uint8)
    sparse.reshape(-1)[:76] = 255
    normal = np.linspace(0, 255, 64 * 64 * 64, dtype=np.uint8).reshape(64, 64, 64)
    out = three_dino.transform_batch(np.stack([sparse, normal]), is_3d=True)['x']
    assert torch.isfinite(out).all()
    assert out[1].min().item() == pytest.approx(-1.0, abs=1e-4)
    assert out[1].max().item() == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match='3D-only'):
        three_dino.transform_batch(np.zeros((2, 64, 64), dtype=np.uint8), is_3d=False)


@pytest.mark.skipif(_REAL_WEIGHTS is None, reason='gated 3DINO weights not cached')
def test_real_weights_load_and_forward():
    enc = three_dino.ThreeDinoEncoder().eval()
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        g, p = enc(x)
    assert g.shape == (2, 1024)
    assert p.shape == (2, 343, 1024)   # 112^3 -> 7^3 patches
    assert enc.embed_dim == 1024
