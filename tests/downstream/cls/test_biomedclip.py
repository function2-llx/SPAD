"""BiomedCLIP adapter tests: official projection contract, native-trunk reuse, revision-pinned loader.

Hub-independent: _load_visual is monkeypatched to controlled fixtures (real tiny timm trunk +
official-style head); the loader test monkeypatches snapshot_download/create_model_from_pretrained.
"""
import inspect
from types import SimpleNamespace

import pytest
import timm
import torch
from torch import nn

import pumit.downstream.cls.backbones.biomedclip as biomedclip


class _FakeVisual(nn.Module):
    """Mirror of open_clip TimmModel's composition (trunk + head) so params register."""

    def __init__(self, trunk: nn.Module, head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def _tiny_trunk(global_pool: str = 'token') -> nn.Module:
    return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0,
                             global_pool=global_pool)


def _head(trunk_dim: int = 192, proj_dim: int = 8) -> nn.Module:
    return nn.Sequential(nn.Dropout(0.0), nn.Linear(trunk_dim, proj_dim, bias=False))


@pytest.fixture
def fake_visual(monkeypatch):
    """Every construction gets a fresh random-init tiny visual, Hub-free."""
    monkeypatch.setattr(biomedclip, '_load_visual',
                        lambda **_: _FakeVisual(_tiny_trunk(), _head()))


def test_returns_official_projected_global_and_projected_patches(fake_visual):
    encoder = biomedclip.BiomedCLIPEncoder(dims=2).eval()
    x = torch.randn(1, 3, 1, 224, 224)
    with torch.no_grad():
        global_features, patch_tokens = encoder(x)
        seq = encoder.visual.trunk.forward_features(x[:, :, 0])
    trunk = encoder.visual.trunk
    torch.testing.assert_close(
        global_features,
        encoder.visual.head(trunk.forward_head(seq, pre_logits=True)),
    )
    torch.testing.assert_close(patch_tokens, encoder.visual.head(seq[:, trunk.num_prefix_tokens:]))
    assert encoder.embed_dim == 8
    assert not hasattr(encoder, 'n_prefix')


def test_2d_global_matches_native_visual_forward(fake_visual):
    encoder = biomedclip.BiomedCLIPEncoder(dims=2).eval()
    x = torch.randn(1, 3, 1, 224, 224)
    with torch.no_grad():
        global_features, _ = encoder(x)
        reference = encoder.visual(x[:, :, 0])   # TimmModel.forward = head(trunk(x))
    torch.testing.assert_close(global_features, reference)


def test_3d_adapted_topology_and_forward(fake_visual):
    encoder = biomedclip.BiomedCLIPEncoder(dims=3, img_size=48).eval()
    trunk = encoder.visual.trunk
    patch_count = 3 ** 3
    assert isinstance(trunk.patch_embed, biomedclip.FixedGridPatchEmbed3d)
    assert trunk.patch_embed.num_patches == patch_count
    assert trunk.pos_embed.shape == (1, 1 + patch_count, trunk.embed_dim)
    x = torch.randn(1, 3, 8, 16, 16)
    with torch.no_grad():
        global_features, patch_tokens = encoder(x)
    assert global_features.shape == (1, encoder.embed_dim)
    assert patch_tokens.shape == (1, patch_count, encoder.embed_dim)
    assert torch.isfinite(global_features).all()
    assert torch.isfinite(patch_tokens).all()


def test_state_dict_strict_roundtrip(fake_visual):
    a = biomedclip.BiomedCLIPEncoder(dims=3, img_size=48)
    b = biomedclip.BiomedCLIPEncoder(dims=3, img_size=48)
    b.load_state_dict(a.state_dict(), strict=True)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


@pytest.mark.parametrize('head', [
    pytest.param(nn.Linear(192, 8, bias=False), id='bare-linear'),
    pytest.param(nn.Sequential(nn.Dropout(0.5), nn.Linear(192, 8, bias=False)), id='dropout-nonzero'),
    pytest.param(nn.Sequential(nn.Dropout(0.0), nn.Linear(192, 8, bias=True)), id='linear-bias'),
    pytest.param(nn.Sequential(nn.Linear(192, 192), nn.Dropout(0.0), nn.Linear(192, 8, bias=False)),
                 id='three-layer'),
])
def test_unexpected_projection_fails_fast(monkeypatch, head):
    monkeypatch.setattr(biomedclip, '_load_visual', lambda **_: _FakeVisual(_tiny_trunk(), head))
    with pytest.raises(RuntimeError, match='projection'):
        biomedclip.BiomedCLIPEncoder(dims=2)


@pytest.mark.parametrize('tamper', [
    pytest.param(lambda t: setattr(t, 'dynamic_img_size', True), id='dynamic-img-size'),
    pytest.param(lambda t: setattr(t, 'global_pool', 'avg'), id='global-pool'),
    pytest.param(lambda t: setattr(t, 'fc_norm', nn.LayerNorm(t.embed_dim)), id='fc-norm'),
    pytest.param(lambda t: setattr(t.head_drop, 'p', 0.1), id='head-drop-rate'),
    pytest.param(lambda t: setattr(t, 'patch_drop', None), id='patch-drop'),
    pytest.param(lambda t: setattr(t, 'norm_pre', nn.LayerNorm(t.embed_dim)), id='norm-pre'),
    pytest.param(lambda t: setattr(t.pos_drop, 'p', 0.1), id='pos-drop-rate'),
    pytest.param(lambda t: setattr(t, 'pos_embed', nn.Parameter(torch.randn(1, 100, t.embed_dim))),
                 id='pos-grid'),
])
def test_unexpected_trunk_topology_fails_fast(monkeypatch, tamper):
    trunk = _tiny_trunk()
    tamper(trunk)
    monkeypatch.setattr(biomedclip, '_load_visual', lambda **_: _FakeVisual(trunk, _head()))
    with pytest.raises(RuntimeError):
        biomedclip.BiomedCLIPEncoder(dims=3, img_size=48)


def test_loader_pins_revision_via_snapshot_and_local_dir(monkeypatch, tmp_path):
    calls = {}
    (tmp_path / 'open_clip_config.json').write_text('{"model_cfg": {"vision_cfg": {}}}')

    def fake_snapshot(repo_id, *, revision, allow_patterns):
        calls['snapshot'] = (repo_id, revision, tuple(allow_patterns))
        return str(tmp_path)

    def fake_create(name, *, vision_cfg):
        calls['create'] = name
        assert vision_cfg['timm_drop_path'] == 0.0
        return SimpleNamespace(visual=_FakeVisual(_tiny_trunk(), _head())), None

    monkeypatch.setattr(biomedclip, 'snapshot_download', fake_snapshot)
    monkeypatch.setattr(biomedclip.open_clip, 'create_model_from_pretrained', fake_create)
    biomedclip.BiomedCLIPEncoder(dims=2)
    assert calls['snapshot'] == (
        biomedclip.BIOMEDCLIP_REPO,
        biomedclip.BIOMEDCLIP_REVISION,
        ('open_clip_config.json', 'open_clip_pytorch_model.bin'),
    )
    assert calls['create'] == f'local-dir:{tmp_path}'


def test_open_clip_version_guard(monkeypatch):
    monkeypatch.setattr(biomedclip.open_clip, '__version__', '0.0.0')
    with pytest.raises(RuntimeError, match=biomedclip.REVIEWED_OPEN_CLIP_VERSION):
        biomedclip.BiomedCLIPEncoder(dims=2)


def test_factory_freeze_and_trainable(fake_visual):
    frozen = biomedclip.biomedclip(dims=2, device='cpu', img_size=224, trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = biomedclip.biomedclip(dims=2, device='cpu', img_size=224, trainable=True)
    assert trainable.training
    assert any(p.requires_grad for p in trainable.parameters())


def test_factory_requires_img_size(fake_visual):
    """img_size has no default: the caller must state the token grid, not inherit one."""
    with pytest.raises(TypeError, match='img_size'):
        biomedclip.biomedclip(dims=2, device='cpu', trainable=False)


def test_factory_does_not_declare_weights(fake_visual):
    """BiomedCLIP loads a pinned HF snapshot, so it takes no weights path.

    The call site in finetune.py forwards --weights only when given, which is what keeps a
    stray path from reaching here; `**_` still absorbs kwargs aimed at other backbones.
    """
    assert 'weights' not in inspect.signature(biomedclip.biomedclip).parameters
