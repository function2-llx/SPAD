"""EVA-02 rope-math regression tests (hub-independent) + native-forward adapter tests.

The adapter tests stand in eva02_tiny (random init, no Hub) for the base/large variants: the
reviewed invariants (rope base, grid, pooling, prefix topology) are identical across variants.
"""
from functools import lru_cache

import pytest
import timm
import torch
from torch import nn

import pumit.downstream.cls.backbones.eva02 as eva02
from pumit.downstream.cls.backbones._timm_patch_embed import FixedGridPatchEmbed3d
from pumit.downstream.cls.backbones.eva02 import (
    EVA_DEPTH_ROPE_BASE,
    EVA_ROPE_BASE,
    _extend_eva_rope_3d,
)

TINY = 'eva02_tiny_patch14_224'
GRID = 16  # 224/14


@pytest.fixture
def tiny_eva(monkeypatch):
    """Stand eva02_tiny (random init) in for both reviewed variants, Hub-free."""
    monkeypatch.setattr(eva02, 'EVA_MODELS', {'base': TINY, 'large': TINY})
    real_create = timm.create_model
    monkeypatch.setattr(
        timm, 'create_model',
        lambda name, pretrained=True, **kw: real_create(name, pretrained=False, **kw),
    )


@lru_cache(maxsize=1)
def _native_model():
    return timm.create_model('eva02_base_patch14_224', pretrained=False, num_classes=0)


def _native_rope() -> torch.Tensor:
    return _native_model().rope.get_embed()


def _relative_coefficients(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sin_a, cos_a = a.chunk(2, dim=-1)
    sin_b, cos_b = b.chunk(2, dim=-1)
    return sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b


def test_eva_rope_depth_one_is_bit_exact():
    rope_2d = _native_rope()
    rope_3d = _extend_eva_rope_3d(rope_2d, depth=1, depth_base=EVA_DEPTH_ROPE_BASE)
    assert torch.equal(rope_3d, rope_2d)


def test_eva_rope_preserves_same_slice_relative_rotation():
    rope_2d = _native_rope()
    depth = 3
    rope_3d = _extend_eva_rope_3d(rope_2d, depth=depth, depth_base=EVA_DEPTH_ROPE_BASE)
    rope_3d = rope_3d.view(depth, rope_2d.shape[0], rope_2d.shape[1])

    native_relative = _relative_coefficients(rope_2d[0], rope_2d[17])
    for depth_idx in range(depth):
        relative = _relative_coefficients(rope_3d[depth_idx, 0], rope_3d[depth_idx, 17])
        torch.testing.assert_close(relative[0], native_relative[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(relative[1], native_relative[1], atol=2e-6, rtol=2e-6)


def test_eva_distinct_depth_family_breaks_shared_family_near_collision():
    rope_2d = _native_rope()
    depth = 16
    distinct = _extend_eva_rope_3d(rope_2d, depth=depth, depth_base=EVA_DEPTH_ROPE_BASE)
    shared = _extend_eva_rope_3d(rope_2d, depth=depth, depth_base=EVA_ROPE_BASE)
    distinct = distinct.view(depth, 16, 16, -1)
    shared = shared.view(depth, 16, 16, -1)

    # On the 16^3 evaluation grid, this distant displacement nearly aliases when depth reuses
    # EVA's in-plane frequency family. The quotient-10 family separates the same pair.
    distinct_gap = torch.linalg.vector_norm(distinct[0, 11, 11] - distinct[14, 0, 0])
    shared_gap = torch.linalg.vector_norm(shared[0, 11, 11] - shared[14, 0, 0])
    assert shared_gap < 0.02
    assert distinct_gap > 4.0


def test_eva_block_accepts_extended_rope_table():
    model = _native_model()
    rope = _extend_eva_rope_3d(_native_rope(), depth=2, depth_base=EVA_DEPTH_ROPE_BASE)
    x = torch.randn(1, 1 + rope.shape[0], model.embed_dim)
    with torch.no_grad():
        out = model.blocks[0](x, rope=rope)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# --- adapter: native-forward reuse, adapted topology, lifecycle ---

def test_eva_2d_delegates_to_native_timm_execution(tiny_eva):
    enc = eva02.Eva02Encoder('base', dims=2, pretrained=False).eval()
    x = torch.randn(1, 3, 1, 224, 224)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
        seq = enc.model.forward_features(x[:, :, 0])
    torch.testing.assert_close(global_features, enc.model.forward_head(seq, pre_logits=True))
    torch.testing.assert_close(patch_tokens, seq[:, enc.model.num_prefix_tokens:])
    assert not hasattr(enc, 'n_prefix')


def test_eva_3d_adapted_topology(tiny_eva):
    enc = eva02.Eva02Encoder('base', dims=3, pretrained=False)
    model = enc.model
    patch_count = GRID ** 3
    assert isinstance(model.patch_embed, FixedGridPatchEmbed3d)
    assert model.patch_embed.num_patches == patch_count
    assert model.pos_embed.shape == (1, 1 + patch_count, model.embed_dim)
    assert model.rope.get_embed().shape[0] == patch_count
    assert not hasattr(enc, 'n_prefix')


def test_eva_3d_forward_smoke(tiny_eva):
    enc = eva02.Eva02Encoder('base', dims=3, pretrained=False).eval()
    x = torch.randn(1, 3, 16, 16, 16)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
    assert global_features.shape == (1, enc.embed_dim)
    assert patch_tokens.shape == (1, GRID ** 3, enc.embed_dim)
    assert torch.isfinite(global_features).all()
    assert torch.isfinite(patch_tokens).all()


def test_eva_state_dict_strict_roundtrip(tiny_eva):
    a2 = eva02.Eva02Encoder('base', dims=2, pretrained=False).eval()
    b2 = eva02.Eva02Encoder('base', dims=2, pretrained=False).eval()
    b2.load_state_dict(a2.state_dict(), strict=True)
    x = torch.randn(1, 3, 1, 224, 224)
    with torch.no_grad():
        assert torch.equal(a2(x)[0], b2(x)[0])

    a3 = eva02.Eva02Encoder('base', dims=3, pretrained=False)
    b3 = eva02.Eva02Encoder('base', dims=3, pretrained=False)
    b3.load_state_dict(a3.state_dict(), strict=True)
    for key, value in a3.state_dict().items():
        assert torch.equal(value, b3.state_dict()[key]), key


def test_eva_rope_buffer_follows_dtype_and_stays_nonpersistent(tiny_eva):
    enc = eva02.Eva02Encoder('base', dims=3, pretrained=False)
    assert not any('rope' in key for key in enc.state_dict())
    enc = enc.double()
    rope_buffers = [buf for name, buf in enc.named_buffers() if 'rope' in name]
    assert rope_buffers and all(buf.dtype == torch.float64 for buf in rope_buffers)


@pytest.mark.parametrize('tamper', [
    pytest.param(lambda m: setattr(m, 'dynamic_img_size', True), id='dynamic-img-size'),
    pytest.param(lambda m: setattr(m, 'patch_drop', nn.Identity()), id='patch-drop'),
    pytest.param(lambda m: setattr(m, 'no_embed_class', True), id='no-embed-class'),
    pytest.param(lambda m: setattr(m, 'global_pool', 'token'), id='global-pool'),
    pytest.param(lambda m: setattr(m, 'norm_pre', nn.LayerNorm(m.embed_dim)), id='norm-pre'),
    pytest.param(lambda m: setattr(m.pos_drop, 'p', 0.1), id='pos-drop-rate'),
    pytest.param(lambda m: setattr(m.head_drop, 'p', 0.1), id='head-drop-rate'),
    pytest.param(lambda m: setattr(m.rope, 'temperature', 1.0), id='rope-base'),
    pytest.param(lambda m: setattr(m, 'pos_embed', nn.Parameter(torch.randn(1, 100, m.embed_dim))),
                 id='pos-grid'),
])
def test_eva_unexpected_topology_fails_fast(tiny_eva, monkeypatch, tamper):
    model = timm.create_model(TINY, pretrained=False, num_classes=0)
    tamper(model)
    monkeypatch.setattr(timm, 'create_model', lambda *a, **k: model)
    with pytest.raises(RuntimeError):
        eva02.Eva02Encoder('base', dims=3, pretrained=False)


def test_eva_factory_freeze_and_trainable(tiny_eva):
    frozen = eva02.eva02_base(dims=2, device='cpu', img_size=224, trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = eva02.eva02_base(dims=2, device='cpu', img_size=224, trainable=True)
    assert trainable.training
    assert any(p.requires_grad for p in trainable.parameters())


def test_eva_factory_requires_img_size(tiny_eva):
    """img_size has no default: at patch 14 it sets the token grid, so it must be stated."""
    with pytest.raises(TypeError, match='img_size'):
        eva02.eva02_base(dims=2, device='cpu', trainable=False)
