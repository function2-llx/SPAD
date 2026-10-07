"""M3D-CLIP compact-wrapper and SDPA-replacement tests (m3d env only, Hub-independent).

Runs under `pixi run -e m3d pytest tests/downstream/cls_m3d` (monai 1.3 / transformers 4.44).
_load_remote_vision is monkeypatched to a minimal fake of the reviewed remote topology
(patch_embedding.position_embeddings + ModuleList of monai TransformerBlocks + norm).
"""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from monai.networks.blocks.selfattention import SABlock as MonaiSABlock
from monai.networks.blocks.transformerblock import TransformerBlock
from torch import nn

import pumit.downstream.cls.backbones.m3d_clip as m3d_clip
from pumit.downstream.cls.backbones.monai_sablock_sdpa import SABlock as SDPASABlock

DIM = 8
LAYERS = 2
NATIVE_TOKENS = 8 * 16 * 16   # 2048, M3D's native (8,16,16) grid
TARGET_TOKENS = 16 ** 3       # 4096


class _FakePatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.randn(1, NATIVE_TOKENS, DIM))


class _FakeVisionEncoder(nn.Module):
    """Minimal stand-in for the reviewed remote ViT structure."""

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, DIM))
        self.patch_embedding = _FakePatchEmbedding()
        self.blocks = nn.ModuleList(
            TransformerBlock(DIM, DIM * 2, num_heads=2, dropout_rate=0.0, qkv_bias=True)
            for _ in range(LAYERS)
        )
        self.norm = nn.LayerNorm(DIM)

    def forward(self, x):
        tok = self.patch_embedding.position_embeddings.expand(x.shape[0], -1, -1)
        seq = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tok], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        return self.norm(seq), []


def _fake_remote_vision():
    return _FakeVisionEncoder(), nn.Linear(DIM, DIM)


@pytest.fixture
def fake_remote(monkeypatch):
    monkeypatch.setattr(m3d_clip, '_load_remote_vision', _fake_remote_vision)


def test_compact_wrapper_drops_text_side(fake_remote):
    enc = m3d_clip.M3DEncoder()
    keys = set(enc.state_dict())
    assert keys
    assert all(key.startswith(('vision_encoder.', 'vision_projection.')) for key in keys)
    assert not hasattr(enc, 'm')
    assert not hasattr(enc, 'n_prefix')
    assert enc.embed_dim == DIM


def test_position_interpolation_applied(monkeypatch):
    source_encoder = _FakeVisionEncoder()
    source_pe = source_encoder.patch_embedding.position_embeddings.data.clone()
    monkeypatch.setattr(m3d_clip, '_load_remote_vision',
                        lambda: (source_encoder, nn.Linear(DIM, DIM)))
    enc = m3d_clip.M3DEncoder()
    pe = enc.vision_encoder.patch_embedding.position_embeddings
    assert pe.shape == (1, TARGET_TOKENS, DIM)
    import einops
    grid = einops.rearrange(source_pe, '1 (d h w) c -> 1 c d h w', d=8, h=16, w=16)
    grid = F.interpolate(grid, size=(16, 16, 16), mode='trilinear', align_corners=False)
    reference = einops.rearrange(grid, '1 c d h w -> 1 (d h w) c')
    torch.testing.assert_close(pe.data, reference)


def test_forward_matches_compact_encode_image(fake_remote):
    enc = m3d_clip.M3DEncoder().eval()
    x = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
        resized = F.interpolate(x, size=m3d_clip.INPUT_SHAPE, mode='trilinear', align_corners=False)
        seq, _ = enc.vision_encoder(resized)
        seq = F.normalize(enc.vision_projection(seq), dim=-1)
    torch.testing.assert_close(global_features, seq[:, 0])
    torch.testing.assert_close(patch_tokens, seq[:, 1:])
    assert patch_tokens.shape == (1, TARGET_TOKENS, DIM)
    # every projected token (global + patch) is L2-normalized
    torch.testing.assert_close(
        patch_tokens.norm(dim=-1), torch.ones(1, TARGET_TOKENS), atol=1e-5, rtol=1e-5
    )


def test_all_attention_blocks_swapped(fake_remote):
    enc = m3d_clip.M3DEncoder()
    swapped = [m for m in enc.vision_encoder.modules() if type(m) is SDPASABlock]
    remaining = [m for m in enc.vision_encoder.modules() if type(m) is MonaiSABlock]
    assert len(swapped) == LAYERS
    assert not remaining


def test_swap_count_mismatch_fails_fast(fake_remote, monkeypatch):
    monkeypatch.setattr(m3d_clip, 'swap_sdpa_attention', lambda root: 0)
    with pytest.raises(RuntimeError, match='SABlock'):
        m3d_clip.M3DEncoder()


def test_strict_state_dict_roundtrip(fake_remote):
    a = m3d_clip.M3DEncoder()
    b = m3d_clip.M3DEncoder()
    b.load_state_dict(a.state_dict(), strict=True)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_remote_loading_pins_revision(monkeypatch):
    calls = {}

    def fake_from_pretrained(name, *, trust_remote_code, revision):
        calls['args'] = (name, trust_remote_code, revision)
        return SimpleNamespace(vision_encoder=_FakeVisionEncoder(), mm_vision_proj=nn.Linear(DIM, DIM))

    import transformers
    monkeypatch.setattr(transformers, 'AutoModel', SimpleNamespace(from_pretrained=fake_from_pretrained))
    vision_encoder, projection = m3d_clip._load_remote_vision()
    assert calls['args'] == (m3d_clip.M3D_NAME, True, m3d_clip.M3D_REVISION)
    assert isinstance(vision_encoder, _FakeVisionEncoder)


def test_factory_contract(fake_remote):
    frozen = m3d_clip.m3d(dims=3, device='cpu', trainable=False)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    trainable = m3d_clip.m3d(dims=3, device='cpu', trainable=True)
    assert trainable.training
    assert any(p.requires_grad for p in trainable.parameters())
    with pytest.raises(ValueError, match='3D-only'):
        m3d_clip.m3d(dims=2, device='cpu')


# --- SDPA replacement: forward parity and lifecycle preservation ---

def _block_container(save_attn: bool = False) -> nn.Sequential:
    return nn.Sequential(
        TransformerBlock(DIM, DIM * 2, num_heads=2, dropout_rate=0.0, qkv_bias=True,
                         save_attn=save_attn)
    )


def test_swap_sdpa_forward_backward_and_lifecycle_parity():
    reference_container = _block_container().double().eval()
    reference_container[0].attn.qkv.bias.requires_grad_(False)
    container = copy.deepcopy(reference_container)
    old_attn = container[0].attn
    old_attn.qkv.bias.requires_grad_(False)
    reference_x = torch.randn(2, 16, DIM, dtype=torch.float64, requires_grad=True)
    x = reference_x.detach().clone().requires_grad_(True)

    n = m3d_clip.swap_sdpa_attention(container)
    assert n == 1
    new_attn = container[0].attn
    assert type(new_attn) is SDPASABlock
    # lifecycle: dtype, mode, per-parameter requires_grad, save_attn, state values
    assert new_attn.qkv.weight.dtype == torch.float64
    assert new_attn.training is False
    assert new_attn.qkv.weight.requires_grad is True
    assert new_attn.qkv.bias.requires_grad is False
    assert new_attn.save_attn is False
    for key, value in old_attn.state_dict().items():
        assert torch.equal(value, new_attn.state_dict()[key]), key

    reference = reference_container(reference_x)
    out = container(x)
    torch.testing.assert_close(out, reference, atol=1e-12, rtol=1e-12)
    grad_output = torch.randn_like(reference)
    reference.backward(grad_output)
    out.backward(grad_output)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-12, rtol=1e-12)
    for (reference_name, reference_param), (name, param) in zip(
        reference_container.named_parameters(), container.named_parameters(), strict=True
    ):
        assert name == reference_name
        if reference_param.grad is None:
            assert param.grad is None, name
        else:
            torch.testing.assert_close(param.grad, reference_param.grad, atol=1e-12, rtol=1e-12)


def test_swap_sdpa_save_attn_branch():
    container = _block_container(save_attn=True).eval()
    old_attn = container[0].attn
    x = torch.randn(2, 16, DIM)
    with torch.no_grad():
        reference = container(x)
        reference_mat = old_attn.att_mat

    m3d_clip.swap_sdpa_attention(container)
    new_attn = container[0].attn
    assert new_attn.save_attn is True
    with torch.no_grad():
        out = container(x)
    torch.testing.assert_close(out, reference, atol=1e-12, rtol=1e-12)
    assert new_attn.att_mat.shape == (2, 2, 16, 16)
    torch.testing.assert_close(new_attn.att_mat, reference_mat, atol=1e-12, rtol=1e-12)
