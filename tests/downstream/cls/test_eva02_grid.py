"""EVA-02 at a target grid other than its native 16x16.

EVA-02 pretrains at 224/patch-14 = a 16x16 token grid. To reach an arbitrary 3D token budget the
adapter must decouple the NATIVE grid (which the pretrained weights describe) from the TARGET grid:

  - pos_embed: split off the cls token, bicubically resize the 16x16 patch grid to the target,
    then replicate along depth. Same operation BiomedCLIP already performs.
  - RoPE: purely sinusoidal, nothing learned -- rebuild timm's table at the target feat_shape,
    keeping ref_feat_shape at the native grid so the angular scale stays on the pretrained
    reference rather than being compressed.
"""
import pytest
import torch

from pumit.downstream.cls.backbones.eva02 import Eva02Encoder, _resize_pos_grid


@pytest.mark.parametrize('edge,expected_grid', [(224, 16), (168, 12), (140, 10)])
def test_encoder_reaches_target_token_grid(edge, expected_grid):
    enc = Eva02Encoder(size='base', dims=3, img_size=edge, patch=14, pretrained=False).eval()
    with torch.no_grad():
        global_features, patch_tokens = enc(torch.randn(1, 3, 8, 8, 8))
    assert patch_tokens.shape[1] == expected_grid ** 3
    assert global_features.shape == (1, enc.embed_dim)
    assert enc.model.pos_embed.shape[1] == 1 + expected_grid ** 3
    assert enc.model.rope.get_embed().shape[0] == expected_grid ** 3


def test_pos_grid_resize_preserves_cls_and_shape():
    dim = 8
    pe = torch.randn(1, 1 + 16 * 16, dim)
    out = _resize_pos_grid(pe, native=16, target=12)
    assert out.shape == (1, 1 + 12 * 12, dim)
    torch.testing.assert_close(out[:, :1], pe[:, :1], msg='cls token must pass through untouched')


def test_pos_grid_resize_is_identity_at_native_size():
    pe = torch.randn(1, 1 + 16 * 16, 8)
    torch.testing.assert_close(_resize_pos_grid(pe, native=16, target=16), pe)


def test_rope_is_rebuilt_not_resampled():
    """A rebuilt 12-grid RoPE must differ from the native 16-grid table, and be self-consistent."""
    small = Eva02Encoder(size='base', dims=3, img_size=168, patch=14, pretrained=False)
    native = Eva02Encoder(size='base', dims=3, img_size=224, patch=14, pretrained=False)
    assert small.model.rope.get_embed().shape[0] == 12 ** 3
    assert native.model.rope.get_embed().shape[0] == 16 ** 3
    # sin/cos halves stay unit-norm pairs: evidence the table holds real angles, not resampled ones
    table = small.model.rope.get_embed()
    sin, cos = table.chunk(2, dim=-1)
    torch.testing.assert_close(sin ** 2 + cos ** 2, torch.ones_like(sin), atol=1e-5, rtol=1e-5)
