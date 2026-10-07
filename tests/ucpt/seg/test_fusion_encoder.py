import torch

from pumit.ucpt.seg.decoder import FusionEncoder


def test_fusion_encoder_shape():
    enc = FusionEncoder(hidden_size=256, num_heads=8, intermediate_size=2048, num_layers=6)
    vision = torch.randn(1, 4608, 256)
    text = torch.randn(1, 4, 256)
    pos = torch.randn(4608, 256)
    out = enc(vision, text, pos)
    assert out.shape == (1, 4608, 256)


def test_attention_module_names_for_weight_load():
    enc = FusionEncoder(256, 8, 2048, 6)
    keys = dict(enc.named_parameters())
    assert any('layers.0.self_attn.q_proj' in k for k in keys)
    assert any('layers.0.cross_attn.v_proj' in k for k in keys)
    assert any('layers.0.layer_norm1' in k for k in keys)
    assert any('layers.0.mlp' in k for k in keys)


def test_attention_masks_pad_keys_with_bias():
    import torch
    from pumit.ucpt.seg.decoder import Attention

    torch.manual_seed(0)
    attn = Attention(hidden_size=32, num_heads=4).eval()
    q = torch.randn(2, 5, 32)          # (B, Nq, C)
    k = torch.randn(2, 6, 32)          # (B, L, C) — 6 keys
    v = torch.randn(2, 6, 32)
    # bias: mask out the last 3 keys of every batch item
    bias = torch.zeros(2, 1, 1, 6)
    bias[:, :, :, 3:] = float('-inf')

    with torch.no_grad():
        out_masked = attn(q, k, v, attn_bias=bias)
        out_sliced = attn(q, k[:, :3], v[:, :3])   # attend only real keys

    assert torch.allclose(out_masked, out_sliced, atol=1e-5), \
        'masking the last 3 keys must equal attending only the first 3'


def test_fusion_encoder_threads_text_padding_bias():
    import torch
    from pumit.ucpt.seg.decoder import FusionEncoder

    torch.manual_seed(0)
    enc = FusionEncoder(hidden_size=32, num_heads=4, intermediate_size=64, num_layers=2).eval()
    vision = torch.randn(2, 10, 32)     # (K, N, C)
    text = torch.randn(2, 6, 32)        # (K, L, C), L=6
    pos = torch.randn(10, 32)
    bias = torch.zeros(2, 1, 1, 6)
    bias[:, :, :, 3:] = float('-inf')

    with torch.no_grad():
        out_masked = enc(vision, text, pos, bias)
        out_sliced = enc(vision, text[:, :3], pos)   # only 3 real text tokens

    assert torch.allclose(out_masked, out_sliced, atol=1e-5)
