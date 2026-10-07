import torch

from pumit.ucpt.seg.decoder import PixelDecoder, SemanticHead


def test_pixel_decoder_allocates_3_runs_2():
    pd = PixelDecoder(hidden_size=256, num_upsampling_stages=3)
    assert len(pd.conv_layers) == 3
    assert len(pd.norms) == 3


def test_semantic_projection_uses_channels_last_3d_weight_layout():
    head = SemanticHead(hidden_size=8, num_heads=2)
    assert head.semantic_projection.weight.stride() == (8, 1, 8, 8, 8)


def test_semantic_head_one_channel_per_forward():
    head = SemanticHead(hidden_size=256, num_heads=8, num_upsampling_stages=3)
    levels = {
        '1/16': torch.randn(1, 256, 8, 24, 24),
        '1/8': torch.randn(1, 256, 16, 48, 48),
        '1/4': torch.randn(1, 256, 32, 96, 96),
    }
    fused_1_16 = torch.randn(1, 256, 8, 24, 24)
    text_k = torch.randn(1, 1, 256)
    bias = torch.zeros(1, 1, 1, text_k.shape[1])
    out = head(fused_1_16, levels, text_k, bias, 0)
    assert out.shape == (1, 1, 32, 96, 96)


def test_semantic_head_prenorm_bare_residual_order():
    """SAM 3 order: attn = cross_attn(norm(pix), text); pix = pix + attn.
    NOT post-norm pix = norm(pix + attn)."""
    import torch
    from pumit.ucpt.seg.decoder import SemanticHead
    torch.manual_seed(0)
    head = SemanticHead(hidden_size=32, num_heads=4, num_upsampling_stages=3).eval()
    K, d, h, w = 2, 4, 8, 8
    fused = torch.randn(K, 32, d, h, w)
    levels = {'1/8': torch.randn(1, 32, 2 * d, 2 * h, 2 * w),
              '1/4': torch.randn(1, 32, 4 * d, 4 * h, 4 * w)}
    text_k = torch.randn(K, 3, 32)      # L=3 tokens now, not 1
    bias = torch.zeros(K, 1, 1, 3)
    # reconstruct the expected decoder-input feature (pre-norm + bare residual)
    pix = fused.flatten(2).transpose(1, 2)
    normed = head.prompt_cross_attn_norm(pix)
    attn = head.prompt_cross_attn(query=normed, key=text_k, value=text_k, attn_bias=bias)
    expected_start = pix + attn

    captured = {}
    orig = head.pixel_decoder.forward
    def spy(start, lv, da):
        captured['start'] = start.flatten(2).transpose(1, 2).clone()
        return orig(start, lv, da)
    head.pixel_decoder.forward = spy

    with torch.no_grad():
        head(fused, levels, text_k, bias, 0)
    assert torch.allclose(captured['start'], expected_start, atol=1e-5), \
        'decoder input must be pix + attn (pre-norm), not norm(pix + attn)'


def test_semantic_head_accepts_multi_token_text():
    """No more L==1 assert; L>1 must work."""
    import torch
    from pumit.ucpt.seg.decoder import SemanticHead
    head = SemanticHead(hidden_size=32, num_heads=4, num_upsampling_stages=3).eval()
    K, d, h, w = 2, 4, 8, 8
    fused = torch.randn(K, 32, d, h, w)
    levels = {'1/8': torch.randn(1, 32, 2 * d, 2 * h, 2 * w),
              '1/4': torch.randn(1, 32, 4 * d, 4 * h, 4 * w)}
    text_k = torch.randn(K, 5, 32)
    bias = torch.zeros(K, 1, 1, 5)
    with torch.no_grad():
        out = head(fused, levels, text_k, bias, 0)
    assert out.shape[0] == K and out.shape[1] == 1
