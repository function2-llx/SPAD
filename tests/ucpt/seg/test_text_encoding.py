import torch

from pumit.ucpt.seg.text_encoding import TextEncoder


def test_text_encoder_projects_token_sequence():
    enc = TextEncoder(text_embed_dim=1152, hidden_size=256)
    x = torch.randn(4, 24, 1152)      # (K, L, 1152)
    out = enc(x)
    assert out.shape == (4, 24, 256)
    assert enc.resizer.in_features == 1152 and enc.resizer.out_features == 256
