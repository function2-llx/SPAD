import torch

from pumit.ucpt.seg import SPADNeck, FusionEncoder, SemanticHead
from pumit.ucpt.seg.pos_embed import sine_pos_embed_3d


def _run(da, d_in):
    neck = SPADNeck()
    fusion = FusionEncoder()
    head = SemanticHead()
    x = torch.randn(1, 1024, d_in, 24, 24)
    levels = neck(x, da)
    l16 = levels['1/16']
    b, c, d, h, w = l16.shape
    tokens = l16.flatten(2).transpose(1, 2)
    pe = sine_pos_embed_3d(d, h, w, c, torch.device('cpu'))
    text = torch.randn(1, 1, 256)
    fused = fusion(tokens, text, pe, None).transpose(1, 2).view(b, c, d, h, w)
    out = head(fused, levels, text, None, da)
    return out, levels['1/4'].shape[2]


def test_da0_output_isotropic():
    out, l4_depth = _run(da=0, d_in=4)
    assert out.shape[2] == l4_depth


def test_da3_output_depth_frozen():
    out, l4_depth = _run(da=3, d_in=12)
    assert out.shape[2] == l4_depth
    assert out.shape[-2:] == (96, 96)
