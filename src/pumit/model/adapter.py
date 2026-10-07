from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.utils import checkpoint

from pumit import spadop
from .vit import ViT

class SimpleViTAdapter(ViT):
    def __init__(
        self, *,
        out_indexes: Sequence[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        dim = self.embed_dim
        patch_size = self.patch_embed.patch_size
        if self.patch_embed.adaptive:
            assert patch_size[0] == patch_size[1] == patch_size[2] == 16
            self.fpn = nn.ModuleList([
                nn.Sequential(
                    spadop.TransposedConv3d(dim, dim >> 1, 2, 2),
                    nn.InstanceNorm3d(dim >> 1, affine=True),
                    nn.LeakyReLU(),
                    spadop.TransposedConv3d(dim >> 1, dim >> 2, 2, 2),
                ),
                spadop.TransposedConv3d(dim, dim >> 1, 2, 2),
                nn.Identity(),
                spadop.MaxPool(2),
            ])
        else:
            assert patch_size[1] == patch_size[2] == 16
            assert patch_size[0] & patch_size[0] - 1 == 0
            aniso_d = max(0, (16 // patch_size[0]).bit_length() - 1)
            get_args = lambda i: ((1 if aniso_d >= i else 2, 2, 2), ) * 2
            self.fpn = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose3d(dim, dim, *get_args(4)),
                    nn.InstanceNorm3d(dim, affine=True),
                    nn.GELU(),
                    nn.ConvTranspose3d(dim, dim, *get_args(3)),
                ),
                nn.ConvTranspose3d(dim, dim, *get_args(4)),
                nn.Identity(),
                nn.MaxPool3d(*get_args(5)),
            ])

        self.out_indexes = out_indexes
        assert len(out_indexes) == len(self.fpn)
        # the backbone is pre-norm
        self.norms = nn.ModuleList([nn.InstanceNorm3d(dim, affine=True) for _ in range(len(out_indexes))])

    def forward(self, x: torch.Tensor):
        patch_tokens = self.patch_embed(x)
        shape = patch_tokens.shape[2:]
        d, h, w = shape
        patch_tokens += spadop.resample(self.pos_embed, shape)
        patch_tokens = self.pos_drop(einops.rearrange(patch_tokens, 'n c ... -> n (...) c'))
        B = patch_tokens.shape[0]
        prefix = [self.cls_token.expand(B, -1, -1)]
        if self.n_register_tokens > 0:
            prefix.append(self.register_tokens.expand(B, -1, -1))
        x = torch.cat([*prefix, patch_tokens], dim=1)
        rope_cos, rope_sin = self.rope.compute(shape)
        n_prefix = self.n_prefix
        states = []
        for block in self.blocks:
            if self.training and self.grad_ckpt:
                x = checkpoint.checkpoint(block, x, rope_cos, rope_sin, n_prefix, use_reentrant=False)
            else:
                x = block(x, rope_cos, rope_sin, n_prefix)
            states.append(x)
        ret = []
        for out_id, norm, fpn in zip(self.out_indexes, self.norms, self.fpn):
            feature_map = einops.rearrange(states[out_id][:, n_prefix:], 'n (d h w) c -> n c d h w', d=d, h=h, w=w)
            ret.append(fpn(norm(feature_map)))
        return ret
