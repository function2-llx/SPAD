from torch import nn

from .conv import Conv3d, TransposedConv3d


def _get_act_layer(name: str) -> nn.Module:
    act_map = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU, 'leaky_relu': nn.LeakyReLU}
    if name.lower() not in act_map:
        raise ValueError(f'Unsupported activation: {name!r}. Supported: {list(act_map)}')
    return act_map[name.lower()]()

__all__ = [
    'PatchEmbed',
    'InversePatchEmbed',
]

class PatchEmbed(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        kernel_size: int,
        hierarchical: bool,
        act: str = 'gelu',
        last_norm: bool = False,
    ):
        """
        Args:
            kernel_size: convolution kernel size in hierarchical structure
        """
        assert patch_size & patch_size - 1 == 0, 'only power of 2 is supported'
        if hierarchical:
            # references:
            # - [Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881)
            # - [Three things everyone should know about Vision Transformers](https://arxiv.org/abs/2203.09795)
            # - [Unleashing Vanilla Vision Transformer with Masked Image Modeling for Object Detection](https://arxiv.org/abs/2204.02964)
            # - [All in Tokens: Unifying Output Space of Visual Tasks via Soft Token](https://arxiv.org/abs/2301.02229)
            padding = kernel_size - 1 >> 1
            num_downsamples = patch_size.bit_length() - 1
            super().__init__(
                *[
                    nn.Sequential(
                        Conv3d(
                            in_channels, out_channels >> num_downsamples - i - 1,
                            kernel_size, 2, padding,
                        ) if i == 0 else Conv3d(
                            out_channels >> num_downsamples - i,
                            out_channels >> num_downsamples - i - 1,
                            kernel_size, 2, padding,
                        ),
                        nn.GroupNorm(1, out_channels >> num_downsamples - i - 1),
                        _get_act_layer(act),
                    )
                    for i in range(num_downsamples)
                ],
                Conv3d(out_channels, out_channels, 1),
            )
        else:
            super().__init__(Conv3d(in_channels, out_channels, patch_size, patch_size))
        if last_norm:
            self.append(nn.GroupNorm(1, out_channels))

class InversePatchEmbed(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        kernel_size: int,
        hierarchical: bool,
        act: str = 'gelu',
    ):
        assert patch_size & patch_size - 1 == 0, 'only power of 2 is supported'
        # a normalization is assumed to precede current module
        if hierarchical:
            num_upsamples = patch_size.bit_length() - 1
            super().__init__(
                *[
                    nn.Sequential(
                        TransposedConv3d(in_channels >> i, in_channels >> i + 1, kernel_size, 2),
                        nn.GroupNorm(1, in_channels >> i + 1),
                        _get_act_layer(act),
                    )
                    for i in range(num_upsamples - 1)
                ],
                TransposedConv3d(in_channels >> num_upsamples - 1, out_channels, kernel_size, 2),
            )
        else:
            super().__init__(
                TransposedConv3d(in_channels, out_channels, patch_size, patch_size)
            )
