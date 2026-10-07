"""SPAD KL-VAE: 3D VAE with spatially adaptive convolutions.

Mirrors the FLUX.1 / SD3 AutoencoderKL architecture, replacing Conv2d with
SPAD Conv3d for anisotropy-aware 3D medical image compression.

State dict keys match the diffusers AutoencoderKL layout so that pre-trained
2D weights load directly via SPAD Conv3d weight inflation.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as ckpt_utils
from xformers.ops import memory_efficient_attention
torch.compiler.allow_in_graph(memory_efficient_attention)

from pumit import spadop
from .loss import CodecOutput


def _encoder_da_schedule(
    da: int | None,
    num_blocks: int = 4,
) -> list[int | None]:
    """Pre-compute the DA value at each encoder stage.

    Returns a list of length num_blocks. For blocks with a downsampler,
    the DA value is for BOTH the resnets and the downsampler in that block.
    DA decrements after each downsampler (when da > 0).
    The final block (no downsampler) gets the post-decrement value.
    """
    schedule = []
    for i in range(num_blocks):
        schedule.append(da)
        if i < num_blocks - 1 and da is not None and da > 0:
            da -= 1
    return schedule


def _decoder_da_schedule(
    da: int | None,
    num_blocks: int = 4,
) -> tuple[list[int | None], list[bool]]:
    """Pre-compute the DA value and upsample-depth flag at each decoder stage.

    Returns:
        da_schedule: list of DA values per block
        upsample_depth: list of bools per block (whether upsampler should upsample depth)
    """
    nds = 0 if da is None else max(3 - min(da, 3), 0)
    da_schedule = []
    upsample_depth = []
    for i in range(num_blocks):
        da_schedule.append(da)
        if i < num_blocks - 1:
            if nds > 0:
                upsample_depth.append(True)
                nds -= 1
            else:
                upsample_depth.append(False)
                if da is not None:
                    da += 1
    return da_schedule, upsample_depth


class ResnetBlock(nn.Module):
    """Pre-activation ResNet block: norm->act->conv->norm->act->conv + shortcut.

    Matches diffusers ResnetBlock2D state_dict keys:
        norm1, conv1, norm2, conv2, [conv_shortcut]
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = spadop.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = spadop.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = f'{prefix}conv_shortcut.weight'
        if key in state_dict and state_dict[key].ndim == 4:
            state_dict[key] = state_dict[key].unsqueeze(2)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h, da)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h, da)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Self-attention block at mid resolution.

    Matches diffusers Attention state_dict keys:
        group_norm, to_q, to_k, to_v, to_out.0
    """

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.group_norm = nn.GroupNorm(groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.group_norm(x)
        b, c, *spatial = x.shape
        h = self.num_heads
        k = c // h
        n = 1
        for s in spatial:
            n *= s
        x = x.reshape(b, c, n).permute(0, 2, 1)
        q = self.to_q(x).reshape(b, n, h, k)
        kk = self.to_k(x).reshape(b, n, h, k)
        v = self.to_v(x).reshape(b, n, h, k)
        x = memory_efficient_attention(q, kk, v)
        x = x.reshape(b, n, c)
        x = self.to_out[0](x)
        x = x.permute(0, 2, 1).reshape(b, c, *spatial)
        return x + residual


class Downsample(nn.Module):
    """Spatial downsampling via stride-2 conv.

    Matches diffusers Downsample2D state_dict key: conv
    SPAD Conv3d handles anisotropy: skips depth downsampling when DA > 0.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = spadop.Conv3d(channels, channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        if da is None or da > 0:
            x = F.pad(x, (0, 1, 0, 1))
        else:
            x = F.pad(x, (0, 1, 0, 1, 0, 1))
        return self.conv(x, da)


class Upsample(nn.Module):
    """Spatial upsampling via interpolation + conv.

    Matches diffusers Upsample2D state_dict key: conv
    Uses nearest-neighbor interpolation followed by stride-1 conv.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = spadop.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, da: int | None, upsample_depth: bool) -> torch.Tensor:
        if upsample_depth:
            x = F.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode='nearest')
        return self.conv(x, da)


class DownEncoderBlock(nn.Module):
    """Encoder block: resnets + optional downsampler.

    Matches diffusers DownEncoderBlock2D state_dict keys:
        resnets.{i}.{...}, downsamplers.0.{...}
    """

    def __init__(self, in_channels: int, out_channels: int, num_resnets: int = 2, add_downsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_resnets):
            self.resnets.append(ResnetBlock(in_channels if i == 0 else out_channels, out_channels))
        self.downsamplers = nn.ModuleList([Downsample(out_channels)]) if add_downsample else None

    def forward(self, x: torch.Tensor, da: int | None, *, grad_ckpt: bool = False) -> torch.Tensor:
        for resnet in self.resnets:
            x = ckpt_utils.checkpoint(resnet, x, da, use_reentrant=False) if grad_ckpt else resnet(x, da)
        if self.downsamplers is not None:
            x = ckpt_utils.checkpoint(self.downsamplers[0], x, da, use_reentrant=False) if grad_ckpt else self.downsamplers[0](x, da)
        return x


class UpDecoderBlock(nn.Module):
    """Decoder block: resnets + optional upsampler.

    Matches diffusers UpDecoderBlock2D state_dict keys:
        resnets.{i}.{...}, upsamplers.0.{...}
    """

    def __init__(self, in_channels: int, out_channels: int, num_resnets: int = 3, add_upsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_resnets):
            self.resnets.append(ResnetBlock(in_channels if i == 0 else out_channels, out_channels))
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if add_upsample else None

    def forward(
        self,
        x: torch.Tensor,
        da: int | None,
        upsample_depth: bool = False,
        *,
        grad_ckpt: bool = False,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            x = ckpt_utils.checkpoint(resnet, x, da, use_reentrant=False) if grad_ckpt else resnet(x, da)
        if self.upsamplers is not None:
            x = (
                ckpt_utils.checkpoint(self.upsamplers[0], x, da, upsample_depth, use_reentrant=False)
                if grad_ckpt
                else self.upsamplers[0](x, da, upsample_depth)
            )
        return x


class MidBlock(nn.Module):
    """Mid block: resnet + attention + resnet.

    Matches diffusers UNetMidBlock2D state_dict keys:
        resnets.{0,1}.{...}, attentions.0.{...}
    """

    def __init__(self, channels: int):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(channels, channels), ResnetBlock(channels, channels)])
        self.attentions = nn.ModuleList([AttnBlock(channels)])

    def forward(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        x = self.resnets[0](x, da)
        x = self.attentions[0](x)
        x = self.resnets[1](x, da)
        return x


class Encoder(nn.Module):
    """VAE Encoder. Mirrors diffusers Encoder state_dict keys."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        grad_ckpt: bool = False,
        use_quant_conv: bool = False,
    ):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.conv_in = spadop.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch_in = block_out_channels[0]
        for i, ch_out in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock(ch_in, ch_out, num_resnets=layers_per_block, add_downsample=not is_last)
            )
            ch_in = ch_out

        self.mid_block = MidBlock(block_out_channels[-1])
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = spadop.Conv3d(block_out_channels[-1], 2 * latent_channels, kernel_size=3, padding=1)
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = f'{prefix}quant_conv.weight'
        if key in state_dict and state_dict[key].ndim == 4:
            state_dict[key] = state_dict[key].unsqueeze(2)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        ckpt = self.training and self.grad_ckpt
        da_schedule = _encoder_da_schedule(da, num_blocks=len(self.down_blocks))
        x = self.conv_in(x, da)
        for block, block_da in zip(self.down_blocks, da_schedule):
            x = block(x, block_da, grad_ckpt=ckpt)
        final_da = da_schedule[-1]
        x = self.mid_block(x, final_da)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x, final_da)
        if self.quant_conv is not None:
            x = self.quant_conv(x)
        return x


class Decoder(nn.Module):
    """VAE Decoder. Mirrors diffusers Decoder state_dict keys."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        grad_ckpt: bool = False,
        use_post_quant_conv: bool = False,
    ):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = spadop.Conv3d(latent_channels, reversed_channels[0], kernel_size=3, padding=1)
        self.mid_block = MidBlock(reversed_channels[0])

        self.up_blocks = nn.ModuleList()
        ch_in = reversed_channels[0]
        for i, ch_out in enumerate(reversed_channels):
            is_last = i == len(reversed_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock(ch_in, ch_out, num_resnets=layers_per_block + 1, add_upsample=not is_last)
            )
            ch_in = ch_out

        self.conv_norm_out = nn.GroupNorm(32, reversed_channels[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = spadop.Conv3d(reversed_channels[-1], out_channels, kernel_size=3, padding=1)

    def _output_block(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x, da)
        return x

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = f'{prefix}post_quant_conv.weight'
        if key in state_dict and state_dict[key].ndim == 4:
            state_dict[key] = state_dict[key].unsqueeze(2)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        ckpt = self.training and self.grad_ckpt
        if self.post_quant_conv is not None:
            x = self.post_quant_conv(x)
        da_schedule, upsample_depth = _decoder_da_schedule(da, num_blocks=len(self.up_blocks))
        x = self.conv_in(x, da)
        x = self.mid_block(x, da)
        for i, block in enumerate(self.up_blocks):
            ud = upsample_depth[i] if i < len(upsample_depth) else False
            x = block(x, da_schedule[i], upsample_depth=ud, grad_ckpt=ckpt)
        final_da = da_schedule[-1]
        x = ckpt_utils.checkpoint(self._output_block, x, final_da, use_reentrant=False) if ckpt else self._output_block(x, final_da)
        return x


class SPADKLVAE(nn.Module):
    """SPAD KL-VAE: 3D VAE with spatially adaptive convolutions.

    Mirrors FLUX.1 / SD3 AutoencoderKL architecture. Loads pre-trained 2D
    weights via SPAD Conv3d weight inflation.

    State dict keys match diffusers AutoencoderKL:
        encoder.{...}, decoder.{...}
    No quant_conv / post_quant_conv (following FLUX.1 convention).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, block_out_channels, layers_per_block, grad_ckpt)
        self.decoder = Decoder(out_channels, latent_channels, block_out_channels, layers_per_block, grad_ckpt)

    def encode(self, x: torch.Tensor, da: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x, da=da)
        mean, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)
        return mean, logvar

    def decode(self, z: torch.Tensor, da: int | None) -> torch.Tensor:
        return self.decoder(z, da=da)

    def forward(
        self,
        x: torch.Tensor,
        da: int | None,
        da_dec: int | None,
    ) -> CodecOutput:
        """Full forward: encode -> sample -> [cross-DA resize] -> decode.

        Args:
            x: Input image tensor (B, C, D, H, W).
            da: Depth anisotropy level for the encoder. None for 2D.
            da_dec: Decoder DA for cross-DA training. None for 2D.

        Returns CodecOutput(recon, mean, logvar).
        """
        assert (da is None) == (da_dec is None)
        if da != da_dec:
            assert da is not None and da_dec is not None
            assert abs(da - da_dec) <= 1
        mean, logvar = self.encode(x, da=da)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        if da != da_dec:
            nds_enc = 3 - min(da, 3)
            nds_dec = 3 - min(da_dec, 3)
            if nds_enc != nds_dec:
                scale_d = 2.0 if da_dec > da else 0.5
                new_d = int(z.shape[2] * scale_d)
                z = F.interpolate(
                    z,
                    size=(new_d, z.shape[3], z.shape[4]),
                    mode='trilinear',
                    align_corners=False,
                )

        recon = self.decode(z, da=da_dec)
        return CodecOutput(recon=recon, mean=mean, logvar=logvar)
