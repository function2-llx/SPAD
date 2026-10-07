"""SPAD FLUX.2 AE: 3D deterministic autoencoder with spatially adaptive convolutions.

Mirrors the FLUX.2 AutoEncoder architecture, replacing Conv2d with SPAD Conv3d.
Uses quant_conv/post_quant_conv bottleneck projections and deterministic encoding
(mean only, no reparameterization).

State dict keys match the diffusers AutoencoderKLFlux2 layout so that pre-trained
2D weights load directly via SPAD Conv3d weight inflation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from pumit.codec.loss import CodecOutput
from pumit.codec.flux1 import Encoder, Decoder


class SPADFlux2AE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        decoder_block_out_channels: tuple[int, ...] = (96, 192, 384, 384),
        layers_per_block: int = 2,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            latent_channels,
            block_out_channels,
            layers_per_block,
            grad_ckpt,
            use_quant_conv=True,
        )
        self.decoder = Decoder(
            out_channels,
            latent_channels,
            decoder_block_out_channels,
            layers_per_block,
            grad_ckpt,
            use_post_quant_conv=True,
        )

    def encode(self, x: torch.Tensor, da: int | None) -> torch.Tensor:
        h = self.encoder(x, da=da)
        mean = h.chunk(2, dim=1)[0]
        return mean

    def decode(self, z: torch.Tensor, da: int | None) -> torch.Tensor:
        return self.decoder(z, da=da)

    def forward(
        self,
        x: torch.Tensor,
        da: int | None,
        da_dec: int | None,
    ) -> CodecOutput:
        assert (da is None) == (da_dec is None)
        if da != da_dec:
            assert da is not None and da_dec is not None
            assert abs(da - da_dec) <= 1
        mean = self.encode(x, da=da)
        z = mean

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
        return CodecOutput(recon=recon, mean=mean)
