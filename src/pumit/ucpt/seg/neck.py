"""SPAD parallel-scale neck: 1/16 ViT map -> {1/4, 1/8, 1/16} pyramid (256-d).

Mirrors Sam3VisionNeck's first 3 branches (the 1/32 maxpool branch is dropped).
Depth scaling follows a per-stage SPAD schedule resolved from the input DA.
"""

import torch
from torch import nn, Tensor

from pumit import spadop
from pumit.ucpt.seg.schedule import StageDA, neck_da_schedule


class SPADNeck(nn.Module):
    """Build a three-level SPAD feature pyramid from the ViT patch map."""

    def __init__(self, in_channels: int = 1024, hidden_size: int = 256):
        """
        Args:
            in_channels: ViT patch-feature width.
            hidden_size: Pyramid feature width.
        """
        super().__init__()
        # --- 4.0 branch (1/4): 2 upsample stages ---
        self.up40_0 = spadop.SPADConvTranspose3d_K2S2(in_channels, in_channels // 2)
        self.act40 = nn.GELU()
        self.up40_1 = spadop.SPADConvTranspose3d_K2S2(in_channels // 2, hidden_size)
        self.proj40_1 = spadop.Conv3d(hidden_size, hidden_size, kernel_size=1).to(
            memory_format=torch.channels_last_3d,
        )
        self.proj40_2 = spadop.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        # --- 2.0 branch (1/8): 1 upsample stage ---
        self.up20_0 = spadop.SPADConvTranspose3d_K2S2(in_channels, in_channels // 2)
        self.proj20_1 = spadop.Conv3d(in_channels // 2, hidden_size, kernel_size=1).to(
            memory_format=torch.channels_last_3d,
        )
        self.proj20_2 = spadop.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        # --- 1.0 branch (1/16): no upsample ---
        self.proj10_1 = spadop.Conv3d(in_channels, hidden_size, kernel_size=1).to(
            memory_format=torch.channels_last_3d,
        )
        self.proj10_2 = spadop.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, x: Tensor, da: int | None) -> dict[str, Tensor]:
        """Build the 1/4, 1/8, and 1/16 feature levels.

        Args:
            x: ViT patch map at 1/16 scale.
            da: Input SPAD depth-adaptation level, or ``None`` for 2D.

        Returns:
            Feature maps keyed by pyramid scale.
        """
        s0, s1 = neck_da_schedule(da)
        # 1/16 branch: no upsample
        # proj10_1 is 1x1 (plain nn.Conv3d) -- no da argument
        l16 = self.proj10_2(self.proj10_1(x), da=s0.da)
        # 1/8 branch: 1 upsample stage
        h = self.up20_0(x, da=_up_da(s0))
        # proj20_1 is 1x1 (plain nn.Conv3d) -- no da argument
        l8 = self.proj20_2(self.proj20_1(h), da=s0.da)
        # 1/4 branch: 2 upsample stages
        h = self.act40(self.up40_0(x, da=_up_da(s0)))
        h = self.up40_1(h, da=_up_da(s1))
        # proj40_1 is 1x1 (plain nn.Conv3d) -- no da argument
        l4 = self.proj40_2(self.proj40_1(h), da=s1.da)
        return {'1/4': l4, '1/8': l8, '1/16': l16}


def _up_da(stage: StageDA) -> int | None:
    """K2S2 da: 0 -> double depth (3D), >=1 -> preserve depth. Driven by upsample_depth."""
    if stage.da is None:
        return None
    return 0 if stage.upsample_depth else 1
