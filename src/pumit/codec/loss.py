"""Codec loss: L1 reconstruction + optional KL divergence."""

from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class CodecOutput:
    recon: Tensor
    mean: Tensor
    logvar: Tensor | None = None


def kl_divergence(mean: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence from N(mean, exp(logvar)) to N(0, 1), averaged over all elements."""
    return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


class CodecLoss(nn.Module):
    """Combined codec loss: L1 + optional KL.

    When output.logvar is None (deterministic AE), KL is skipped.
    """

    def __init__(self, l1_weight: float = 1.0, kl_weight: float = 1e-6, smooth_l1_beta: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.kl_weight = kl_weight
        self.smooth_l1_beta = smooth_l1_beta

    def forward(self, x: Tensor, output: CodecOutput) -> dict[str, Tensor]:
        l1 = nn.functional.smooth_l1_loss(output.recon, x, beta=self.smooth_l1_beta)
        loss = self.l1_weight * l1
        result = {'l1': l1}
        if output.logvar is not None:
            kl = kl_divergence(output.mean, output.logvar)
            result['kl'] = kl
            loss = loss + self.kl_weight * kl
        result['loss'] = loss
        return result


VAELoss = CodecLoss
