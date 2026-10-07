"""Seven-stage specialist controls for tied and Cross-DA SPAD training."""

from __future__ import annotations

import torch
from nnunetv2.utilities.helpers import dummy_context
from torch.amp import autocast

from pumit.spad_unet.architecture import SPADResEncUNet
from pumit.spad_unet.geometry import select_da_pair
from pumit.spad_unet.experiments.spad import SPADUNetTrainer, _downsample_targets


class SPADCrossDAResEncUNet(SPADResEncUNet):
    """Expose independent encoder and decoder DA states through ``forward``."""

    def forward(
        self,
        x: torch.Tensor,
        da_encoder: int | None = None,
        da_decoder: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if da_encoder is None:
            da_encoder = self.inference_da
        return self.forward_sample(x, da_encoder, da_decoder)


class SPADTiedDASpecialistTrainer(SPADUNetTrainer):
    """Named tied-DA control for the seven-stage specialist comparison."""


class SPADCrossDASpecialistTrainer(SPADUNetTrainer):
    """Train a single-dataset specialist with independent Cross-DA sampling."""

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target_full = target[0].to(self.device, non_blocking=True)
        else:
            target_full = target.to(self.device, non_blocking=True)

        da_encoder, da_decoder = select_da_pair(
            self.continuous_da,
            cross=True,
        )

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == 'cuda'
            else dummy_context()
        ):
            output = self.network(data, da_encoder, da_decoder)
            if isinstance(output, list):
                loss = self.loss(output, _downsample_targets(target_full, output))
            else:
                loss = self.loss(output, target_full)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': loss.detach().cpu().numpy()}
