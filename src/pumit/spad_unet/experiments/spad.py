"""Single-dataset SPAD U-Net experiment integration for nnU-Net."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.spad_unet.geometry import (
    compute_continuous_da,
    decompose_continuous_da,
    sample_da,
)


def _downsample_targets(
    target: torch.Tensor,
    outputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    targets = []
    for output in outputs:
        if output.shape[2:] == target.shape[2:]:
            targets.append(target)
        else:
            targets.append(
                F.interpolate(
                    target.float(),
                    size=output.shape[2:],
                    mode='nearest-exact',
                ).to(target.dtype)
            )
    return targets


class SPADUNetTrainer(RetainPeriodicCheckpointsMixin, nnUNetTrainer):
    """nnU-Net trainer with SPAD architecture and stochastic DA.

    Key overrides:
    - train_step/validation_step: pass DA to the plans-defined SPAD network
    - Deep supervision: DA-dependent GT downsampling via stride schedule
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.continuous_da = compute_continuous_da(
            tuple(self.configuration_manager.spacing)
        )

    def _get_deep_supervision_scales(self):
        # Return None to disable pre-downsampling in the data pipeline.
        # We handle DS target computation per-batch in train_step since
        # stochastic DA changes spatial sizes each iteration.
        return None

    def _get_ddp_kwargs(self) -> dict:
        return {'find_unused_parameters': True}

    def _build_loss(self):
        loss = self._build_base_loss()

        if self.enable_deep_supervision:
            n_levels = (
                int(self.configuration_manager.network_arch_init_kwargs['n_stages']) - 1
            )
            weights = np.array([1 / (2**i) for i in range(n_levels)])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        # target is always full-res since we disabled pre-downsampling (_get_deep_supervision_scales returns None)
        if isinstance(target, list):
            target_full = target[0].to(self.device, non_blocking=True)
        else:
            target_full = target.to(self.device, non_blocking=True)

        da = sample_da(self.continuous_da)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == 'cuda'
            else dummy_context()
        ):
            output = self.network(data, da)
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

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target_full = target[0].to(self.device, non_blocking=True)
        else:
            target_full = target.to(self.device, non_blocking=True)

        da, _, _ = decompose_continuous_da(self.continuous_da)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == 'cuda'
            else dummy_context()
        ):
            output = self.network(data, da)
            if isinstance(output, list):
                loss = self.loss(output, _downsample_targets(target_full, output))
                output_fullres = output[0]
            else:
                loss = self.loss(output, target_full)
                output_fullres = output

        # Full-res output for online evaluation
        target_eval = target_full

        axes = [0] + list(range(2, output_fullres.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_fullres) > 0.5).long()
        else:
            output_seg = output_fullres.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output_fullres.shape, device=output_fullres.device, dtype=torch.float16
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_eval != self.label_manager.ignore_label).float()
                target_eval[target_eval == self.label_manager.ignore_label] = 0
            else:
                if target_eval.dtype == torch.bool:
                    mask = ~target_eval[:, -1:]
                else:
                    mask = 1 - target_eval[:, -1:]
                target_eval = target_eval[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target_eval, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }
