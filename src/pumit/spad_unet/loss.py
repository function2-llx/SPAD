"""Shared partial-label loss for Universal SPAD U-Net systems."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn import functional as F

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


SAMPLE_MEAN_LOSS_NORMALIZATION = 'mean_dataset_local_losses_global_samples'
REGION_BALANCED_LOSS_NORMALIZATION = 'mean_active_regions_global_batch'
SAMPLE_LEVEL_LOSS_NORMALIZATIONS = (
    SAMPLE_MEAN_LOSS_NORMALIZATION,
)
SUPPORTED_LOSS_NORMALIZATIONS = (
    *SAMPLE_LEVEL_LOSS_NORMALIZATIONS,
    REGION_BALANCED_LOSS_NORMALIZATION,
)
SIGMOID_REGIONS = 'sigmoid_regions'
SOFTMAX_LABELS = 'softmax_labels'
SUPPORTED_PREDICTION_MODES = (
    SIGMOID_REGIONS,
    SOFTMAX_LABELS,
)


@dataclass(frozen=True)
class UniversalPartialLabelBatch:
    """Per-sample targets and their active rows in the universal output bank."""

    targets: list[torch.Tensor]
    region_indices: list[torch.Tensor]

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
    ) -> 'UniversalPartialLabelBatch':
        return UniversalPartialLabelBatch(
            [
                target.to(device, non_blocking=non_blocking)
                for target in self.targets
            ],
            [
                indices.to(device, non_blocking=non_blocking)
                for indices in self.region_indices
            ],
        )


_deep_supervision_weights_cache: dict[tuple[int, torch.device, bool], torch.Tensor] = {}


def deep_supervision_weights(
    num_levels: int,
    device: torch.device,
    keep_last_for_ddp: bool = False,
) -> torch.Tensor:
    """Return nnU-Net v2 deep-supervision weights."""
    if num_levels < 1:
        raise ValueError('num_levels must be positive')
    key = (num_levels, device, keep_last_for_ddp)
    if key not in _deep_supervision_weights_cache:
        # torch.tensor(list, device='cuda') syncs the stream (pageable H2D), so build once.
        weights = torch.tensor(
            [1 / (2 ** level) for level in range(num_levels)],
            device=device,
        )
        weights[-1] = 1e-6 if keep_last_for_ddp and num_levels > 1 else 0
        if num_levels == 1:
            weights[0] = 1
        _deep_supervision_weights_cache[key] = weights / weights.sum()
    return _deep_supervision_weights_cache[key]


def select_sample_region_logits(
    canonical_logits: list[torch.Tensor],
    sample_index: int,
    region_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract one sample's active-region logits from full-bank batch output.

    Args:
        canonical_logits: Deep supervision outputs, each [B, C_full, *spatial].
        sample_index: Which batch item to extract.
        region_indices: 1D LongTensor of active channel indices.

    Returns:
        List of [1, C_active, *spatial_at_level] tensors, one per DS level.
    """
    return [
        level[sample_index : sample_index + 1].index_select(1, region_indices)
        for level in canonical_logits
    ]


class UniversalPartialLabelLoss(torch.nn.Module):
    """DC+BCE with deep supervision for the Universal partial-label setting.

    Shared interface:
    - `sample_loss(logits, target)`: computes DS-weighted loss for one sample
      from already-active logits under the selected normalization.
    - `mean_sample_loss(sample_losses, active_region_counts)`: aggregates local
      per-sample losses into the DDP-correct global mean.
    - `forward(canonical_logits, targets)`: routes a
      `UniversalPartialLabelBatch` from the full canonical bank, calls
      both methods, and returns the mean. Used by Corpus-grid batched forwarding.
    """

    def __init__(
        self,
        batch_dice: bool = False,
        keep_last_for_ddp: bool = False,
        loss_normalization: str = SAMPLE_MEAN_LOSS_NORMALIZATION,
        global_sample_count: int | None = None,
        global_active_region_count: int | None = None,
        ddp_world_size: int = 1,
    ):
        super().__init__()
        if loss_normalization not in SUPPORTED_LOSS_NORMALIZATIONS:
            raise ValueError(
                f'unsupported loss normalization {loss_normalization!r}; '
                f'expected one of {SUPPORTED_LOSS_NORMALIZATIONS}'
            )
        if global_sample_count is not None and global_sample_count < 1:
            raise ValueError('global_sample_count must be positive')
        if (
            global_active_region_count is not None
            and global_active_region_count < 1
        ):
            raise ValueError('global_active_region_count must be positive')
        if (
            global_sample_count is not None
            and global_active_region_count is not None
        ):
            raise ValueError(
                'only one fixed global normalization count may be provided'
            )
        if (
            loss_normalization in SAMPLE_LEVEL_LOSS_NORMALIZATIONS
            and global_active_region_count is not None
        ):
            raise ValueError(
                'sample-level loss requires global_sample_count'
            )
        if (
            loss_normalization == REGION_BALANCED_LOSS_NORMALIZATION
            and global_sample_count is not None
        ):
            raise ValueError(
                'region-balanced loss requires global_active_region_count'
            )
        if ddp_world_size < 1:
            raise ValueError('ddp_world_size must be positive')
        fixed_global_count = (
            global_sample_count
            if loss_normalization in SAMPLE_LEVEL_LOSS_NORMALIZATIONS
            else global_active_region_count
        )
        if fixed_global_count is None and ddp_world_size != 1:
            raise ValueError(
                'ddp_world_size requires a fixed global normalization count'
            )
        self.keep_last_for_ddp = keep_last_for_ddp
        self.loss_normalization = loss_normalization
        self.global_sample_count = global_sample_count
        self.global_active_region_count = global_active_region_count
        self.ddp_world_size = ddp_world_size
        self.criterion = DC_and_BCE_loss(
            {},
            {
                'batch_dice': batch_dice,
                'do_bg': True,
                'smooth': 1e-5,
                'ddp': False,
            },
            use_ignore_label=False,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.label_criterion = DC_and_CE_loss(
            {
                'batch_dice': batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': False,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    @staticmethod
    def _foreground_channels_to_labels(target: torch.Tensor) -> torch.Tensor:
        """Convert mutually exclusive foreground channels to local class labels."""
        class_values = torch.arange(
            1,
            target.shape[1] + 1,
            dtype=torch.long,
            device=target.device,
        ).view(1, -1, *([1] * (target.ndim - 2)))
        return (target.long() * class_values).sum(dim=1, keepdim=True)

    def sample_loss(
        self,
        active_logits: list[torch.Tensor],
        target: torch.Tensor,
        prediction_mode: str = SIGMOID_REGIONS,
    ) -> torch.Tensor:
        """Compute DS-weighted loss for one sample from already-active logits.

        Args:
            active_logits: Deep supervision outputs for one sample with only its
                active channels, each [1, C_active, *spatial_at_level].
            target: Foreground target channels [1, C_foreground, *full_spatial].
            prediction_mode: Dataset-local prediction and loss semantics.

        Returns:
            Scalar per-sample loss. Sample-level normalization preserves the
            dataset-local criterion. Region-balanced normalization restores
            the corresponding active-region sum.
        """
        if prediction_mode not in SUPPORTED_PREDICTION_MODES:
            raise ValueError(
                f'unsupported prediction mode {prediction_mode!r}; '
                f'expected one of {SUPPORTED_PREDICTION_MODES}'
            )
        if (
            prediction_mode == SOFTMAX_LABELS
            and self.loss_normalization not in SAMPLE_LEVEL_LOSS_NORMALIZATIONS
        ):
            raise ValueError(
                'softmax label loss requires sample-level normalization'
            )
        expected_channels = target.shape[1] + (
            1 if prediction_mode == SOFTMAX_LABELS else 0
        )
        if active_logits[0].shape[1] != expected_channels:
            raise ValueError(
                f'{prediction_mode} logits have {active_logits[0].shape[1]} '
                f'channels for {target.shape[1]} foreground target channels'
            )

        device = active_logits[0].device
        num_levels = len(active_logits)
        weights = deep_supervision_weights(
            num_levels,
            device,
            self.keep_last_for_ddp,
        )

        level_losses = []
        for level_index, level_logits in enumerate(active_logits):
            if (
                level_index == num_levels - 1
                and num_levels > 1
                and not self.keep_last_for_ddp
            ):
                level_losses.append(torch.zeros((), device=device))
                continue
            if level_logits.shape[2:] == target.shape[2:]:
                level_target = target
            else:
                level_target = F.interpolate(
                    target.float(),
                    size=level_logits.shape[2:],
                    mode='nearest-exact',
                )
            if prediction_mode == SOFTMAX_LABELS:
                level_target = self._foreground_channels_to_labels(level_target)
                level_loss = self.label_criterion(level_logits, level_target)
            else:
                level_loss = self.criterion(level_logits, level_target)
            if self.loss_normalization == REGION_BALANCED_LOSS_NORMALIZATION:
                level_loss = level_loss * level_logits.shape[1]
            level_losses.append(level_loss)
        return (weights * torch.stack(level_losses)).sum()

    def mean_sample_loss(
        self,
        sample_losses: list[torch.Tensor],
        active_region_counts: list[int] | None = None,
    ) -> torch.Tensor:
        """Normalize local sample losses over the global logical batch.

        DDP averages gradients across ranks. Scaling each local sum by
        ``world_size / global_count`` gives the same gradient and logged loss
        as a single-process global mean over datasets or active regions.
        """
        if not sample_losses:
            raise ValueError('sample_losses must not be empty')
        if active_region_counts is not None:
            if len(sample_losses) != len(active_region_counts):
                raise ValueError(
                    f'sample loss/count mismatch: losses={len(sample_losses)}, '
                    f'counts={len(active_region_counts)}'
                )
            if any(count < 1 for count in active_region_counts):
                raise ValueError(
                    'every sample must have at least one active region'
                )

        local_sum = torch.stack(sample_losses).sum()
        fixed_global_count = (
            self.global_sample_count
            if self.loss_normalization in SAMPLE_LEVEL_LOSS_NORMALIZATIONS
            else self.global_active_region_count
        )
        if fixed_global_count is not None:
            return (
                local_sum
                * self.ddp_world_size
                / fixed_global_count
            )

        if self.loss_normalization in SAMPLE_LEVEL_LOSS_NORMALIZATIONS:
            local_count = len(sample_losses)
        else:
            if active_region_counts is None:
                raise ValueError(
                    'region-balanced loss requires active_region_counts when '
                    'no fixed global count is configured'
                )
            local_count = sum(active_region_counts)
        global_count = local_sum.new_tensor(float(local_count))
        world_size = 1
        if dist.is_initialized():
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        return local_sum * world_size / global_count

    def forward(
        self,
        canonical_logits: torch.Tensor | list[torch.Tensor],
        targets: UniversalPartialLabelBatch,
    ) -> torch.Tensor:
        """Compute mean partial-label loss over the batch from full canonical logits.

        Args:
            canonical_logits: Full canonical-bank output or DS outputs.
            targets: Native-trainer-compatible partial-label targets.

        Returns:
            Scalar loss (mean over samples in the global logical batch).
        """
        if isinstance(canonical_logits, torch.Tensor):
            canonical_logits = [canonical_logits]
        batch_size = canonical_logits[0].shape[0]

        if (
            len(targets.targets) != batch_size
            or len(targets.region_indices) != batch_size
        ):
            raise ValueError(
                f'batch_size mismatch: logits B={batch_size}, '
                f'targets={len(targets.targets)}, '
                f'indices={len(targets.region_indices)}'
            )

        sample_losses = []
        active_region_counts = []
        for i in range(batch_size):
            sample_logits = select_sample_region_logits(
                canonical_logits, i, targets.region_indices[i]
            )
            sample_losses.append(self.sample_loss(sample_logits, targets.targets[i]))
            active_region_counts.append(int(targets.region_indices[i].numel()))

        return self.mean_sample_loss(sample_losses, active_region_counts)
