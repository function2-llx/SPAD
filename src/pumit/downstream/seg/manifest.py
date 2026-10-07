"""Shared manifest schema for the downstream segmentation queues.

A manifest is a YAML list of independent experiments, one mapping each.

An optional ``fold`` selects a split other than fold 0: an integer index into ``splits_final.json`` or the ``name`` of a custom entry there (label-fraction splits), which trains into ``fold_<name>``.
A frozen arm with no pretrained trunk (random init) states ``weights: null`` explicitly; an absent key is an error.

An optional ``num_gpus`` fixes the DDP width; pending jobs in a queue must all specify the same width or all leave it unset.
When unset, :func:`adaptive_num_gpus` derives the width from pending jobs and available GPUs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DATASET_NAME_PATTERN = re.compile(r'Dataset[0-9]{3}_.+')
# Safe as a fold_<fold> path component; 'all' is rejected separately because it folds val into train.
FOLD_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
DEFAULT_FOLD = '0'
DEFAULT_TRAINER = 'DownstreamSegTrainer'
TRAINERS = {DEFAULT_TRAINER, 'nnUNetTrainerRetainCheckpoints'}
DEFAULT_COMPILE_MODE = 'reduce-overhead'
COMPILE_MODES = {'default', DEFAULT_COMPILE_MODE, 'max-autotune'}
DEFAULT_COMPILE_DYNAMIC = 'false'
COMPILE_DYNAMIC_VALUES = {DEFAULT_COMPILE_DYNAMIC, 'true', 'auto'}
# nnU-Net announces each epoch as "<timestamp>: Epoch <n>"; "Epoch time: ..." must not match.
EPOCH_PATTERN = re.compile(r':\s*Epoch (\d+)\s*$', re.MULTILINE)


@dataclass(frozen=True)
class Job:
    name: str
    dataset: str
    plan: str
    weights: Path | None
    fold: str = DEFAULT_FOLD
    trainer: str = DEFAULT_TRAINER
    compile_mode: str = DEFAULT_COMPILE_MODE
    compile_dynamic: str = DEFAULT_COMPILE_DYNAMIC
    # Explicit width when configured; otherwise resolved by the queue.
    num_gpus: int | None = None

    @property
    def launch_args(self) -> tuple[str, ...]:
        args = (self.dataset, self.plan)
        if self.weights is not None:
            args += ('--weights', str(self.weights))
        if self.fold != DEFAULT_FOLD:
            args += ('--fold', self.fold)
        if self.trainer != DEFAULT_TRAINER:
            args += ('--trainer', self.trainer)
        args += ('--compile-mode', self.compile_mode)
        if self.compile_dynamic != DEFAULT_COMPILE_DYNAMIC:
            args += ('--compile-dynamic', self.compile_dynamic)
        if self.num_gpus is not None and self.num_gpus != 1:
            args += ('--num-gpus', str(self.num_gpus))
        return args

    @property
    def plan_path(self) -> Path:
        return Path(os.environ['nnUNet_preprocessed']) / self.dataset / f'{self.plan}.json'

    @property
    def global_batch_size(self) -> int:
        """Plan batch size, which bounds the DDP world size nnU-Net will accept."""
        configuration = json.loads(self.plan_path.read_text())['configurations']['3d_fullres']
        return int(configuration['batch_size'])

    @property
    def run_dir(self) -> Path:
        results_root = Path(os.environ['nnUNet_results'])
        return (
            results_root
            / self.dataset
            / f'{self.trainer}__{self.plan}__3d_fullres'
            / f'fold_{self.fold}'
        )

    @property
    def done(self) -> Path:
        """Validation summary; ``checkpoint_final.pth`` predates it and is not a completion marker."""
        return self.run_dir / 'validation/summary.json'

    def retained_checkpoint(self, epoch: int) -> Path:
        """Immutable periodic checkpoint, the only safe input for validating a still-training arm."""
        return self.run_dir / f'checkpoint_epoch_{epoch:04d}.pth'

    def retained_epochs(self) -> set[int]:
        return {
            int(path.stem.split('_')[-1])
            for path in self.run_dir.glob('checkpoint_epoch_*.pth')
        }

    def validation_dir(self, epoch: int) -> Path:
        return self.run_dir / 'periodic_validation' / f'checkpoint_epoch_{epoch:04d}'


def load_jobs(path: Path) -> list[Job]:
    records = yaml.safe_load(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f'{path}: expected a non-empty YAML list')

    jobs = []
    names = set()
    run_dirs = set()
    expected_fields = {
        'name',
        'dataset',
        'plan',
        'weights',
        'fold',
        'trainer',
        'compile_mode',
        'compile_dynamic',
        'num_gpus',
    }
    for index, record in enumerate(records, start=1):
        context = f'{path}: job {index}'
        if not isinstance(record, dict):
            raise ValueError(f'{context}: expected a mapping')
        unexpected_fields = record.keys() - expected_fields
        if unexpected_fields:
            raise ValueError(f'{context}: unexpected fields {sorted(unexpected_fields)}')
        missing_fields = {'name', 'dataset', 'plan'} - record.keys()
        if missing_fields:
            raise ValueError(f'{context}: missing fields {sorted(missing_fields)}')
        name = record['name']
        dataset = record['dataset']
        plan = record['plan']
        weights = record.get('weights')
        fold = record.get('fold', DEFAULT_FOLD)
        trainer = record.get('trainer', DEFAULT_TRAINER)
        compile_mode = record.get('compile_mode', DEFAULT_COMPILE_MODE)
        compile_dynamic = record.get('compile_dynamic', DEFAULT_COMPILE_DYNAMIC)
        num_gpus = record.get('num_gpus')
        if num_gpus is not None and (
            isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1
        ):
            raise ValueError(f'{context}: num_gpus must be a positive integer')
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f'{context}: name must be a non-empty filename')
        if name in names:
            raise ValueError(f'{context}: duplicate job name {name!r}')
        if not isinstance(dataset, str) or DATASET_NAME_PATTERN.fullmatch(dataset) is None:
            raise ValueError(f'{context}: dataset must be a full nnU-Net dataset name')
        if not isinstance(plan, str) or not plan or '__' in plan:
            raise ValueError(f'{context}: plan must be a valid nnU-Net plans identifier')
        if weights is not None and (not isinstance(weights, str) or not weights):
            raise ValueError(f'{context}: weights must be a non-empty path when provided')
        if isinstance(fold, int) and not isinstance(fold, bool):
            fold = str(fold)
        if (
            not isinstance(fold, str)
            or FOLD_PATTERN.fullmatch(fold) is None
            or fold == 'all'
        ):
            raise ValueError(
                f'{context}: fold must be a split index or the name of a splits_final.json entry'
            )
        if trainer not in TRAINERS:
            raise ValueError(f'{context}: unsupported trainer {trainer!r}')
        if trainer == DEFAULT_TRAINER and 'weights' not in record:
            # An explicit `weights: null` marks a deliberately weightless arm (random-init trunk).
            raise ValueError(f'{context}: {DEFAULT_TRAINER} requires weights (or an explicit `weights: null`)')
        if trainer == 'nnUNetTrainerRetainCheckpoints' and weights is not None:
            raise ValueError(f'{context}: {trainer} trains from scratch without weights')
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f'{context}: unsupported compile_mode {compile_mode!r}')
        if compile_dynamic not in COMPILE_DYNAMIC_VALUES:
            raise ValueError(
                f'{context}: unsupported compile_dynamic {compile_dynamic!r}'
            )
        job = Job(
            name=name,
            dataset=dataset,
            plan=plan,
            weights=Path(weights) if weights is not None else None,
            fold=fold,
            trainer=trainer,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
            num_gpus=num_gpus,
        )
        if job.run_dir in run_dirs:
            raise ValueError(f'{context}: duplicate output path {job.run_dir}')
        names.add(name)
        run_dirs.add(job.run_dir)
        jobs.append(job)

    return jobs


def adaptive_num_gpus(job_count: int, visible: int, max_world_size: int) -> int:
    """Choose one DDP width for a manifest whose jobs leave ``num_gpus`` unset.

    Widen by powers of two until the pending jobs can occupy the node, so a width always divides a
    power-of-two GPU count and no devices are stranded. The ceiling is the smaller of the visible count and
    the plans' global batch size, because nnU-Net asserts ``global_batch_size >= world_size`` and would
    otherwise abort as soon as few enough jobs remain for the rule to reach for a wide split.
    """
    if job_count <= 0:
        raise ValueError(f'job_count must be positive, got {job_count}')
    if visible <= 0:
        raise ValueError(f'visible must be positive, got {visible}')
    if max_world_size <= 0:
        raise ValueError(f'max_world_size must be positive, got {max_world_size}')
    ceiling = min(visible, max_world_size)
    width = 1
    while width * job_count < visible and width * 2 <= ceiling:
        width *= 2
    return width


def latest_epoch(run_dir: Path) -> int:
    """Return the highest epoch any training log announces, or 0 before the run reaches one.

    nnU-Net writes one timestamped log per restart with unpadded month and day fields, so the names
    do not sort chronologically and the newest run is not always the last file. Take the maximum
    across all of them; resumes only ever move forward, so the maximum is the current epoch.
    """
    return max(
        (
            int(epoch)
            for log_path in run_dir.glob('training_log_*.txt')
            for epoch in EPOCH_PATTERN.findall(log_path.read_text(errors='replace'))
        ),
        default=0,
    )
