"""Run the Corpus-grid Universal DDP training-path preflight."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from nnunetv2.run.run_training import get_trainer_from_args

from pumit.spad_unet.loss import REGION_BALANCED_LOSS_NORMALIZATION
from pumit.spad_unet.universal import (
    CORPUS_GRID_UNIVERSAL_PLANS_IDENTIFIER,
    UniversalResidualEncoderUNet,
)


UNIVERSAL_DATASET_ID = os.environ.get(
    'CORPUS_GRID_PREFLIGHT_DATASET_ID',
    '591',
)


def main() -> None:
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    initialize_start = time.perf_counter()
    trainer = get_trainer_from_args(
        UNIVERSAL_DATASET_ID,
        '3d_fullres',
        0,
        'CorpusGridUniversalTrainer',
        os.environ.get(
            'CORPUS_GRID_PREFLIGHT_PLANS',
            CORPUS_GRID_UNIVERSAL_PLANS_IDENTIFIER,
        ),
        False,
    )
    trainer.initialize()
    initialize_seconds = time.perf_counter() - initialize_start

    if torch.get_autocast_dtype('cuda') is not torch.bfloat16:
        raise TypeError('Corpus-grid trainer did not select CUDA BF16 autocast')
    if trainer.grad_scaler is not None:
        raise TypeError('Corpus-grid BF16 training must not use GradScaler')

    network = trainer.network.module
    if not isinstance(network, UniversalResidualEncoderUNet):
        raise TypeError(f'unexpected network type: {type(network).__name__}')
    # The loss stays eager; only the shape-static components compile.
    for component in ('encoder', 'task_aware_bottleneck', 'decoder'):
        if getattr(network, component)._compiled_call_impl is None:
            raise TypeError(f'Corpus-grid {component} is not compiled')
    loss_module = network.partial_label_loss
    if loss_module.loss_normalization != REGION_BALANCED_LOSS_NORMALIZATION:
        raise ValueError('Corpus-grid preflight requires region-balanced loss')
    planned_active_region_count = (
        trainer.configuration_manager.network_arch_init_kwargs[
            'num_active_regions_per_global_batch'
        ]
    )
    if loss_module.global_active_region_count != planned_active_region_count:
        raise ValueError('Corpus-grid active-region normalization count is incorrect')

    reader = trainer.replay_reader
    global_batch_size = trainer.configuration_manager.batch_size
    records_per_cycle = math.lcm(
        2 * reader.logical_batch_size,
        global_batch_size,
    )
    steps_per_cycle = records_per_cycle // global_batch_size
    record_offset = rank * trainer.batch_size

    train_loader, val_loader = trainer.get_dataloaders()
    first_parameter = next(network.parameters())
    before = first_parameter.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    step_reports = []
    flat_replay: list[tuple[str, bool]] = []
    num_steps = int(
        os.environ.get('CORPUS_GRID_PREFLIGHT_STEPS', str(steps_per_cycle))
    )
    # Group coverage and complementary foreground are only decidable over whole group pairs.
    if num_steps < steps_per_cycle or num_steps % steps_per_cycle:
        raise ValueError(
            f'CORPUS_GRID_PREFLIGHT_STEPS must be a positive multiple of '
            f'{steps_per_cycle} for global batch {global_batch_size}, got {num_steps}'
        )
    for step in range(num_steps):
        expected_records = reader.records_at(
            global_batch_size * step + record_offset,
            trainer.batch_size,
        )
        batch = next(train_loader)
        expected_dataset_ids = tuple(
            record.dataset_id for record in expected_records
        )
        if tuple(batch['dataset_ids']) != expected_dataset_ids:
            raise ValueError('dataloader batch does not match replay records')

        local_replay = tuple(
            (record.dataset_id, record.force_foreground)
            for record in expected_records
        )
        gathered_replay = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_replay, local_replay)
        global_replay = tuple(
            item
            for rank_replay in gathered_replay
            for item in rank_replay
        )
        global_dataset_ids = tuple(
            dataset_id for dataset_id, _ in global_replay
        )
        foreground_set = {
            dataset_id
            for dataset_id, force_foreground in global_replay
            if force_foreground
        }
        flat_replay.extend(global_replay)

        started = time.perf_counter()
        result = trainer.train_step(batch)
        torch.cuda.synchronize()
        loss = float(result['loss'])
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError(f'non-finite train loss: {loss}')
        gradients = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise FloatingPointError('missing or non-finite gradients')
        step_reports.append({
            'step': step,
            'seconds': time.perf_counter() - started,
            'loss': loss,
            'datasets': global_dataset_ids,
            'forced_foreground_datasets': sorted(foreground_set),
        })

    replay_groups = [
        flat_replay[offset:offset + reader.logical_batch_size]
        for offset in range(0, len(flat_replay), reader.logical_batch_size)
    ]
    group_foreground_sets = []
    for group in replay_groups:
        if sorted(dataset_id for dataset_id, _ in group) != sorted(trainer.datasets):
            raise ValueError('a replay group does not cover all datasets exactly once')
        group_foreground_sets.append({
            dataset_id for dataset_id, force_foreground in group if force_foreground
        })
    for first, second in zip(
        group_foreground_sets[::2],
        group_foreground_sets[1::2],
        strict=True,
    ):
        if first & second:
            raise ValueError(
                'paired replay groups do not use complementary foreground sets'
            )
        if first | second != set(trainer.datasets):
            raise ValueError(
                'paired replay groups do not cover all foreground datasets'
            )
    if torch.equal(before, first_parameter):
        raise RuntimeError('optimizer steps did not update the first parameter')

    checkpoint_path = (
        Path(os.environ['CORPUS_GRID_PREFLIGHT_DIR'])
        / 'checkpoint_smoke.pth'
    )
    if rank == 0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    trainer.save_checkpoint(str(checkpoint_path))
    dist.barrier()
    saved_parameter = first_parameter.detach().clone()
    with torch.no_grad():
        first_parameter.add_(1)
    trainer.load_checkpoint(str(checkpoint_path))
    if trainer.current_epoch != 1:
        raise ValueError(
            f'checkpoint restored epoch {trainer.current_epoch}, expected 1'
        )
    if not torch.equal(saved_parameter, first_parameter):
        raise ValueError('checkpoint did not restore the model parameter')

    report = {
        'rank': rank,
        'logical_device': local_rank,
        'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank],
        'autocast_dtype': str(torch.get_autocast_dtype('cuda')),
        'loss_normalization': loss_module.loss_normalization,
        'global_active_region_count': loss_module.global_active_region_count,
        'global_batch_size': global_batch_size,
        'local_batch_size': trainer.batch_size,
        'steps_per_replay_cycle': steps_per_cycle,
        'compiled_components': [
            name
            for name in ('encoder', 'task_aware_bottleneck', 'decoder')
            if getattr(network, name)._compiled_call_impl is not None
        ],
        'initialize_seconds': initialize_seconds,
        'steps': step_reports,
        'gradient_parameter_count': len(gradients),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'checkpoint_resume_epoch': trainer.current_epoch,
    }
    print(json.dumps(report, sort_keys=True), flush=True)

    for loader in (train_loader, val_loader):
        finish = getattr(loader, '_finish', None)
        if finish is not None:
            finish()
    dist.barrier()
    if rank == 0:
        checkpoint_path.unlink()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
