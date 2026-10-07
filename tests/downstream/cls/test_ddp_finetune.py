"""CPU multi-process checks for the global-batch finetune contract."""

from contextlib import nullcontext
from datetime import timedelta
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from pumit.downstream.cls import finetune


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 5)

    def forward(self, x):
        features = self.proj(x.flatten(1))
        return features, features.unsqueeze(1)


def _transform(raw, *, is_3d):
    return {'x': torch.from_numpy(raw.copy()).float() / 32}


def _models():
    torch.manual_seed(13)
    return TinyEncoder(), torch.nn.Linear(5, 2)


def _optimizer(encoder, head):
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=0.003, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    return optimizer, scheduler


def _parameters(encoder, head):
    return [p.detach().clone() for p in [*encoder.parameters(), *head.parameters()]]


def _worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    images = np.arange(11 * 8, dtype=np.uint8).reshape(11, 2, 2, 2)
    labels = np.arange(11, dtype=np.int64) % 2
    spec = SimpleNamespace(flag='mock', size=2, is_3d=True, n_classes=2)

    def arrays(flag, size, split, root=None):
        n = {'train': 11, 'val': 5, 'test': 1}[split]
        return images[:n], labels[:n]

    with (
        patch.object(finetune, 'build_arrays', arrays),
        patch.object(torch, 'autocast', lambda **kwargs: nullcontext()),
    ):
        references = {}
        for augment in (False, True):
            encoder, head = _models()
            optimizer, scheduler = _optimizer(encoder, head)
            finetune.train_steps(
                encoder, head, spec, 'cpu', world_size * 4, 3, _transform,
                optimizer, scheduler, torch.Generator().manual_seed(7), augment=augment,
            )
            references[augment] = _parameters(encoder, head)

        dist.init_process_group(
            'gloo', rank=rank, world_size=world_size,
            init_method=f'file://{rendezvous}', timeout=timedelta(seconds=40),
        )
        try:
            for augment in (False, True):
                for accum_steps in (1, 2):
                    raw_encoder, raw_head = _models()
                    encoder, head = DDP(raw_encoder), DDP(raw_head)
                    optimizer, scheduler = _optimizer(encoder, head)
                    with (
                        patch.object(encoder, 'no_sync', wraps=encoder.no_sync) as encoder_no_sync,
                        patch.object(head, 'no_sync', wraps=head.no_sync) as head_no_sync,
                    ):
                        finetune.train_steps(
                            encoder, head, spec, 'cpu', world_size * 4, 3, _transform,
                            optimizer, scheduler, torch.Generator().manual_seed(7),
                            augment=augment, accum_steps=accum_steps,
                        )
                        assert encoder_no_sync.call_count == 3 * (accum_steps - 1)
                        assert head_no_sync.call_count == 3 * (accum_steps - 1)
                    assert scheduler.last_epoch == 3
                    for actual, expected in zip(
                        _parameters(encoder, head), references[augment], strict=True,
                    ):
                        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)

            for split in ('val', 'test'):
                logits, actual_labels = finetune.eval_split(
                    encoder, head, spec, split, 'cpu', 2, _transform,
                )
                raw_images, expected_labels = arrays('mock', 2, split)
                with torch.no_grad():
                    features, _ = raw_encoder(**_transform(raw_images, is_3d=True))
                    expected_logits = raw_head(features)
                torch.testing.assert_close(logits, expected_logits)
                torch.testing.assert_close(actual_labels, torch.from_numpy(expected_labels))

            _check_fit(encoder, head, spec, rank)
        finally:
            dist.destroy_process_group()


def _check_fit(encoder, head, spec, rank):
    train_calls = []
    metric_calls = []

    def train(*args, **kwargs):
        train_calls.append(1)
        with torch.no_grad():
            encoder.module.proj.bias.fill_(len(train_calls))

    def evaluate(logits, labels, flag, size, split, root=None):
        assert rank == 0, 'only rank zero should compute dataset metrics'
        metric_calls.append(split)
        auc = [0.9, 0.6][len(train_calls) - 1] if split == 'val' else 0.5
        return {'auc': auc, 'acc': 0.5}

    with patch.object(finetune, 'train_steps', train):
        run = finetune.fit(
            encoder, head, spec, total_steps=2, eval_every=1, eval_batch_size=2,
            device='cpu', batch_size=dist.get_world_size() * 4, transform_batch=_transform,
            opt=None, sched=None, data_root='.', evaluate_fn=evaluate, progress=False,
        )
    assert run.best_epoch == 0
    assert run.best_metrics['auc'] == 0.9
    assert [entry['val_auc'] for entry in run.history] == [0.9, 0.6]
    assert metric_calls == (['val', 'test', 'val', 'test'] if rank == 0 else [])
    torch.testing.assert_close(encoder.module.proj.bias, torch.ones(5))


@pytest.mark.parametrize('world_size', [2, 3])
def test_distributed_finetune_global_batch_eval_and_selection(tmp_path, world_size):
    processes = mp.spawn(
        _worker, args=(world_size, str(tmp_path / 'rendezvous')),
        nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('distributed finetune checks exceeded 90 seconds')
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_main_requires_local_rank_to_enable_ddp(monkeypatch):
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.delenv('LOCAL_WORLD_SIZE', raising=False)
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('WORLD_SIZE', '3')
    monkeypatch.setattr(
        'sys.argv',
        ['finetune', '--backbone', 'voco-l', '--datasets', 'unused', '--flag', 'mock', '--out', 'unused'],
    )
    calls = []
    monkeypatch.setattr(finetune, '_run', lambda args: calls.append(finetune._rank_world()))
    with patch.object(dist, 'init_process_group') as initialize:
        finetune.main()
    initialize.assert_not_called()
    assert calls == [(0, 1)]


def test_main_rejects_cross_node_torchrun(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '2')
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setattr(
        'sys.argv',
        ['finetune', '--backbone', 'voco-l', '--datasets', 'unused', '--flag', 'mock', '--out', 'unused'],
    )
    with pytest.raises(ValueError, match='single node'):
        finetune.main()


def test_global_batch_is_not_silently_changed_for_gpu_count(monkeypatch):
    monkeypatch.setattr(finetune, '_rank_world', lambda: (0, 3))
    with pytest.raises(ValueError, match='global batch_size 32'):
        finetune._run(SimpleNamespace(batch_size=32, accum_steps=1))
