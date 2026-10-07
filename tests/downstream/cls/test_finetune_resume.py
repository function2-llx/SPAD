"""Exercise update-boundary recovery through the real optimizer and DDP loop."""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import random
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
from pumit.downstream.cls.optim import build_warmup_cosine


class InterruptedRun(Exception):
    pass


class StochasticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 5)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        features = self.dropout(self.proj(x.flatten(1)))
        return features, features.unsqueeze(1)


def _arrays(flag, size, split, root=None):
    count = {'train': 11, 'val': 5, 'test': 3}[split]
    images = np.arange(count * 8, dtype=np.uint8).reshape(count, 2, 2, 2)
    return images, np.arange(count, dtype=np.int64) % 2


def _transform(raw, *, is_3d):
    # Exercise Python and NumPy streams in addition to model dropout and sampler RNG.
    scale = 0.8 + 0.1 * random.random() + 0.1 * np.random.random()
    return {'x': torch.from_numpy(raw.copy()).float() / 32 * scale}


def _models(seed=13):
    finetune.seed_process(seed)
    encoder, head = StochasticEncoder(), torch.nn.Linear(5, 2)
    if dist.is_initialized():
        encoder, head = DDP(encoder), DDP(head)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=0.003, weight_decay=0.01,
    )
    scheduler = build_warmup_cosine(optimizer, total_steps=6, warmup_steps=2)
    return encoder, head, optimizer, scheduler


def _fit(models, path):
    encoder, head, optimizer, scheduler = models

    def evaluate(logits, labels, flag, size, split, root=None):
        # Select a non-final checkpoint while retaining an output-dependent metric.
        return {
            'auc': 1 - scheduler.last_epoch / 10,
            'acc': float(logits.softmax(-1)[:, 1].mean()),
        }

    return finetune.fit(
        encoder, head, SimpleNamespace(flag='mock', size=2, is_3d=True, n_classes=2),
        total_steps=6, eval_every=2, eval_batch_size=2, augment=True, accum_steps=2,
        device='cpu', batch_size=8, transform_batch=_transform, opt=optimizer,
        sched=scheduler, data_root='.', evaluate_fn=evaluate, progress=False,
        seed=7, checkpoint_path=path,
    )


def _assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected, strict=True):
            _assert_same(item, reference)
    else:
        assert actual == expected


def _random_draw():
    return (random.random(), np.random.random(), torch.rand(5))


def _interrupt_after_interval(models, path):
    original_train = finetune.train_steps
    calls = 0

    def train(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # An unfinished interval must be replayed from its preceding checkpoint.
            partial = list(args)
            partial[5] = 1
            original_train(*partial, **kwargs)
            raise InterruptedRun()
        original_train(*args, **kwargs)

    with patch.object(finetune, 'train_steps', train), pytest.raises(InterruptedRun):
        _fit(models, path)


def _check_resume(directory):
    reference_path = directory / 'reference.pt'
    resumed_path = directory / 'resumed.pt'
    with (
        patch.object(finetune, 'build_arrays', _arrays),
        patch.object(torch, 'autocast', lambda **kwargs: nullcontext()),
    ):
        reference_models = _models()
        reference = _fit(reference_models, reference_path)
        reference_rng = _random_draw()
        _interrupt_after_interval(_models(), resumed_path)
        interrupted = torch.load(resumed_path, weights_only=True)
        assert interrupted['step'] == 2
        assert [entry['step'] for entry in interrupted['history']] == [2]

        # Reconstructing a process consumes unrelated RNG before loading the checkpoint.
        resumed_models = _models(seed=991)
        resumed = _fit(resumed_models, resumed_path)
        _assert_same(_random_draw(), reference_rng)
        assert resumed == reference
        assert resumed.best_epoch == 0
        assert [entry['step'] for entry in resumed.history] == [2, 4, 6]
        for actual, expected in zip(resumed_models[:2], reference_models[:2], strict=True):
            _assert_same(actual.state_dict(), expected.state_dict())
        _assert_same(resumed_models[2].state_dict(), reference_models[2].state_dict())
        _assert_same(resumed_models[3].state_dict(), reference_models[3].state_dict())
        _assert_same(
            torch.load(resumed_path, weights_only=True),
            torch.load(reference_path, weights_only=True),
        )

        finished_models = _models(seed=997)
        with patch.object(finetune, 'train_steps', side_effect=AssertionError('already finished')):
            finished = _fit(finished_models, resumed_path)
        assert finished == reference
        for actual, expected in zip(finished_models[:2], reference_models[:2], strict=True):
            _assert_same(actual.state_dict(), expected.state_dict())


def test_single_process_resume_preserves_training_and_best_checkpoint(tmp_path):
    bars = []

    class Progress:
        def __init__(self, **kwargs):
            self.total = kwargs['total']
            self.initial = self.n = kwargs['initial']
            self.closed = False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self, count):
            self.n += count

        def set_postfix(self, **kwargs):
            pass

    with patch.object(finetune, 'tqdm', Progress):
        _check_resume(tmp_path)
    assert [(bar.initial, bar.n, bar.total) for bar in bars] == [
        (0, 6, 6), (0, 3, 6), (2, 6, 6), (6, 6, 6),
    ]
    assert all(bar.closed for bar in bars)


def _ddp_worker(rank, rendezvous, directory, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', rank=rank, world_size=2, init_method=f'file://{rendezvous}',
        timeout=timedelta(seconds=40),
    )
    try:
        if mode == 'equivalence':
            _check_resume(Path(directory))
        else:
            with (
                patch.object(finetune, 'build_arrays', _arrays),
                patch.object(torch, 'autocast', lambda **kwargs: nullcontext()),
            ):
                _interrupt_after_interval(_models(seed=991), Path(directory) / 'cross-world.pt')
    finally:
        dist.destroy_process_group()


def _spawn_check(tmp_path, mode):
    processes = mp.spawn(
        _ddp_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path), mode),
        nprocs=2, join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('distributed resume checks exceeded 90 seconds')
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_ddp_resume_preserves_each_rank_rng_and_optimizer(tmp_path):
    _spawn_check(tmp_path, 'equivalence')


def test_resume_can_change_between_single_process_and_ddp(tmp_path):
    path = tmp_path / 'cross-world.pt'
    with (
        patch.object(finetune, 'build_arrays', _arrays),
        patch.object(torch, 'autocast', lambda **kwargs: nullcontext()),
    ):
        _interrupt_after_interval(_models(), path)
        assert torch.load(path, weights_only=True)['step'] == 2
        _spawn_check(tmp_path, 'cross-world')
        assert torch.load(path, weights_only=True)['step'] == 4
        run = _fit(_models(seed=991), path)
        assert [entry['step'] for entry in run.history] == [2, 4, 6]
        assert torch.load(path, weights_only=True)['step'] == 6
