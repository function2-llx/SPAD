from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import numpy as np
import torch

import pumit.downstream.cls.finetune as finetune


def test_train_steps_accumulation_matches_full_batch(monkeypatch):
    """accum_steps splits the SAME optimizer-step batch; the updates must match bit-for-bit-ish."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (10, 3), dtype=np.uint8)
    labels = rng.integers(0, 2, 10).astype(np.int64)
    monkeypatch.setattr(finetune, 'build_arrays',
                        lambda flag, size, split, root=None: (images, labels))

    def transform(raw, *, is_3d):
        return {'x': torch.from_numpy(raw.astype(np.float32))}

    class StubEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 5)

        def forward(self, x):
            g = self.proj(x)
            return g, g.unsqueeze(1)

    def run(accum_steps):
        torch.manual_seed(0)
        encoder, head = StubEncoder(), torch.nn.Linear(5, 2)
        opt = torch.optim.SGD([*encoder.parameters(), *head.parameters()], lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        generator = torch.Generator()
        generator.manual_seed(7)
        progress = Mock()
        finetune.train_steps(encoder, head, SimpleNamespace(flag='mock', size=1, is_3d=True),
                             'cpu', 4, 3, transform, opt, sched, generator,
                             accum_steps=accum_steps, progress_bar=progress)
        assert progress.update.call_count == 3
        assert all(call.args == (1,) for call in progress.update.call_args_list)
        return [p.detach().clone() for p in [*encoder.parameters(), *head.parameters()]]

    for full, accumulated in zip(run(1), run(2), strict=True):
        torch.testing.assert_close(full, accumulated)


def test_fit_records_test_diagnostics_but_selects_only_on_validation(monkeypatch):
    encoder = torch.nn.Linear(1, 1, bias=False)
    head = torch.nn.Linear(1, 1, bias=False)
    val_auc = iter([0.2, 0.8, 0.3])
    train_epochs = []
    train_orders = []
    evaluated_splits = []

    def fake_train_steps(encoder, head, spec, device, batch_size, steps, transform_batch,
                         opt, sched, shuffle_generator, **kwargs):
        assert steps == 1, 'eval_every=1 means one optimizer step between validations'
        train_epochs.append(len(train_epochs))
        train_orders.append(torch.randperm(6, generator=shuffle_generator).tolist())
        encoder.weight.data.fill_(len(train_epochs))

    def fake_eval_split(encoder, head, spec, split, *args, **kwargs):
        evaluated_splits.append(split)
        return torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)

    test_auc = iter([0.9, 0.4, 0.95])

    def fake_evaluate(logits, labels, flag, size, split, root=None):
        if split == 'val':
            return {'auc': next(val_auc), 'acc': 0.5}
        if split == 'test':
            return {'auc': next(test_auc), 'acc': 0.6}
        raise AssertionError(split)

    monkeypatch.setattr(finetune, 'train_steps', fake_train_steps)
    monkeypatch.setattr(finetune, 'eval_split', fake_eval_split)

    run = finetune.fit(
        encoder,
        head,
        SimpleNamespace(flag='mock', size=1),
        total_steps=3,
        eval_every=1,
        device='cpu',
        batch_size=1,
        transform_batch=None,
        opt=object(),
        sched=object(),
        data_root='.',
        evaluate_fn=fake_evaluate,
        progress=False,
        seed=7,
    )

    assert train_epochs == [0, 1, 2]
    expected_generator = torch.Generator().manual_seed(7)
    expected_orders = [torch.randperm(6, generator=expected_generator).tolist() for _ in range(3)]
    assert train_orders == expected_orders
    assert evaluated_splits == ['val', 'test', 'val', 'test', 'val', 'test']
    assert run.best_epoch == 1
    assert run.best_metrics == {'auc': 0.8, 'acc': 0.5}
    assert encoder.weight.item() == 2.0
    assert [record['test_auc'] for record in run.history] == [0.9, 0.4, 0.95]
    assert [record['test_acc'] for record in run.history] == [0.6, 0.6, 0.6]


def test_lr_backbone_defaults_to_the_base_lr():
    """Omitting --lr-backbone ties the head to the backbone's top block, as MAE/BEiT do."""
    parser = finetune.build_arg_parser()
    base = ['--backbone', 'ucpt', '--datasets', 'd', '--flag', 'f', '--out', 'o']
    assert finetune.resolve_lrs(parser.parse_args([*base, '--lr', '1e-4'])) == (1e-4, 1e-4)
    assert finetune.resolve_lrs(parser.parse_args([*base, '--lr', '1e-3', '--lr-backbone', '1e-5'])) == (1e-5, 1e-3)


def test_the_old_split_lr_flags_are_gone():
    """The protocol's 100x split is now written as --lr 1e-3 --lr-backbone 1e-5."""
    parser = finetune.build_arg_parser()
    base = ['--backbone', 'ucpt', '--datasets', 'd', '--flag', 'f', '--out', 'o']
    for flag in ('--lr-encoder', '--lr-head'):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, flag, '1e-5'])


@pytest.mark.parametrize('drop_path', [0.0, 0.1])
def test_drop_path_reaches_non_spad_backbone_factory(monkeypatch, drop_path):
    received = {}

    def create_model(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(
        finetune, 'BACKBONES', {'test': SimpleNamespace(create_model=create_model)},
    )
    args = SimpleNamespace(
        backbone='test', vit_config=None, drop_path=drop_path,
        img_size=192, weights=None, dims=3,
    )
    finetune.build_trainable_encoder(args, 'cpu')
    assert received['drop_path_rate'] == drop_path
    assert received['trainable'] is True


def test_omitted_drop_path_preserves_factory_default(monkeypatch):
    received = {}

    def create_model(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(
        finetune, 'BACKBONES', {'test': SimpleNamespace(create_model=create_model)},
    )
    args = SimpleNamespace(
        backbone='test', vit_config=None, drop_path=None,
        img_size=192, weights=None, dims=3,
    )
    finetune.build_trainable_encoder(args, 'cpu')
    assert 'drop_path_rate' not in received
