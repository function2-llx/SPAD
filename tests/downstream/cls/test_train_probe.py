import sys

import numpy as np
import orjson
import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pumit.downstream.cls.train_probe as train_probe
from pumit.downstream.cls.metrics import fallback_metrics
from pumit.downstream.cls.train_probe import train_one_seed


def legacy_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Pre-refactor MedMNIST loop retained as a deterministic parity oracle."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = torch.nn.Linear(train_x.shape[1], 2)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = train_x.shape[0]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * ((n_train + batch_size - 1) // batch_size))
    best_auc = -1.0
    best_metrics = None
    best_state = None
    for _ in range(epochs):
        head.train()
        permutation = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = permutation[start:start + batch_size]
            loss = torch.nn.functional.cross_entropy(head(train_x[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        head.eval()
        with torch.no_grad():
            metrics = fallback_metrics(head(val_x), val_y.numpy(), n_classes=2)
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_metrics = metrics
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_metrics


def test_classification_probe_matches_pre_refactor_loop():
    train_x = torch.tensor([
        [-2.0, 0.0],
        [-1.0, 0.5],
        [1.0, -0.5],
        [2.0, 0.0],
    ])
    train_y = torch.tensor([0, 0, 1, 1])
    val_x = torch.tensor([[-1.5, 0.25], [-0.5, 0.0], [0.5, 0.0], [1.5, -0.25]])
    val_y = torch.tensor([0, 0, 1, 1])
    kwargs = {
        'epochs': 4,
        'lr': 0.05,
        'weight_decay': 0.01,
        'batch_size': 2,
        'seed': 7,
    }

    legacy_head, legacy_metrics = legacy_linear_probe(train_x, train_y, val_x, val_y, **kwargs)
    run = train_one_seed(
        train_x,
        train_y,
        val_x,
        val_y,
        n_classes=2,
        head_type='linear',
        device='cpu',
        progress=False,
        **kwargs,
    )

    assert run.best_metrics == pytest.approx(legacy_metrics)
    for key, expected in legacy_head.state_dict().items():
        torch.testing.assert_close(run.model.state_dict()[key], expected)


@pytest.mark.parametrize('fixed_updates', [False, True])
def test_classification_probe_cli_preserves_legacy_fields_and_adds_selection_schema(tmp_path, monkeypatch, fixed_updates):
    features_dir = tmp_path / 'features'
    features_dir.mkdir()
    labels = torch.tensor([0, 0, 1, 1])
    for split, offset in [('train', 0.0), ('val', 0.1), ('test', 0.2)]:
        global_features = torch.tensor([
            [-2.0 + offset, 0.0],
            [-1.0 + offset, 0.5],
            [1.0 + offset, -0.5],
            [2.0 + offset, 0.0],
        ])
        torch.save(
            {
                'native': global_features,
                'label': labels,
            },
            features_dir / f'{split}.pt',
        )

    monkeypatch.setattr(
        train_probe,
        'evaluate',
        lambda logits, target, flag, size, split, *, root: fallback_metrics(logits, target, n_classes=2),
    )
    output = tmp_path / 'probe.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_probe',
            '--features-dir',
            str(features_dir),
            '--flag',
            'mock',
            '--size',
            '1',
            '--n-classes',
            '2',
            '--head',
            'linear',
            *(['--updates', '5', '--eval-every', '2', '--batch-size', '3'] if fixed_updates else ['--epochs', '1']),
            '--seeds',
            '1',
            '--device',
            'cpu',
            '--out',
            str(output),
        ],
    )

    train_probe.main()

    result = orjson.loads(output.read_bytes())
    assert result['protocol'] == 'frozen_probe'
    assert not output.with_suffix('.heads.pt').exists(), 'head weights are opt-in'
    assert result['selection_policy'] == {'split': 'val', 'metric': 'auc', 'direction': 'maximize'}
    selection = result['per_seed_selection'][0]
    if fixed_updates:
        assert 'selected_epoch' not in selection
        assert selection['selected_update'] in [2, 4, 5]
        assert result['updates'] == 5
        assert result['eval_every'] == 2
        assert result['batch_size'] == 3
        assert [entry['update'] for entry in result['per_seed_history'][0]] == [2, 4, 5]
        assert all('metrics' in entry for entry in result['per_seed_history'][0])
    else:
        assert selection['selected_epoch'] == 0
        assert 'per_seed_history' not in result
        assert 'updates' not in result
    assert result['metrics']['val']['auc']['mean'] == result['val_auc_mean']
    assert result['metrics']['test']['auc']['mean'] == result['auc_mean']


@pytest.mark.parametrize(
    'budget_args',
    [
        ['--updates', '0', '--eval-every', '2'],
        ['--updates', '-1', '--eval-every', '2'],
        ['--updates', '1.5', '--eval-every', '2'],
        ['--updates', '5'],
        ['--updates', '5', '--eval-every', '0'],
        ['--epochs', '1', '--updates', '5', '--eval-every', '2'],
        ['--eval-every', '2'],
    ],
)
def test_fixed_update_cli_rejects_invalid_budget_before_loading_features(monkeypatch, budget_args):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_probe', '--features-dir', 'unused', '--flag', 'mock', '--size', '1',
            '--n-classes', '2', '--out', 'unused', *budget_args,
        ],
    )
    with pytest.raises(SystemExit) as error:
        train_probe.main()
    assert error.value.code == 2


def test_saved_probe_heads_round_trip_into_finetune(tmp_path, monkeypatch):
    """--save-heads must produce weights that finetune's --head-init accepts per seed."""
    import torch

    from pumit.downstream.cls.finetune import load_probe_head_state

    features_dir = tmp_path / 'features'
    features_dir.mkdir()
    labels = torch.tensor([0, 0, 1, 1])
    for split in ('train', 'val', 'test'):
        torch.save({'native': torch.randn(4, 3), 'label': labels}, features_dir / f'{split}.pt')

    monkeypatch.setattr(
        train_probe,
        'evaluate',
        lambda logits, target, flag, size, split, *, root: fallback_metrics(logits, target, n_classes=2),
    )
    output = tmp_path / 'probe.json'
    monkeypatch.setattr(sys, 'argv', [
        'train_probe', '--features-dir', str(features_dir), '--flag', 'mock', '--size', '1',
        '--n-classes', '2', '--head', 'mlp', '--epochs', '1', '--seeds', '2',
        '--device', 'cpu', '--out', str(output), '--save-heads',
    ])
    train_probe.main()

    heads = output.with_suffix('.heads.pt')
    state = load_probe_head_state(str(heads), 'mlp', 'mock', seed=1)
    head = train_probe.build_cls_head(3, 2, 'mlp')
    head.load_state_dict(state)                      # strict: shapes and keys must match
    assert state.keys() == head.state_dict().keys()
    assert load_probe_head_state(str(heads), 'mlp', 'mock', seed=0).keys() == state.keys()

    with pytest.raises(ValueError, match='holds .mlp. heads'):
        load_probe_head_state(str(heads), 'linear', 'mock', seed=0)
    with pytest.raises(ValueError, match='was fit on'):
        load_probe_head_state(str(heads), 'mlp', 'other', seed=0)
    with pytest.raises(ValueError, match='holds 2 seeds'):
        load_probe_head_state(str(heads), 'mlp', 'mock', seed=2)
