import sys

import orjson
import pytest
import torch

import pumit.downstream.spatial.train_spacing as train_spacing
from pumit.downstream.spatial.spacing import masked_l1_loss, train_spacing_probe


def test_spacing_probe_selects_minimum_validation_mae():
    train_x = torch.tensor([
        [-2.0, 0.0],
        [-1.0, 0.5],
        [1.0, -0.5],
        [2.0, 0.0],
    ])
    train_spacing = torch.stack((train_x[:, 0], train_x[:, 0] * 0.5, train_x[:, 1]), dim=1)
    val_x = torch.tensor([[-1.5, 0.25], [1.5, -0.25]])
    val_spacing = torch.stack((val_x[:, 0], val_x[:, 0] * 0.5, val_x[:, 1]), dim=1)

    run = train_spacing_probe(
        train_x,
        train_spacing,
        val_x,
        val_spacing,
        head_type='linear',
        epochs=5,
        lr=0.1,
        weight_decay=0.0,
        batch_size=4,
        device=torch.device('cpu'),
        seed=2,
        progress=False,
    )

    assert run.best_metrics['mae_mean'] == min(record['metrics']['mae_mean'] for record in run.history)
    assert run.best_epoch == min(
        range(len(run.history)),
        key=lambda epoch: run.history[epoch]['metrics']['mae_mean'],
    )


def test_spacing_objective_fails_on_batch_without_targets():
    with pytest.raises(ValueError, match='without any valid targets'):
        masked_l1_loss(torch.zeros(2, 3), torch.full((2, 3), float('nan')))


def test_spacing_probe_cli_records_validation_selection(tmp_path, monkeypatch):
    features_dir = tmp_path / 'features'
    output_dir = tmp_path / 'output'
    features_dir.mkdir()
    torch.save(
        {
            'cls': torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]),
            'spacing': torch.tensor([
                [0.5, 1.0, 2.0],
                [0.75, 1.25, 2.25],
                [1.25, 1.75, 2.75],
                [1.5, 2.0, 3.0],
            ]),
        },
        features_dir / 'train.pt',
    )
    torch.save(
        {
            'cls': torch.tensor([[-1.5], [1.5]]),
            'spacing': torch.tensor([[0.6, 1.1, 2.1], [1.4, 1.9, 2.9]]),
        },
        features_dir / 'val.pt',
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_spacing',
            '--features-dir',
            str(features_dir),
            '--output-dir',
            str(output_dir),
            '--head',
            'linear',
            '--epochs',
            '1',
            '--seeds',
            '1',
            '--device',
            'cpu',
        ],
    )

    train_spacing.main()

    result = orjson.loads((output_dir / 'spacing_linear.json').read_bytes())
    assert result['protocol'] == 'frozen_probe'
    assert result['selection_policy'] == {
        'split': 'val',
        'metric': 'mae_mean',
        'direction': 'minimize',
    }
    assert result['per_seed_selection'][0]['selected_epoch'] == 0
    assert result['metrics']['val'] == result['summary']
