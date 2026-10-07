import pytest
import torch
import torch.nn.functional as F

import pumit.downstream.cached_probe as cached_probe
from pumit.downstream.cached_probe import (
    aggregate_metric_records,
    build_probe_head,
    train_cached_probe,
)
from pumit.downstream.evaluation import MetricSelection


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {'mae': F.l1_loss(prediction, target).item()}


def test_metric_selection_supports_both_directions():
    maximize = MetricSelection(metric='auc', direction='maximize')
    minimize = MetricSelection(metric='mae', direction='minimize')

    assert maximize.is_better({'auc': 0.8}, None)
    assert maximize.is_better({'auc': 0.8}, {'auc': 0.7})
    assert not maximize.is_better({'auc': 0.7}, {'auc': 0.8})
    assert minimize.is_better({'mae': 0.2}, {'mae': 0.3})
    assert not minimize.is_better({'mae': 0.3}, {'mae': 0.2})


def test_metric_selection_fails_on_missing_or_non_finite_metric():
    selection = MetricSelection(metric='mae', direction='minimize')
    with pytest.raises(KeyError, match='mae'):
        selection.value({'auc': 0.5})
    with pytest.raises(ValueError, match='finite'):
        selection.value({'mae': float('nan')})


@pytest.mark.parametrize(
    ('head_type', 'linear_layers'),
    [('linear', 1), ('mlp', 2), ('mlp2', 3), ('mlp3', 4)],
)
def test_build_probe_head_preserves_existing_head_variants(head_type: str, linear_layers: int):
    head = build_probe_head(4, 3, head_type)
    assert head(torch.zeros(2, 4)).shape == (2, 3)
    assert sum(isinstance(module, torch.nn.Linear) for module in head.modules()) == linear_layers


def test_cached_probe_restores_validation_selected_state():
    train_x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = 2 * train_x
    val_x = torch.tensor([[-1.5], [1.5]])
    val_y = 2 * val_x
    selection = MetricSelection(metric='mae', direction='minimize')

    run = train_cached_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        head_factory=lambda input_dim: build_probe_head(input_dim, 1, 'linear'),
        objective=F.l1_loss,
        evaluator=regression_metrics,
        selection=selection,
        epochs=5,
        lr=0.2,
        weight_decay=0.0,
        batch_size=4,
        device='cpu',
        seed=3,
        progress=False,
    )

    selected_metrics = regression_metrics(run.model(val_x), val_y)
    assert run.best_epoch in range(5)
    assert len(run.history) == 5
    assert selected_metrics == pytest.approx(run.best_metrics)
    assert run.best_metrics['mae'] == min(record['metrics']['mae'] for record in run.history)


def test_aggregate_metric_records_requires_consistent_finite_metrics():
    summary = aggregate_metric_records([{'auc': 0.7, 'acc': 0.8}, {'auc': 0.9, 'acc': 0.6}])
    assert summary == {
        'auc': {'mean': pytest.approx(0.8), 'std': pytest.approx(0.1), 'n': 2},
        'acc': {'mean': pytest.approx(0.7), 'std': pytest.approx(0.1), 'n': 2},
    }

    with pytest.raises(ValueError, match='same metric keys'):
        aggregate_metric_records([{'auc': 0.7}, {'acc': 0.8}])
    with pytest.raises(ValueError, match='finite'):
        aggregate_metric_records([{'auc': float('nan')}])


@pytest.mark.parametrize(('n_train', 'batch_size'), [(5, 3), (2, 5)])
def test_fixed_updates_keep_full_batches_and_restore_selected_update(monkeypatch, n_train, batch_size):
    features = torch.arange(n_train, dtype=torch.float32).reshape(-1, 1)
    batches, predictions, learning_rates, schedulers = [], [], [], []
    scheduler_cls = cached_probe.CosineAnnealingLR

    def make_scheduler(optimizer, **kwargs):
        scheduler = scheduler_cls(optimizer, **kwargs)
        schedulers.append(scheduler)
        return scheduler

    def objective(prediction, target):
        batches.append(target.flatten().tolist())
        learning_rates.append(schedulers[0].get_last_lr()[0])
        return F.mse_loss(prediction, target)

    def evaluator(prediction, target):
        predictions.append(prediction.clone())
        return {'mae': [0.2, 0.1, 0.3][len(predictions) - 1]}

    monkeypatch.setattr(cached_probe, 'CosineAnnealingLR', make_scheduler)
    run = train_cached_probe(
        features, features, features, features,
        head_factory=lambda input_dim: build_probe_head(input_dim, 1, 'linear'),
        objective=objective,
        evaluator=evaluator,
        selection=MetricSelection(metric='mae', direction='minimize'),
        epochs=50,
        updates=5,
        eval_every=2,
        lr=0.1,
        weight_decay=0.01,
        batch_size=batch_size,
        device='cpu',
        seed=7,
        progress=False,
    )

    assert len(batches) == 5
    assert all(len(batch) == batch_size for batch in batches)
    stream = [item for batch in batches for item in batch]
    for start in range(0, len(stream) - n_train + 1, n_train):
        assert sorted(stream[start:start + n_train]) == list(range(n_train))
    assert schedulers[0].T_max == 5
    assert schedulers[0].last_epoch == 5
    assert schedulers[0].get_last_lr() == [0.0]
    assert learning_rates[0] == 0.1
    assert learning_rates == sorted(learning_rates, reverse=True)
    assert [record['update'] for record in run.history] == [2, 4, 5]
    assert run.best_update == 4
    assert run.best_epoch == -1
    assert run.best_metrics == {'mae': 0.1}
    torch.testing.assert_close(run.model(features), predictions[1])
