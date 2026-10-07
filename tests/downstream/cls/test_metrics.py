import numpy as np
import pytest
import torch

from pumit.downstream.cls.metrics import evaluate, fallback_metrics, logits_to_scores


def test_logits_to_scores_softmax():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    scores = logits_to_scores(logits)
    assert scores.shape == (2, 2)
    np.testing.assert_allclose(scores.sum(axis=1), [1.0, 1.0], atol=1e-5)
    assert scores[0, 0] > scores[0, 1]


def test_fallback_metrics_perfect():
    logits = torch.tensor([[9., 0, 0], [0, 9., 0], [0, 0, 9.]])
    labels = np.array([0, 1, 2])
    m = fallback_metrics(logits, labels, n_classes=3)
    assert m['acc'] == 1.0
    assert m['auc'] > 0.99


def test_fallback_metrics_binary():
    logits = torch.tensor([[9., 0.], [0., 9.], [9., 0.], [0., 9.]])
    labels = np.array([0, 1, 0, 1])
    m = fallback_metrics(logits, labels, n_classes=2)
    assert m['acc'] == 1.0
    assert m['auc'] > 0.99


def test_evaluate_uses_official_evaluator_and_checks_label_order(monkeypatch):
    import medmnist

    class FakeEvaluator:
        def __init__(self, flag, split, size, root):
            assert (flag, split, size, root) == ('pathmnist', 'test', 1, '/data')
            self.labels = np.array([[0], [1]])

        def evaluate(self, scores):
            assert scores.shape == (2, 2)
            return 0.75, 0.5

    monkeypatch.setattr(medmnist, 'Evaluator', FakeEvaluator)
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert evaluate(logits, np.array([0, 1]), 'pathmnist', 1, 'test', root='/data') == {
        'auc': 0.75,
        'acc': 0.5,
    }

    with pytest.raises(ValueError, match='official npz ordering'):
        evaluate(logits, np.array([1, 0]), 'pathmnist', 1, 'test', root='/data')


def test_evaluate_does_not_hide_evaluator_failures(monkeypatch):
    import medmnist

    class FailingEvaluator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('bad root')

    monkeypatch.setattr(medmnist, 'Evaluator', FailingEvaluator)
    with pytest.raises(RuntimeError, match='bad root'):
        evaluate(torch.zeros(1, 2), np.array([0]), 'pathmnist', 1, 'test', root='/missing')
