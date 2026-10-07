"""TTA contract: the 8 flip views, and probability averaging through the metric path."""
import numpy as np
import torch

from pumit.downstream.cls.augment import FLIP_VIEWS, flip_view, flips_enabled_for
from pumit.downstream.cls.metrics import logits_to_scores


def test_flip_views_are_the_eight_distinct_orientations():
    assert len(FLIP_VIEWS) == 8
    assert len(set(FLIP_VIEWS)) == 8
    assert () in FLIP_VIEWS                                  # identity view is included
    assert all(set(v) <= {1, 2, 3} for v in FLIP_VIEWS)
    assert all(tuple(sorted(v)) == v for v in FLIP_VIEWS)    # axes ascending, so flips commute


def test_every_view_is_an_exact_involution_on_the_batch_axes():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(3, 4, 4, 4), dtype=np.uint8)
    seen = []
    for axes in FLIP_VIEWS:
        v = flip_view(x, axes)
        assert v.shape == x.shape and v.dtype == x.dtype
        np.testing.assert_array_equal(flip_view(v, axes), x)   # applying twice restores
        seen.append(v.tobytes())
    assert len(set(seen)) == 8                                 # the 8 views differ on real data


def test_identity_view_does_not_copy():
    x = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    assert flip_view(x, ()) is x


def test_averaged_probabilities_survive_the_metric_softmax():
    """eval_split returns log(mean prob); logits_to_scores softmaxes it, which must invert."""
    torch.manual_seed(0)
    views = [torch.randn(5, 3) for _ in range(8)]
    probs = torch.stack([torch.softmax(v, dim=1) for v in views]).mean(dim=0)
    recovered = logits_to_scores(probs.log())
    np.testing.assert_allclose(recovered, probs.numpy().astype(np.float64), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(recovered.sum(axis=1), 1.0, rtol=1e-6)   # float32 round-trip


def test_averaging_probabilities_differs_from_averaging_logits():
    """Guards the choice: nnU-Net averages softmax, and the two are not interchangeable."""
    logits = torch.tensor([[[4.0, 0.0]], [[0.0, 4.0]], [[4.0, 0.0]], [[0.0, 4.0]],
                           [[4.0, 0.0]], [[0.0, 4.0]], [[4.0, 0.0]], [[3.0, 0.0]]])
    prob_avg = torch.stack([torch.softmax(v, dim=1) for v in logits]).mean(dim=0)
    logit_avg = torch.softmax(torch.stack(list(logits)).mean(dim=0), dim=1)
    assert not torch.allclose(prob_avg, logit_avg, atol=1e-3)


def test_tta_is_gated_off_where_flips_are():
    assert not flips_enabled_for('organmnist3d')
    assert all(flips_enabled_for(f) for f in
               ('nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d',
                'vesselmnist3d', 'synapsemnist3d'))


def test_eval_split_averages_over_all_views(monkeypatch):
    """A view-sensitive encoder must produce a different result under tta than without."""
    from pumit.downstream.cls import finetune

    class Spec:
        flag, size, is_3d = 'nodulemnist3d', 64, True

    images = np.arange(2 * 4 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4, 4)
    labels = np.array([0, 1])
    monkeypatch.setattr(finetune, 'build_arrays', lambda *a, **k: (images, labels))
    seen = []

    class Enc(torch.nn.Module):
        def forward(self, x):
            seen.append(x.sum().item())
            return x.flatten(1)[:, :2], x.flatten(1)[:, None, :2]

    enc = Enc()
    head = torch.nn.Identity()
    kwargs = dict(spec=Spec(), split='test', device='cpu', batch_size=2,
                  transform_batch=lambda im, is_3d: {'x': torch.from_numpy(im).float()})
    plain, _ = finetune.eval_split(enc, head, **kwargs)
    n_plain = len(seen)
    tta, _ = finetune.eval_split(enc, head, **kwargs, tta=True)
    assert len(seen) - n_plain == 8 * n_plain          # 8 forward passes per batch
    assert not torch.allclose(plain, tta)
    np.testing.assert_allclose(logits_to_scores(tta).sum(axis=1), 1.0, rtol=1e-6)
