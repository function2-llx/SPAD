"""Step-based training contract: infinite permutation-cycling sampler + step-interval validation.

Epoch boundaries carry no meaning for this harness -- the sampler already draws a fresh random
permutation each pass, so a "pass over the data" is just an arbitrary chunk of that stream.
Training is therefore defined by a fixed update budget, and validation by a step interval, so both
are identical across datasets regardless of their size.
"""
import pytest
import torch

from pumit.downstream.cls.finetune import iter_batch_indices


def test_cycles_through_fresh_permutations():
    """Each pass over n is a permutation; consecutive passes differ; every index appears once."""
    gen = torch.Generator().manual_seed(0)
    n, bs = 10, 5
    batches = list(iter_batch_indices(n, bs, steps=4, generator=gen))
    assert len(batches) == 4
    assert all(len(b) == bs for b in batches), 'every step must be a full batch'
    first_pass = sorted(torch.cat(batches[:2]).tolist())
    second_pass = sorted(torch.cat(batches[2:]).tolist())
    assert first_pass == list(range(n)), 'a full pass must cover every index exactly once'
    assert second_pass == list(range(n))
    assert torch.cat(batches[:2]).tolist() != torch.cat(batches[2:]).tolist(), 'passes must reshuffle'


def test_yields_exactly_the_requested_step_count():
    """The budget is in steps, not epochs: no rounding up to a dataset-sized boundary."""
    gen = torch.Generator().manual_seed(0)
    for steps in (1, 7, 33, 500, 1000):
        assert len(list(iter_batch_indices(1027, 32, steps=steps, generator=gen))) == steps


def test_drops_the_ragged_tail_so_every_step_is_full_batch():
    """n=10, bs=4 -> 2 full batches per pass; the 2-sample remainder rolls into the next pass."""
    gen = torch.Generator().manual_seed(0)
    batches = list(iter_batch_indices(10, 4, steps=5, generator=gen))
    assert all(len(b) == 4 for b in batches)
    # a partial batch would change the effective batch size mid-run, and with CUDA graphs
    # (reduce-overhead) a varying shape would force recompilation
    assert len({len(b) for b in batches}) == 1


def test_is_deterministic_for_a_given_seed():
    a = list(iter_batch_indices(100, 8, steps=20, generator=torch.Generator().manual_seed(7)))
    b = list(iter_batch_indices(100, 8, steps=20, generator=torch.Generator().manual_seed(7)))
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    c = list(iter_batch_indices(100, 8, steps=20, generator=torch.Generator().manual_seed(8)))
    assert [x.tolist() for x in a] != [x.tolist() for x in c]


def test_batch_larger_than_dataset_still_fills():
    """Small val-like sets must still yield full batches by spanning permutations."""
    gen = torch.Generator().manual_seed(0)
    batches = list(iter_batch_indices(5, 8, steps=3, generator=gen))
    assert all(len(b) == 8 for b in batches)


def test_rejects_invalid_arguments():
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match='steps'):
        list(iter_batch_indices(10, 2, steps=0, generator=gen))
    with pytest.raises(ValueError, match='batch_size'):
        list(iter_batch_indices(10, 0, steps=1, generator=gen))


def test_eval_boundaries_handle_budget_smaller_than_interval():
    """total_steps < eval_every must still validate once, at the final step (not crash).

    Reachable from the CLI: --num-updates 20 with the default --eval-every 50.
    """
    from pumit.downstream.cls.finetune import eval_boundaries

    assert eval_boundaries(total_steps=20, eval_every=50) == [20]
    assert eval_boundaries(total_steps=50, eval_every=50) == [50]
    assert eval_boundaries(total_steps=1, eval_every=50) == [1]
    # the ordinary case is unchanged, and the final step is always included
    assert eval_boundaries(total_steps=1000, eval_every=50)[:3] == [50, 100, 150]
    assert eval_boundaries(total_steps=1000, eval_every=50)[-1] == 1000
    assert eval_boundaries(total_steps=120, eval_every=50) == [50, 100, 120]


def test_eval_uses_a_single_padded_shape():
    """Every eval batch is exactly eval_batch_size, so CUDA graphs capture ONE eval shape.

    Under compile reduce-overhead each distinct input shape captures its own graph with a
    permanently retained memory pool; ragged last batches added a pool per split and exhausted
    the GPU (observed: 56.5 GiB in private pools -> OOM). Padding keeps it to one.
    """
    import numpy as np
    from pumit.downstream.cls.finetune import iter_eval_batches

    n, bs = 310, 128
    batches = list(iter_eval_batches(n, bs))
    assert [len(idx) for idx, _ in batches] == [bs, bs, bs], 'all eval batches must be full'
    # the valid-count mask tells the caller how many rows of the last batch are real
    assert [valid for _, valid in batches] == [128, 128, 54]
    # indices cover 0..n-1 exactly once, with the tail padded by repeating the final index
    covered = np.concatenate([idx[:valid] for idx, valid in batches])
    assert covered.tolist() == list(range(n))


def test_eval_padding_is_exact_for_a_multiple():
    from pumit.downstream.cls.finetune import iter_eval_batches
    batches = list(iter_eval_batches(256, 128))
    assert [(len(i), v) for i, v in batches] == [(128, 128), (128, 128)]


def test_eval_handles_split_smaller_than_batch():
    from pumit.downstream.cls.finetune import iter_eval_batches
    batches = list(iter_eval_batches(50, 128))
    assert len(batches) == 1
    idx, valid = batches[0]
    assert len(idx) == 128 and valid == 50
