# tests/ucpt/test_packing.py
"""UCPT packer tests. Unit tests use synthetic samples (no real dataset)."""
from pumit.data import DATA_ROOT

import msgpack
import numpy as np
import orjson
import pytest

from pumit.ucpt.packing import CostModel
from pumit.ucpt.mask_store import MASK_FRAME_KEY, mask_shard_path


_TEST_FIT_RESULT = {
    'unlab': {'coef': [1.0, 1e-2, 1e-6]},
    '2d': {'coef': [2.0, 1e-2, 0.0, 0.2, 1e-4, 0.5]},
    'da0': {'coef': [3.0, 1e-2, 0.0, 0.5, 5e-4, 0.5]},
    'da1': {'coef': [3.0, 1e-2, 0.0, 0.4, 4e-4, 0.5]},
    'da2': {'coef': [3.0, 1e-2, 0.0, 0.3, 3e-4, 0.5]},
    'da3': {'coef': [3.0, 1e-2, 0.0, 0.2, 2e-4, 0.5]},
    'da4': {'coef': [3.0, 1e-2, 0.0, 0.1, 1e-4, 0.5]},
}
_TEST_COST_MODEL = CostModel.from_fit_result(_TEST_FIT_RESULT)


def _lab(n, n_patches=500, base=0):
    """Labeled synthetic samples carry an explicit kind flag and ``da_enc``."""
    return [{'img': f'/fake/lab{base+i}.npy', 'n_patches': n_patches, 'da_enc': 0,
             'labeled': True,
             'seg_cost_queries': 1}
            for i in range(n)]


def _unlab(n, n_patches=500, base=0):
    """Unlabeled synthetic samples carry the same explicit kind flag."""
    return [
        {'img': f'/fake/unlab{base+i}.npy', 'n_patches': n_patches, 'labeled': False}
        for i in range(n)
    ]


def _pack(lab, unlab, budget_ms, f):
    from pumit.ucpt.packing import pack_samples
    return list(pack_samples(
        labeled_iter=iter(lab), unlabeled_iter=iter(unlab),
        budget_ms=budget_ms, label_budget_fraction=f,
        cost_model=_TEST_COST_MODEL,
    ))


def _write_cost_model(path):
    path.write_bytes(orjson.dumps(_TEST_FIT_RESULT))
    return path


def test_cost_model_loads_benchmark_fit(tmp_path):
    cost_model = CostModel.from_file(_write_cost_model(tmp_path / 'coef.json'))
    assert cost_model == _TEST_COST_MODEL
    assert cost_model.as_fit_result() == _TEST_FIT_RESULT


def test_cost_model_requires_every_bucket():
    fit_result = dict(_TEST_FIT_RESULT)
    fit_result.pop('da4')
    with pytest.raises(ValueError, match='cost-model buckets'):
        CostModel.from_fit_result(fit_result)


def test_embedded_default_cost_model_is_complete():
    assert set(CostModel.default().as_fit_result()) == set(_TEST_FIT_RESULT)


def test_cost_labeled_da_dispatch():
    c0 = _TEST_COST_MODEL.cost_labeled(2048, 24, 0)
    c4 = _TEST_COST_MODEL.cost_labeled(2048, 24, 4)
    c2d = _TEST_COST_MODEL.cost_labeled(1024, 3, None)
    assert c0 > 0 and c4 > 0 and c2d > 0
    assert c0 >= c4  # da0 K=24 is the decoder-heavy corner
    with pytest.raises((KeyError, ValueError)):
        _TEST_COST_MODEL.cost_labeled(2048, 24, 7)  # fail-loud on unknown da


def test_sample_cost_reads_da_enc():
    s = {
        'n_patches': 2048,
        'da_enc': 0,
        'labeled': True,
        'seg_cost_queries': 24,
    }
    assert _TEST_COST_MODEL.sample_cost(s) == _TEST_COST_MODEL.cost_labeled(2048, 24, 0)


def test_pack_min_one_of_each_everywhere():
    """Adversarial: f=0, f=1, floor costlier than the whole budget -- every
    emitted batch has >= 1 labeled AND >= 1 unlabeled sample."""
    cost = _TEST_COST_MODEL.cost_unlabeled(500)
    for f in (0.0, 0.5, 1.0):
        for budget in (cost / 2, cost, 10 * cost):
            batches = _pack(_lab(8), _unlab(20), budget_ms=budget, f=f)
            assert batches, f'no batches at f={f} budget={budget}'
            for b in batches:
                n_lab = sum(sample['labeled'] for sample in b['samples'])
                n_unlab = sum(not sample['labeled'] for sample in b['samples'])
                assert n_lab >= 1, f'zero-labeled at f={f} budget={budget}'
                assert n_unlab >= 1, f'zero-unlabeled at f={f} budget={budget}'


def test_pack_budget_adherence_bounded_overshoot():
    """Total batch cost <= budget + at most one unconditionally admitted
    sample per stream (labeled floor + carried-over unlabeled)."""
    cost = _TEST_COST_MODEL.cost_unlabeled(500)
    cost_lab = _TEST_COST_MODEL.sample_cost(_lab(1)[0])  # labeled sample carries the seg increment
    budget = 4.5 * cost
    batches = _pack(_lab(10), _unlab(30), budget_ms=budget, f=0.5)
    for b in batches:
        total = sum(_TEST_COST_MODEL.sample_cost(s) for s in b['samples'])
        assert total <= budget + cost_lab + cost
    # Compound case: floor alone exceeds the whole budget AND a carried-over
    # unlabeled sample is admitted unconditionally -> up to 2 samples overshoot.
    batches = _pack(_lab(4), _unlab(4), budget_ms=cost / 2, f=0.5)
    for b in batches:
        total = sum(_TEST_COST_MODEL.sample_cost(s) for s in b['samples'])
        assert total <= cost / 2 + cost_lab + cost


def test_pack_carry_over_first_of_phase():
    """A non-fitting draw is the first sample of its phase in the NEXT batch:
    labeled overflow becomes the next floor; unlabeled overflow is admitted
    unconditionally at the start of the next unlabeled phase."""
    cost = _TEST_COST_MODEL.cost_unlabeled(500)
    cost_lab = _TEST_COST_MODEL.sample_cost(_lab(1)[0])
    # f*budget holds exactly 1 labeled (the floor); budget holds floor + 1 unlabeled.
    lab, unlab = _lab(4), _unlab(4)
    batches = _pack(lab, unlab, budget_ms=cost_lab + 1.5 * cost, f=0.5)
    # Batch k: [lab[k], unlab[k]] -- the overflowed lab[k+1]/unlab[k+1] lead the
    # next batch's phases, preserving per-stream input order.
    emitted_lab = [s['img'] for b in batches for s in b['samples'] if s['labeled']]
    emitted_unlab = [s['img'] for b in batches for s in b['samples'] if not s['labeled']]
    assert emitted_lab == [s['img'] for s in lab]
    assert emitted_unlab == [s['img'] for s in unlab[:len(emitted_unlab)]]
    for b in batches:
        assert b['samples'][0]['labeled']  # floor leads every batch


def test_pack_prefix_equality_no_loss_no_dup():
    """Emitted labeled subsequence == consumed prefix of the labeled input
    (order and multiplicity): the double-emission / lost-sample tripwire."""
    lab, unlab = _lab(9, n_patches=300), _unlab(15, n_patches=700)
    batches = _pack(lab, unlab, budget_ms=20.0, f=0.4)
    emitted = [s['img'] for b in batches for s in b['samples'] if s['labeled']]
    assert emitted == [s['img'] for s in lab[:len(emitted)]]
    emitted_u = [s['img'] for b in batches for s in b['samples'] if not s['labeled']]
    assert emitted_u == [s['img'] for s in unlab[:len(emitted_u)]]


def test_pack_labeled_exhaustion_terminates_never_floorless():
    """Labeled exhaustion mid-fill: the in-progress batch completes (unlabeled
    fill still runs) and is emitted; packing then terminates at the next floor
    acquisition -- no floorless batch is ever emitted."""
    batches = _pack(_lab(3, n_patches=10), _unlab(50, n_patches=10), budget_ms=1e9, f=1.0)
    assert len(batches) == 1
    assert sum(sample['labeled'] for sample in batches[0]['samples']) == 3


def test_pack_unlabeled_exhaustion_terminates_never_unlabeledless():
    """Unlabeled exhaustion: symmetric to labeled. The in-progress batch
    completes and is emitted; packing terminates at the next floor
    acquisition. No batch is emitted without >= 1 unlabeled sample."""
    batches = _pack(_lab(50, n_patches=10), _unlab(3, n_patches=10), budget_ms=1e9, f=0.0)
    assert len(batches) == 1
    assert sum(not sample['labeled'] for sample in batches[0]['samples']) == 3
    assert sum(sample['labeled'] for sample in batches[0]['samples']) >= 1


def test_pack_deterministic():
    a = _pack(_lab(8), _unlab(20), budget_ms=30.0, f=0.5)
    b = _pack(_lab(8), _unlab(20), budget_ms=30.0, f=0.5)
    assert a == b


class _ExplicitCostModel:
    def sample_cost(self, sample):
        return float(sample['cost'])


def test_nearest_rounding_admits_crossing_sample_when_closer():
    from pumit.ucpt.packing import pack_samples

    labeled = [
        {'labeled': True, 'cost': 6.0, 'id': 'l0'},
        {'labeled': True, 'cost': 6.0, 'id': 'l1'},
    ]
    unlabeled = [{'labeled': False, 'cost': 1.0, 'id': 'u0'}]
    batches = list(
        pack_samples(
            labeled_iter=iter(labeled),
            unlabeled_iter=iter(unlabeled),
            budget_ms=16.0,
            label_budget_fraction=0.625,
            cost_model=_ExplicitCostModel(),
        )
    )
    assert [[sample['id'] for sample in batch['samples']] for batch in batches] == [
        ['l0', 'l1', 'u0'],
    ]


def test_nearest_rounding_carries_on_tie():
    from pumit.ucpt.packing import pack_samples

    batches = list(
        pack_samples(
            labeled_iter=iter([
                {'labeled': True, 'cost': 6.0, 'id': 'l0'},
                {'labeled': True, 'cost': 6.0, 'id': 'l1'},
            ]),
            unlabeled_iter=iter([{'labeled': False, 'cost': 1.0, 'id': 'u0'}]),
            budget_ms=10.0,
            label_budget_fraction=0.9,
            cost_model=_ExplicitCostModel(),
        )
    )
    assert [[sample['id'] for sample in batch['samples']] for batch in batches] == [
        ['l0', 'u0'],
    ]


def test_repack_exactly_reuses_unlabeled_sequence():
    from pumit.ucpt.packing import repack_samples

    old_labeled = [{'labeled': True, 'cost': 6.0, 'id': f'l{i}'} for i in range(2)]
    fresh_labeled = ({'labeled': True, 'cost': 6.0, 'id': f'fresh{i}'} for i in range(10))
    old_unlabeled = [{'labeled': False, 'cost': 6.0, 'id': f'u{i}'} for i in range(6)]
    batches = list(
        repack_samples(
            labeled_iter=iter(old_labeled + list(fresh_labeled)),
            unlabeled_samples=old_unlabeled,
            batches=3,
            budget_ms=12.0,
            label_budget_fraction=0.5,
            cost_model=_ExplicitCostModel(),
        )
    )
    emitted_unlabeled = [
        sample['id']
        for batch in batches
        for sample in batch['samples']
        if not sample['labeled']
    ]
    emitted_labeled = [
        sample['id']
        for batch in batches
        for sample in batch['samples']
        if sample['labeled']
    ]
    assert len(batches) == 3
    assert emitted_unlabeled == [sample['id'] for sample in old_unlabeled]
    assert emitted_labeled[:2] == ['l0', 'l1']


def test_write_batch_shards_count_and_contents(tmp_path):
    from pumit.ucpt.packing import write_batch_shards

    samples = _unlab(10, n_patches=100)
    batches = [{'step_idx': i, 'samples': [samples[i], samples[i]]}
               for i in range(10)]
    n = write_batch_shards(iter(batches), tmp_path, batches_per_shard=4)
    assert n == 3  # 4 + 4 + 2 (final partial shard)

    shards = sorted(tmp_path.glob('shard_*.msgpack'))
    assert len(shards) == 3
    counts = []
    for sp in shards:
        with open(sp, 'rb') as f:
            sh = msgpack.unpack(f, raw=False)
        counts.append(len(sh['batches']))
    assert counts == [4, 4, 2]
    # total_patches sanity: 10 batches * 2 samples * 100 patches
    total = sum(s['n_patches'] for sp in shards for sh in [msgpack.unpack(open(sp, 'rb'), raw=False)]
                for b in sh['batches'] for s in b['samples'])
    assert total == 10 * 2 * 100


def test_write_shard_refuses_to_replace_existing_output(tmp_path):
    from pumit.ucpt.packing import write_shard

    path = tmp_path / 'shard_00000.msgpack'
    write_shard([{'step_idx': 0, 'samples': _lab(1) + _unlab(1)}], path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_shard([{'step_idx': 0, 'samples': _lab(2) + _unlab(1)}], path)
    assert path.read_bytes() == original


def _fake_sample_iter(rng, config_path, labeled, **kwargs):
    """Infinite fake sample stream with one deterministic tag per iterator."""
    tag = int(rng.integers(0, 2**63))
    base = {'n_patches': 100, 'da_enc': 0, 'depth': 16,
            'spacing_label': [2.0, 1.0, 1.0], 'rope_rescale': 1.0, 'params': [{}]}
    i = 0
    while True:
        if labeled:
            yield {**base, 'img': f'/fake/lab{tag}_{i}.npy',
                   'labeled': True,
                   'seg_cost_queries': 1,
                   'dataset': 'd1', 'key': f'k{i}', 'modality': 'CT',
                   'classes': [{'is_positive': False, 'target_voxels': 0}],
                   MASK_FRAME_KEY: b''}
        else:
            yield {**base, 'img': f'/fake/unlab{tag}_{i}.npy', 'labeled': False}
        i += 1


def _fake_generation_globals(config_path):
    pool = ([object()], np.ones(1))
    return object(), (pool, pool), object()


def test_generation_fingerprint_supports_integer_config_keys():
    from pumit.ucpt.stream.build import generation_fingerprint

    kwargs = {
        'seed': 42,
        'batches_per_shard': 4,
        'budget_ms': 5.0,
        'label_budget_fraction': 0.5,
        'cost_model': _TEST_COST_MODEL.as_fit_result(),
        'source_meta_sha256': None,
    }
    config = {'max_depth_per_da': {0: 128, 1: 64}}
    assert generation_fingerprint(config=config, **kwargs) == generation_fingerprint(config=config, **kwargs)
    assert generation_fingerprint(config=config, **kwargs) != generation_fingerprint(
        config={'max_depth_per_da': {0: 128, 1: 32}}, **kwargs,
    )


def test_generate_stream_requires_one_explicit_cost_model_source(tmp_path):
    from pumit.ucpt.stream.build import build_stream

    kwargs = {
        'config_path': 'tests/ucpt/fixtures/gen.yaml',
        'shards': 1,
        'shard_offset': 0,
        'output_dir': tmp_path / 'stream',
        'batches_per_shard': 1,
        'shard_workers': 1,
        'sample_workers': 1,
        'sample_prefetch_factor': 1,
        'budget_ms': 5.0,
        'seed': 42,
        'label_budget_fraction': 0.5,
        'source_stream': None,
        'force': False,
    }
    with pytest.raises(ValueError, match='exactly one'):
        build_stream(**kwargs, cost_model_path=None, use_default_cost_model=False)
    with pytest.raises(ValueError, match='exactly one'):
        build_stream(
            **kwargs,
            cost_model_path=_write_cost_model(tmp_path / 'coef.json'),
            use_default_cost_model=True,
        )


def test_generate_stream_end_to_end_with_fake_samples(monkeypatch, tmp_path):
    """Drive the full pipeline with one shard thread and fake sample iterators.
    Asserts: shard count correct, shards readable, per-shard batch count exact,
    floor guarantee holds, meta valid.

    The shared sample pool receives no work because the fake iterator emits samples directly.
    """
    from pumit.ucpt.stream import build
    from pumit.ucpt.stream.manifest import finalize_stream

    monkeypatch.setattr(build, '_sample_iter', _fake_sample_iter)
    monkeypatch.setattr(build, '_load_worker_globals', _fake_generation_globals)

    out = tmp_path / 'stream'
    cost_model_path = _write_cost_model(tmp_path / 'coef.json')
    n_shards = 2
    # The labeled and unlabeled floors cost 6.56 ms together. A 7 ms budget
    # holds exactly one of each, so batches close quickly (pack_samples only yields a
    # batch once the budget is exceeded; a huge budget would never close a
    # batch against the infinite inline stream -> hang).
    for shard_offset in range(n_shards):
        build.build_stream(
            config_path='tests/ucpt/fixtures/gen.yaml',
            shards=1, shard_offset=shard_offset, output_dir=out, batches_per_shard=4,
            shard_workers=1, sample_workers=1, sample_prefetch_factor=1,
            budget_ms=7.0, seed=42, label_budget_fraction=0.5,
            cost_model_path=cost_model_path, use_default_cost_model=False,
            source_stream=None, force=False,
        )
    finalize_stream(out, expected_shards=n_shards, workers=0)

    shards = sorted(out.glob('shard_*.msgpack'))
    assert len(shards) == n_shards
    for sp in shards:
        with open(sp, 'rb') as f:
            sh = msgpack.unpack(f, raw=False)
        assert len(sh['batches']) == 4
        assert sh['total_patches'] > 0
        for b in sh['batches']:
            assert 'mask_seed' not in b
            assert sum(sample['labeled'] for sample in b['samples']) >= 1
    with open(out / 'meta.yaml') as f:
        import yaml
        meta = yaml.safe_load(f)
    assert meta['stream_complete'] is True
    assert meta['n_shards'] == n_shards
    assert meta['batches_per_shard'] == 4
    assert meta['label_budget_fraction'] == 0.5
    assert meta['cost_model'] == _TEST_FIT_RESULT
    assert 'label_fraction' not in meta


def test_generate_stream_shards_are_independent(monkeypatch, tmp_path):
    """Each shard's RNG is namespaced by shard_id, so distinct shards draw
    distinct sample streams (the img names bake in the per-shard seed). This is
    what makes the shards decorrelated rather than duplicates."""
    from pumit.ucpt.stream import build

    monkeypatch.setattr(build, '_sample_iter', _fake_sample_iter)
    monkeypatch.setattr(build, '_load_worker_globals', _fake_generation_globals)

    out = tmp_path / 'stream'
    cost_model_path = _write_cost_model(tmp_path / 'coef.json')
    build.build_stream(
        config_path='tests/ucpt/fixtures/gen.yaml',
        shards=2, shard_offset=0, output_dir=out, batches_per_shard=4,
        shard_workers=1, sample_workers=1, sample_prefetch_factor=1,
        budget_ms=7.0, seed=42, label_budget_fraction=0.5,
        cost_model_path=cost_model_path, use_default_cost_model=False,
        source_stream=None, force=False,
    )
    s0 = msgpack.unpack(open(out / 'shard_00000.msgpack', 'rb'), raw=False)
    s1 = msgpack.unpack(open(out / 'shard_00001.msgpack', 'rb'), raw=False)
    imgs0 = {s['img'] for b in s0['batches'] for s in b['samples']}
    imgs1 = {s['img'] for b in s1['batches'] for s in b['samples']}
    assert imgs0 and imgs1
    assert imgs0.isdisjoint(imgs1)


def test_build_stream_refuses_shards_without_plan(tmp_path):
    from pumit.ucpt.stream.build import build_stream

    out = tmp_path / 'stream'
    out.mkdir()
    (out / 'shard_99999.msgpack').write_bytes(b'stale')
    cost_model_path = _write_cost_model(tmp_path / 'coef.json')
    with pytest.raises(RuntimeError, match='contains build outputs but no build.yaml'):
        build_stream(
            config_path='tests/ucpt/fixtures/gen.yaml',
            shards=1, shard_offset=0, output_dir=out, batches_per_shard=4,
            shard_workers=1, sample_workers=1, sample_prefetch_factor=1,
            budget_ms=5.0, seed=42,
            label_budget_fraction=0.5,
            cost_model_path=cost_model_path, use_default_cost_model=False,
            source_stream=None, force=False,
        )


def test_generate_stream_default_resume_skips_existing_shards(monkeypatch, tmp_path):
    """Repeated generation keeps complete shards and fills only the gaps."""
    from pumit.ucpt.stream import build

    monkeypatch.setattr(build, '_sample_iter', _fake_sample_iter)
    monkeypatch.setattr(build, '_load_worker_globals', _fake_generation_globals)

    out = tmp_path / 'stream'
    cost_model_path = _write_cost_model(tmp_path / 'coef.json')
    kwargs = dict(
        config_path='tests/ucpt/fixtures/gen.yaml',
        shards=3, shard_offset=0, output_dir=out, batches_per_shard=4,
            shard_workers=1, sample_workers=1, sample_prefetch_factor=1,
        budget_ms=7.0, seed=42, label_budget_fraction=0.5,
        cost_model_path=cost_model_path, use_default_cost_model=False,
        source_stream=None,
    )
    # First run: full generation, then drop shard 1 to simulate an interrupt.
    build.build_stream(**kwargs)
    kept = (out / 'shard_00000.msgpack').read_bytes()
    (out / 'shard_00001.msgpack').unlink()
    mask_shard_path(out, 1).unlink()
    (out / 'reports' / 'shard_00001.json').unlink()

    # Shard 0 must be untouched (byte-identical), shard 1 refilled.
    build.build_stream(**kwargs)
    assert (out / 'shard_00000.msgpack').read_bytes() == kept
    assert sorted(p.name for p in out.glob('shard_*.msgpack')) == [
        'shard_00000.msgpack', 'shard_00001.msgpack', 'shard_00002.msgpack']

    build.build_stream(**kwargs, force=True)
    assert (out / 'shard_00000.msgpack').read_bytes() == kept


def _has_v3_foreground_sidecars() -> bool:
    paths = list(DATA_ROOT.glob('*/fg_coords/index.json'))
    return bool(paths) and all(orjson.loads(path.read_bytes()).get('version') == 3 for path in paths)


needs_dataset = pytest.mark.skipif(
    not DATA_ROOT.exists()
    or not list(DATA_ROOT.glob('*/meta.json'))
    or not _has_v3_foreground_sidecars(),
    reason=f'Real dataset with foreground sidecars ({DATA_ROOT}/) not available',
)


@needs_dataset
def test_generate_stream_against_real_dataset(tmp_path):
    """Full chain against the real dataset: generate_chunk -> pack -> write.
    Asserts divisibility honored (no partial shard), shards readable, meta valid."""
    from pumit.ucpt.stream.build import build_stream

    out = tmp_path / 'stream'
    cost_model_path = _write_cost_model(tmp_path / 'coef.json')
    n_shards = 2
    build_stream(
        config_path='tests/ucpt/fixtures/gen.yaml',
        shards=n_shards, shard_offset=0, output_dir=out, batches_per_shard=4,
        shard_workers=2, sample_workers=4, sample_prefetch_factor=2,
        budget_ms=6000, seed=42, label_budget_fraction=0.5,
        cost_model_path=cost_model_path, use_default_cost_model=False,
        source_stream=None, force=False,
    )

    shards = sorted(out.glob('shard_*.msgpack'))
    assert len(shards) == n_shards
    for sp in shards:
        with open(sp, 'rb') as f:
            sh = msgpack.unpack(f, raw=False)
        assert len(sh['batches']) == 4
        for b in sh['batches']:
            assert all('n_patches' in s for s in b['samples'])
            assert sum(sample['labeled'] for sample in b['samples']) >= 1
        assert sh['total_patches'] > 0
    with open(out / 'build.yaml') as f:
        import yaml
        meta = yaml.safe_load(f)
    assert meta['batches_per_shard'] == 4
    assert meta['label_budget_fraction'] == 0.5
