# tests/ucpt/test_sample_stream.py
"""UCPT sample stream generator tests.

Unit tests build pools from fake records (no real dataset).
The functional test (needs_dataset) drives generate_sample against DATA_ROOT.
"""
import numpy as np
import pytest

from pumit.data import DATA_ROOT


def _fake_records():
    """5 records, 2 labeled. Mirrors build_training_data's record schema."""
    return [
        {'key': 'a', 'dataset': 'd1', 'shape': np.array([20, 256, 256]),
         'spacing': np.array([2.0, 1.0, 1.0]), 'img': '/fake/a.npy', 'weight': 1.0,
         'modality': 'CT', 'label': False, 'label_classes': {}},
        {'key': 'b', 'dataset': 'd1', 'shape': np.array([20, 256, 256]),
         'spacing': np.array([2.0, 1.0, 1.0]), 'img': '/fake/b.npy', 'weight': 3.0,
         'modality': 'CT', 'label': True,
         'label_classes': {'src1': {'positive': ['liver', 'spleen'], 'negative': ['kidney']}}},
        {'key': 'c', 'dataset': 'd2', 'shape': np.array([1, 384, 384]),
         'spacing': np.array([float('nan'), 1.0, 1.0]), 'img': '/fake/c.npy', 'weight': 1.0,
         'modality': 'CT', 'label': False, 'label_classes': {}},
        {'key': 'd', 'dataset': 'd2', 'shape': np.array([50, 256, 256]),
         'spacing': np.array([2.0, 1.0, 1.0]), 'img': '/fake/d.npy', 'weight': 1.0,
         'modality': 'MRI', 'label': True,
         'label_classes': {'src1': {'positive': ['tumor'], 'negative': []}}},
        {'key': 'e', 'dataset': 'd2', 'shape': np.array([100, 256, 256]),
         'spacing': np.array([2.0, 1.0, 1.0]), 'img': '/fake/e.npy', 'weight': 1.0,
         'modality': 'CT', 'label': False, 'label_classes': {}},
    ]


def test_build_pools_splits_full_and_labeled():
    from pumit.ucpt.sample_stream import build_pools

    records = _fake_records()
    (rec_all, w_all), (rec_lab, w_lab) = build_pools(records)

    assert len(rec_all) == 5
    assert len(rec_lab) == 2
    assert {r['key'] for r in rec_lab} == {'b', 'd'}
    assert np.isclose(w_all.sum(), 1.0)
    assert np.isclose(w_lab.sum(), 1.0)
    # 'b' has weight 3.0, 'd' has 1.0 -> normalized 0.75 / 0.25
    assert np.isclose(w_lab[0], 0.75)
    assert np.isclose(w_lab[1], 0.25)


def test_normalize_label_contract_is_complete_and_does_not_mutate_record():
    import copy
    from pumit.ucpt.seg.label_contract import normalize_label_contract

    record = _fake_records()[1]
    snapshot = copy.deepcopy(record['label_classes'])
    label_classes, num_pairs = normalize_label_contract(record)

    assert label_classes == snapshot
    assert num_pairs == 3
    assert record['label_classes'] == snapshot


class _FakePipeline:
    """Returns deterministic affine params without touching disk."""
    def __init__(self, *, forced: bool = False):
        self.forced = forced

    def sample_params(self, state, rng):
        sp = {
            'da_enc': None if state['shape'][0] == 1 else 4,
            'n_patches': 16,
            'crop_size': [1 if state['shape'][0] == 1 else 16, 256, 256],
            'spacing_label': [2.0, 1.0, 1.0],
            'foreground_forced': self.forced and '_center_class' in state,
        }
        return [sp]


class _FakeClassSampler:
    def __init__(self):
        self.focuses = []

    def _sample_classes(self, label_contract, *, focus):
        self.focuses.append(focus)
        classes = []
        for source in sorted(label_contract):
            info = label_contract[source]
            classes.extend({
                'source': source,
                'name': name,
                'is_positive': True,
                'target_voxels': 10,
            } for name in info['positive'])
            classes.extend({
                'source': source,
                'name': name,
                'is_positive': False,
                'target_voxels': 0,
            } for name in info['negative'])
        return classes

    def __call__(self, record, label_contract, params, rng, *, focus):
        return self._sample_classes(label_contract, focus=focus)

    def sample_materialized_with_stats(self, record, label_contract, params, rng, *, focus):
        from pumit.ucpt.mask_store import encode_positive_masks

        classes = self._sample_classes(label_contract, focus=focus)
        shape = tuple(params[0]['crop_size'])
        masks = []
        for item in classes:
            if not item['is_positive']:
                continue
            mask = np.zeros(shape, dtype=np.bool_)
            mask.reshape(-1)[:item['target_voxels']] = True
            masks.append(mask)
        return classes, encode_positive_masks(masks), {}


def _fake_pools(records):
    from pumit.ucpt.sample_stream import build_pools
    return build_pools(records)


def _gen_n(pipeline, pools, class_sampler, labeled, n, seed=42):
    """Call the lib's generate_sample n times under one RNG (no CLI, no pool)."""
    from pumit.ucpt.sample_stream import generate_sample
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = generate_sample(pipeline, pools, class_sampler, rng, labeled)
        if s is not None:
            out.append(s)
    return out


_BASE_FIELDS = {'img', 'spacing_label', 'params', 'da_enc', 'n_patches', 'depth', 'rope_rescale', 'labeled'}


def test_generate_sample_all_labeled():
    from pumit.ucpt.sample_stream import generate_sample  # noqa: F401 (sanity import)
    from pumit.ucpt.mask_store import MASK_FRAME_KEY

    pipeline, pools = _FakePipeline(), _fake_pools(_fake_records())
    out = _gen_n(pipeline, pools, _FakeClassSampler(), labeled=True, n=50)

    assert len(out) == 50
    for s in out:
        assert s['labeled'] and 'label_classes' not in s and s['classes']
        assert s['seg_cost_queries'] in {1, 3}
        assert s['seg_cost_queries'] == len(s['classes'])
        assert s['key'] in {'b', 'd'}
        assert 'dataset' in s and 'modality' in s
        assert isinstance(s[MASK_FRAME_KEY], bytes) and s[MASK_FRAME_KEY]
        assert _BASE_FIELDS <= set(s)


def test_generate_sample_all_unlabeled():
    pipeline, pools = _FakePipeline(), _fake_pools(_fake_records())
    out = _gen_n(pipeline, pools, _FakeClassSampler(), labeled=False, n=50)

    assert len(out) == 50
    for s in out:
        assert not s['labeled'] and 'label_classes' not in s
        assert 'key' not in s
        assert _BASE_FIELDS <= set(s)


def test_build_pools_excludes_non_dict_label_classes():
    """List-valued label_classes (e.g. MnMs) must be kept out of the labeled
    pool so stream generation never receives a non-dict contract."""
    from pumit.ucpt.sample_stream import build_pools

    records = _fake_records() + [
        {'key': 'mnms', 'dataset': 'MnMs', 'shape': np.array([20, 256, 256]),
         'spacing': np.array([2.0, 1.0, 1.0]), 'img': '/fake/mnms.npy', 'weight': 1.0,
         'modality': 'MRI', 'label': True,
         'label_classes': {'MnMs': ['LV', 'myocardium', 'RV']}},  # bare list, not dict
    ]
    (rec_all, _), (rec_lab, _) = build_pools(records)

    assert len(rec_all) == 6  # full pool keeps it (used as unlabeled SSL image)
    assert {r['key'] for r in rec_lab} == {'b', 'd'}  # 'mnms' excluded from labeled pool


def test_generate_sample_accepts_negative_only_record():
    from pumit.ucpt.sample_stream import generate_sample
    from pumit.ucpt.mask_store import MASK_FRAME_KEY

    records = [
        {'key': 'np', 'dataset': 'd1', 'shape': np.array([20, 256, 256]),
         'spacing': np.array([2.0, 1.0, 1.0]), 'img': '/fake/np.npy', 'weight': 1.0,
         'modality': 'CT', 'label': True,
         'label_classes': {'src1': {'positive': [], 'negative': ['kidney']}}},
    ]
    pipeline, pools = _FakePipeline(), _fake_pools(records)
    rng = np.random.default_rng(42)
    sampler = _FakeClassSampler()
    out = [generate_sample(pipeline, pools, sampler, rng, True) for _ in range(30)]
    assert all(s is not None for s in out)
    assert all(s['labeled'] and 'label_classes' not in s for s in out)
    assert all('focus_class' not in s for s in out)
    assert all(s[MASK_FRAME_KEY] == b'' for s in out)


def test_generation_entrypoint_samples_the_current_medical_and_display_recipe(tmp_path, monkeypatch):
    import orjson
    import pandas as pd
    import yaml

    import pumit.ucpt.sample_stream as sample_stream
    from pumit.ucpt.input import InputNormalizer

    data_root = tmp_path / 'preprocess'
    records = [
        {
            'key': key, 'dataset': 'Mixed', 'img': str(data_root / 'Mixed' / 'images' / f'{key}.npy'),
            'shape': np.array([1, 16, 16]), 'spacing': np.array([np.nan, 1.0, 1.0]),
            'weight': 1.0, 'modality': 'US', 'label': True,
            'label_classes': {'source': {'positive': [], 'negative': ['lesion']}},
        }
        for key in ('medical', 'display')
    ]
    metadata_path = tmp_path / 'fingerprint' / 'Mixed' / 'samples.json'
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_bytes(orjson.dumps({
        'medical': {'image_dtype': 'float32'},
        'display': {'image_dtype': {'original': 'int16', 'cache': 'uint8'}},
    }))
    config_path = tmp_path / 'generation.yaml'
    config_path.write_text(yaml.safe_dump({
        'size_xy_choices': [16], 'size_xy_choices_2d': [16], 'max_depth_per_da': {0: 16},
        'fg_coords': True, 'seg_positive_queries': 1, 'seg_negative_queries': 1,
        'seg_min_positive_voxels': 1, 'seg_mask_batch_size': 1,
    }))
    monkeypatch.setattr(sample_stream, 'DATA_ROOT', data_root)
    monkeypatch.setattr(
        sample_stream, 'build_training_data',
        lambda **kwargs: (pd.DataFrame(records).set_index('key'), None, None),
    )
    monkeypatch.setattr('pumit.ucpt.fg_coords.FgCoordCache', lambda *args, **kwargs: object())
    monkeypatch.setattr(sample_stream, 'CropClassSampler', lambda **kwargs: _FakeClassSampler())
    try:
        pipeline, pools, sampler = sample_stream._load_generation_globals(str(config_path))
        assert isinstance(pipeline.transforms[-1], InputNormalizer)
        generated = _gen_n(pipeline, pools, sampler, labeled=True, n=200)
        medical = [sample for sample in generated if sample['key'] == 'medical']
        display = [sample for sample in generated if sample['key'] == 'display']
        assert medical and display
        for sample in display:
            assert sample['params'][1:4] == [{'enabled': False}] * 3
        assert any(sample['params'][1]['enabled'] for sample in medical)
        assert any(sample['params'][3]['enabled'] for sample in medical)
        for sample in medical:
            scale, contrast, gamma = sample['params'][1:4]
            assert contrast == {'enabled': False}
            if scale['enabled']:
                assert all(-0.2 <= factor <= 0.2 for factor in scale['factors'])
            if gamma['enabled']:
                assert not gamma['invert']
                assert all(0.8 <= value <= 1.2 for value in gamma['gammas'])
    finally:
        sample_stream._load_generation_globals.cache_clear()


def _has_current_fg_sidecars() -> bool:
    import orjson

    meta_paths = list(DATA_ROOT.glob('*/meta.json')) if DATA_ROOT.exists() else []
    if not meta_paths:
        return False
    for meta_path in meta_paths:
        index_path = meta_path.parent / 'fg_coords' / 'index.json'
        if not index_path.exists():
            return False
        index = orjson.loads(index_path.read_bytes())
        if not isinstance(index, dict) or index.get('version') != 3:
            return False
    return True


needs_dataset = pytest.mark.skipif(
    not _has_current_fg_sidecars(),
    reason=f'Real dataset and v3 foreground sidecars ({DATA_ROOT}/) not available',
)


@needs_dataset
def test_generate_sample_against_real_dataset():
    """Drives the real build_training_data path: pools build, labeled keys are a
    subset of labeled records, and each stream produces only its kind."""
    from pumit.ucpt.sample_stream import _load_generation_globals, generate_sample

    cfg = 'tests/ucpt/fixtures/gen.yaml'
    pipeline, pools, class_sampler = _load_generation_globals(cfg)
    (records_all, _), (records_lab, _) = pools
    assert len(records_lab) > 0
    assert len(records_lab) < len(records_all)
    labeled_keys = {r['key'] for r in records_lab}

    rng = np.random.default_rng(42)
    labeled_out = []
    for _ in range(2):
        s = generate_sample(pipeline, pools, class_sampler, rng, True)
        if s is not None:
            labeled_out.append(s)
    assert labeled_out, 'labeled stream should emit samples'
    for s in labeled_out:
        assert s['key'] in labeled_keys
        assert s['labeled']
        assert 'label_classes' not in s and s['classes']

    unlabeled_out = []
    for _ in range(2):
        s = generate_sample(pipeline, pools, class_sampler, rng, False)
        if s is not None:
            unlabeled_out.append(s)
    assert unlabeled_out
    for s in unlabeled_out:
        assert not s['labeled'] and 'label_classes' not in s


def test_labeled_size_xy_cap():
    """Labeled draws respect labeled_size_xy_max(_2d); unlabeled keep the full range."""
    from pumit.ucpt.transforms import AffinePatchLoader

    loader = AffinePatchLoader(
        size_xy_choices=[128, 192, 256, 320, 384],
        size_xy_choices_2d=[128, 192, 256, 320, 384, 448, 512],
        max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12},
        labeled_size_xy_max=256,
        labeled_size_xy_max_2d=384,
    )
    rng = np.random.default_rng(0)
    state_3d = {'shape': np.array([40, 512, 512]), 'spacing': np.array([2.0, 1.0, 1.0])}
    state_2d = {'shape': np.array([1, 1024, 1024]), 'spacing': np.array([float('nan'), 1.0, 1.0])}

    for _ in range(20):
        sp = loader.sample_params({**state_3d, 'labeled_draw': True}, rng)
        assert sp['crop_size'][1] <= 256
        sp = loader.sample_params({**state_2d, 'labeled_draw': True}, rng)
        assert sp['crop_size'][1] <= 384
    # Unlabeled: eligibility picks the max choice (512-extent 3D -> 384; 1024 2D -> 512).
    assert loader.sample_params(state_3d, rng)['crop_size'][1] == 384
    assert loader.sample_params(state_2d, rng)['crop_size'][1] == 512


def test_generate_sample_labeled_sets_labeled_draw():
    """generate_sample marks labeled states so the loader can apply the cap;
    unlabeled draws (even of labeled records) must not carry the flag."""
    seen: list[bool] = []

    class _SpyPipeline(_FakePipeline):
        def sample_params(self, state, rng):
            seen.append(bool(state.get('labeled_draw', False)))
            return super().sample_params(state, rng)

    pipeline, pools = _SpyPipeline(), _fake_pools(_fake_records())
    _gen_n(pipeline, pools, _FakeClassSampler(), labeled=True, n=10)
    assert seen and all(seen)
    seen.clear()
    _gen_n(pipeline, pools, _FakeClassSampler(), labeled=False, n=10)
    assert seen and not any(seen)


# --- foreground oversampling ---

def _make_cache(tmp_path):
    import orjson
    from pumit.ucpt.fg_coords import FgCoordCache
    d = tmp_path / 'd1' / 'fg_coords'
    d.mkdir(parents=True)
    coords = np.array([[10, 128, 128]], dtype=np.int16)
    np.save(d / 'coords.npy', coords)
    (d / 'index.json').write_bytes(orjson.dumps({
        'version': 3,
        'coord_cap': 1024,
        'records': [['b', 'src1', 'liver', 0, 1, [10, 128, 128], [11, 129, 129]]],
    }))
    return FgCoordCache(tmp_path, datasets=['d1'])


def test_focus_class_is_used_only_when_forcing_occurs():
    pools = _fake_pools([_fake_records()[1]])
    sampler = _FakeClassSampler()
    out = _gen_n(_FakePipeline(forced=True), pools, sampler, labeled=True, n=1)[0]
    assert sampler.focuses[0] in {('src1', 'liver'), ('src1', 'spleen')}
    assert 'focus_class' not in out

    sampler = _FakeClassSampler()
    _gen_n(_FakePipeline(forced=False), pools, sampler, labeled=True, n=1)
    assert sampler.focuses == [None]


def test_build_ucpt_pipeline_accepts_fg_cache(tmp_path):
    from pumit.ucpt.transforms import build_ucpt_pipeline
    cache = _make_cache(tmp_path)  # helper already defined earlier in this file (Task 5)
    pipe = build_ucpt_pipeline(
        size_xy_choices=[128], size_xy_choices_2d=[128],
        max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12},
        fg_cache=cache, force_fraction=0.5,
    )
    loader = pipe.transforms[0]
    assert loader.fg_cache is cache
    assert loader.force_fraction == 0.5
