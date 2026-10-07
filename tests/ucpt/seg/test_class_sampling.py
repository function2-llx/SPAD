from pathlib import Path

import numpy as np

from pumit.ucpt.seg.class_sampling import flatten_label_contract, sample_classes
from pumit.ucpt.seg.data import CropClassSampler


def _item(source, name, positive, voxels):
    return {
        'source': source,
        'name': name,
        'is_positive': positive,
        'target_voxels': voxels,
    }


def test_flatten_label_contract_keeps_source_class_identity():
    positive, negative = flatten_label_contract({
        'b': {'positive': ['x'], 'negative': ['n']},
        'a': {'positive': ['x', 'y'], 'negative': []},
    })

    assert positive == [('a', 'x'), ('a', 'y'), ('b', 'x')]
    assert negative == [('b', 'n')]


def test_sampling_is_deterministic_and_retains_focus():
    positive = [
        _item('a', 'x', True, 10),
        _item('a', 'y', True, 20),
        _item('b', 'z', True, 30),
    ]
    negative = [_item('b', 'n', False, 0)]

    first = sample_classes(
        positive,
        negative,
        positive_queries=2,
        negative_queries=1,
        rng=np.random.default_rng(123),
        focus=('b', 'z'),
    )
    second = sample_classes(
        positive,
        negative,
        positive_queries=2,
        negative_queries=1,
        rng=np.random.default_rng(123),
        focus=('b', 'z'),
    )

    assert first == second
    assert ('b', 'z') in {(item['source'], item['name']) for item in first}
    assert sum(bool(item['is_positive']) for item in first) == 2
    assert sum(not item['is_positive'] for item in first) == 1


def test_sampling_does_not_merge_classes_with_matching_text_semantics():
    selected = sample_classes(
        [_item('a', 'x', True, 10), _item('b', 'x2', True, 12)],
        [],
        positive_queries=2,
        negative_queries=0,
        rng=np.random.default_rng(1),
    )

    assert {(item['source'], item['name']) for item in selected} == {
        ('a', 'x'),
        ('b', 'x2'),
    }


class _FgCache:
    def __init__(self, bboxes):
        self.bboxes = bboxes

    def bbox(self, dataset, key, source, name):
        start, stop = self.bboxes[(source, name)]
        return np.asarray(start), np.asarray(stop)


def test_crop_class_sampler_uses_bbox_filter_resampling_and_tau(monkeypatch):
    masks = {
        'kept': np.pad(np.ones((1, 1, 2), dtype=np.uint8), ((0, 3), (0, 7), (0, 6))),
        'tiny': np.pad(np.ones((1, 1, 1), dtype=np.uint8), ((0, 3), (0, 7), (0, 7))),
        'zero': np.zeros((4, 8, 8), dtype=np.uint8),
    }
    loaded = []

    def fake_load_mask_crop(*, cls_name, **kwargs):
        loaded.append(cls_name)
        return masks[cls_name]

    monkeypatch.setattr('pumit.ucpt.seg.data._load_mask_crop', fake_load_mask_crop)
    sampler = CropClassSampler(
        data_root=Path('/unused'),
        fg_cache=_FgCache({
            ('src', 'kept'): ([0, 0, 0], [1, 1, 2]),
            ('src', 'tiny'): ([0, 0, 0], [1, 1, 1]),
            ('src', 'zero'): ([0, 0, 0], [1, 1, 1]),
            ('src', 'outside'): ([4, 0, 0], [5, 1, 1]),
        }),
        positive_queries=8,
        negative_queries=8,
        min_positive_voxels=2,
        mask_batch_size=2,
    )
    params = [{
        'crop_size': [4, 8, 8],
        'load_slice_start': [0, 0, 0],
        'load_slice_stop': [4, 8, 8],
        'affine': np.eye(4).ravel().tolist(),
    }]
    classes = sampler(
        {'dataset': 'ds', 'key': 'key', 'shape': [4, 8, 8]},
        {
            'src': {
                'positive': ['kept', 'tiny', 'zero', 'outside'],
                'negative': ['explicit'],
            },
        },
        params,
        np.random.default_rng(0),
        focus=('src', 'kept'),
    )

    by_name = {item['name']: item for item in classes}
    assert sorted(loaded) == ['kept', 'tiny', 'zero']
    assert by_name['kept']['is_positive'] and by_name['kept']['target_voxels'] == 2
    assert not by_name['zero']['is_positive']
    assert not by_name['outside']['is_positive']
    assert not by_name['explicit']['is_positive']
    assert 'tiny' not in by_name
