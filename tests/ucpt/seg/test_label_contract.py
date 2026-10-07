import pytest

from pumit.ucpt.seg.label_contract import build_label_manifest, normalize_label_contract


def _record(
    dataset: str,
    key: str,
    *,
    positive: list[str],
    negative: list[str],
) -> dict:
    return {
        'dataset': dataset,
        'key': key,
        'label': True,
        'label_classes': {
            'source': {
                'positive': positive,
                'negative': negative,
            },
        },
    }


def test_build_label_manifest_keeps_positive_and_negative_only_records():
    positive = _record('dataset', 'positive', positive=['liver'], negative=['kidney'])
    negative_only = _record('dataset', 'negative', positive=[], negative=['liver'])
    unlabeled = {
        'dataset': 'dataset',
        'key': 'unlabeled',
        'label': False,
        'label_classes': {},
    }

    manifest = build_label_manifest([positive, negative_only, unlabeled])

    assert manifest == {
        ('dataset', 'positive'): {
            'source': {
                'positive': ['liver'],
                'negative': ['kidney'],
            },
        },
        ('dataset', 'negative'): {
            'source': {
                'positive': [],
                'negative': ['liver'],
            },
        },
    }


def test_build_label_manifest_rejects_duplicate_record_identity():
    record = _record('dataset', 'key', positive=['liver'], negative=[])

    with pytest.raises(ValueError, match='duplicate label manifest key'):
        build_label_manifest([record, record])


def test_normalize_label_contract_rejects_positive_negative_overlap():
    record = _record('dataset', 'key', positive=['liver'], negative=['liver'])

    with pytest.raises(ValueError, match='both positive and negative'):
        normalize_label_contract(record)
