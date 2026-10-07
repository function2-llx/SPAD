"""Tests for the Universal canonical region bank."""

import pytest

from pumit.spad_unet.universal import (
    CanonicalRegionRegistry,
    foreground_region_names,
    foreground_region_values,
)


@pytest.fixture
def dataset_jsons():
    return {
        '001': {
            'labels': {
                'background': 0,
                'liver': [1, 2],
                'tumor': [2],
            },
        },
        '002': {
            'labels': {
                'background': 0,
                'hepatic organ': [1],
                'vessel': [2],
            },
        },
    }


def test_foreground_regions_preserve_dataset_order(dataset_jsons):
    assert foreground_region_names(dataset_jsons['001']) == ('liver', 'tumor')
    assert foreground_region_values(dataset_jsons['001']) == ((1, 2), (2,))


def test_registry_shares_only_explicit_regions(dataset_jsons):
    registry = CanonicalRegionRegistry(
        dataset_jsons,
        {'organ/liver': {'001': 'liver', '002': 'hepatic organ'}},
    )

    first = registry.dataset_mappings['001']
    second = registry.dataset_mappings['002']
    assert first.canonical_indices[0] == second.canonical_indices[0]
    assert first.canonical_names[1] == 'dataset-001/tumor'
    assert second.canonical_names[1] == 'dataset-002/vessel'
    assert len(registry) == 3


def test_registry_rejects_ambiguous_sharing(dataset_jsons):
    with pytest.raises(ValueError, match='belongs to both'):
        CanonicalRegionRegistry(
            dataset_jsons,
            {
                'organ/liver': {'001': 'liver', '002': 'hepatic organ'},
                'organ/abdomen': {'001': 'liver', '002': 'vessel'},
            },
        )
