"""Tests for the val split used by downstream evaluation probes."""

import numpy as np
import pytest

from pumit.data import build_training_data
from pumit.data.loading import DATA_ROOT


@pytest.fixture(scope="module")
def split_data():
    """Load the real dataset split once for all tests in this module."""
    train, val, info = build_training_data(
        data_root=DATA_ROOT,
        verbose=False,
        max_da=4,
    )
    return train, val, info


class TestValSplit:
    def test_val_larger_than_1000(self, split_data):
        """With ~400K samples and 3%+ holdout, val should be well over 1000."""
        _, val, _ = split_data
        assert len(val) > 1000, f"Val split too small: {len(val)}"

    def test_every_group_has_at_least_one_val(self, split_data):
        """Every (dataset, modality) pair must have at least 1 val sample."""
        train, val, _ = split_data
        all_keys = set(
            zip(train["dataset"], train["modality"])
        ) | set(
            zip(val["dataset"], val["modality"])
        )
        val_keys = set(zip(val["dataset"], val["modality"]))
        missing = all_keys - val_keys
        assert not missing, f"Groups missing from val: {missing}"

    def test_deterministic(self):
        """Same val.json files should produce the exact same split across calls."""
        _, val1, _ = build_training_data(
            data_root=DATA_ROOT,
            verbose=False,
            max_da=4,
        )
        _, val2, _ = build_training_data(
            data_root=DATA_ROOT,
            verbose=False,
            max_da=4,
        )
        assert val1.index.tolist() == val2.index.tolist()

    def test_no_overlap_between_train_and_val(self, split_data):
        """Train and val must be disjoint within each (dataset, modality) group."""
        train, val, _ = split_data
        for (ds, mod), val_group in val.groupby(["dataset", "modality"]):
            train_group = train[(train["dataset"] == ds) & (train["modality"] == mod)]
            overlap = val_group.index.intersection(train_group.index)
            assert len(overlap) == 0, (
                f"Train/val overlap in ({ds}, {mod}): {len(overlap)} samples"
            )
