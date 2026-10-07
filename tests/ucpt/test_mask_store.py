"""Tests for occurrence-level materialized positive-mask storage."""

import os

import numpy as np
import torch

from pumit.ucpt.mask_store import (
    MASK_FRAME_KEY,
    MASK_REF_KEY,
    decode_positive_masks,
    encode_positive_masks,
    read_mask_frame,
    write_mask_shard,
)


def test_positive_mask_frame_round_trip_preserves_non_byte_aligned_rows():
    rng = np.random.default_rng(7)
    masks = [rng.random((3, 5, 11)) > threshold for threshold in (0.3, 0.8)]

    decoded = decode_positive_masks(
        encode_positive_masks(masks),
        count=len(masks),
        shape=masks[0].shape,
    )

    assert decoded.dtype == torch.bool
    assert np.array_equal(decoded.numpy(), np.stack(masks))


def test_mask_shard_round_trip_tracks_each_labeled_occurrence(tmp_path):
    masks = [
        np.eye(5, dtype=np.bool_)[None],
        np.flip(np.eye(5, dtype=np.bool_), axis=1).copy()[None],
    ]
    batches = [{
        'step_idx': 0,
        'samples': [
            {
                'labeled': True,
                'classes': [
                    {'is_positive': True},
                    {'is_positive': False},
                    {'is_positive': True},
                ],
                MASK_FRAME_KEY: encode_positive_masks(masks),
            },
            {
                'labeled': False,
            },
            {
                'labeled': True,
                'classes': [{'is_positive': False}],
                MASK_FRAME_KEY: b'',
            },
        ],
    }]
    path = tmp_path / 'masks' / 'shard_00000.bin'

    stats = write_mask_shard(batches, path)

    positive_sample, unlabeled_sample, negative_sample = batches[0]['samples']
    assert MASK_FRAME_KEY not in positive_sample
    assert MASK_FRAME_KEY not in negative_sample
    assert MASK_REF_KEY not in unlabeled_sample
    assert negative_sample[MASK_REF_KEY][1] == 0
    assert stats == {
        'mask_labeled_samples': 2,
        'mask_positive_masks': 2,
        'mask_storage_bytes': path.stat().st_size,
    }

    fd = os.open(path, os.O_RDONLY)
    try:
        frame = read_mask_frame(fd, positive_sample[MASK_REF_KEY])
        empty_frame = read_mask_frame(fd, negative_sample[MASK_REF_KEY])
    finally:
        os.close(fd)

    decoded = decode_positive_masks(frame, count=2, shape=masks[0].shape)
    assert np.array_equal(decoded.numpy(), np.stack(masks))
    assert empty_frame == b''
    assert decode_positive_masks(empty_frame, count=0, shape=masks[0].shape).shape == (0, 1, 5, 5)
