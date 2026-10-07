"""Latent work planning must keep every GPU batch shape-uniform."""

from __future__ import annotations

from pumit.ucpt.stream.latent_backend import _build_encode_work


def test_latent_bucket_key_includes_inplane_size():
    samples = [
        {
            'da_enc': 0,
            'depth': 64,
            'n_patches': 256,
            'params': [{'patch_size': [64, 128, 128]}],
        },
        {
            'da_enc': 0,
            'depth': 64,
            'n_patches': 576,
            'params': [{'patch_size': [64, 384, 384]}],
        },
    ]

    work = _build_encode_work(
        samples,
        [0, 1],
        shard_id=0,
        memory_budget_gb=80,
    )

    for item in work:
        sizes = {
            samples[index]['params'][0]['patch_size'][1]
            for index in item.selected_indices
        }
        assert len(sizes) == 1
