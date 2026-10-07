from copy import deepcopy
import json

import msgpack
import numpy as np
import pytest
import yaml

from pumit.ucpt.mask_store import MASK_FRAME_KEY, MASK_REF_KEY, MASK_STORAGE, encode_positive_masks, mask_shard_path, write_mask_shard
from pumit.ucpt.packing import CostModel, PACKING_POLICY, write_shard
from pumit.ucpt.stream.build import _msgpack_equal, _validate_batches
from pumit.ucpt.stream.filtering import UNLABELED_FILTER_CONTRACT, filter_stream, filter_unlabeled_batches
from pumit.ucpt.stream.manifest import ALGORITHM, sha256_file
from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT


def _sample(dataset, key, patches, *, labeled=False):
    sample = {
        'img': f'preprocess/{dataset}/images/{key}.npy',
        'n_patches': patches,
        'labeled': labeled,
        'da_enc': 0,
        'spacing_label': [float('nan'), 1.0, 2.0],
        'params': [
            {'opaque_spatial_state': [1.0, -0.0, 1e-17]},
            {'enabled': True, 'factors': [0.17]},
            {'enabled': False},
            {'enabled': True, 'gammas': [0.91], 'invert': False},
            {},
        ],
        'rope_rescale': 0.75,
    }
    if labeled:
        sample.update({
            'dataset': dataset,
            'key': key,
            'seg_cost_queries': 1,
            'classes': [{'source': dataset, 'name': 'lesion', 'is_positive': True, 'target_voxels': 1}],
            MASK_REF_KEY: [0, 5],
        })
    return sample


def _batches():
    return [
        {
            'step_idx': 0,
            'retained_batch_field': {'tag': 'first'},
            'samples': [
                _sample('Other', 'label_a', 2, labeled=True),
                _sample('Other', 'keep_a', 2),
                _sample('ISLES22', 'remove_a', 3),
                _sample('Other', 'keep_b', 5),
            ],
        },
        {
            'step_idx': 1,
            'retained_batch_field': {'tag': 'second'},
            'samples': [
                _sample('Other', 'label_b', 2, labeled=True),
                _sample('Other', 'keep_c', 7),
                _sample('ISLES22', 'remove_b', 11),
                _sample('ISLES22', 'remove_c', 13),
                _sample('Other', 'keep_d', 17),
            ],
        },
    ]


def test_filter_preserves_samples_batches_and_source_with_merged_row_spans():
    batches = _batches()
    before = deepcopy(batches)

    filtered, spans = filter_unlabeled_batches(batches, ['ISLES22'])

    assert spans == [[0, 2], [5, 17], [41, 58]]
    assert _msgpack_equal(batches, before)
    expected = [
        {**before[0], 'samples': [before[0]['samples'][i] for i in (0, 1, 3)]},
        {**before[1], 'samples': [before[1]['samples'][i] for i in (0, 1, 4)]},
    ]
    assert _msgpack_equal(filtered, expected)


def test_filter_keeps_all_labeled_even_when_the_dataset_is_excluded():
    batches = [{'step_idx': 7, 'samples': [_sample('ISLES22', 'label', 2, labeled=True), _sample('Other', 'keep', 3)]}]

    filtered, spans = filter_unlabeled_batches(batches, ['ISLES22'])

    assert _msgpack_equal(filtered, batches)
    assert spans == [[0, 3]]


def test_filter_without_matching_samples_retains_the_complete_sequence():
    batches = _batches()
    filtered, spans = filter_unlabeled_batches(batches, ['AbsentDataset'])
    assert _msgpack_equal(filtered, batches)
    assert spans == [[0, 58]]


@pytest.mark.parametrize(
    'samples',
    [[], [_sample('Other', 'label', 2, labeled=True)], [_sample('ISLES22', 'remove', 3)]],
)
def test_filter_rejects_a_batch_without_surviving_unlabeled_samples(samples):
    batches = [{'step_idx': 19, 'samples': samples}]
    before = deepcopy(batches)
    with pytest.raises(ValueError, match='batch 19: no unlabeled samples remain'):
        filter_unlabeled_batches(batches, ['ISLES22'])
    assert _msgpack_equal(batches, before)


def _ready_source(tmp_path, *, normalization_location='top', empty_after=False):
    source = tmp_path / 'source'
    source.mkdir()
    rows = []
    for shard_id in range(2):
        batches = _batches()
        if empty_after:
            batches[0]['samples'] = [batches[0]['samples'][0], batches[0]['samples'][2]]
        for batch in batches:
            for sample in batch['samples']:
                if sample['labeled']:
                    del sample[MASK_REF_KEY]
                    sample[MASK_FRAME_KEY] = encode_positive_masks([np.ones((1, 1, 1), dtype=bool)])
        mask_path = mask_shard_path(source, shard_id)
        mask_stats = write_mask_shard(batches, mask_path)
        shard_path = source / f'shard_{shard_id:05d}.msgpack'
        write_shard(batches, shard_path)
        rows.append({
            'shard_id': shard_id,
            'output_sha256': sha256_file(shard_path),
            'n_total_records': 20,
            'n_labeled_records': 9,
            'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
            'mask_storage': MASK_STORAGE,
            'mask_sha256': sha256_file(mask_path),
            **mask_stats,
            **_validate_batches(batches, CostModel.default(), 100.0, shard_id),
        })
    manifest = source / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    normalization = {'contract': 'fixed-source-v1', 'source_stats_count': 123}
    meta = {
        'algorithm': ALGORITHM,
        'stream_complete': True,
        'n_shards': 2,
        'seed': 42,
        'packing_policy': PACKING_POLICY,
        'batches_per_shard': 2,
        'budget_ms': 100.0,
        'label_budget_fraction': 0.75,
        'cost_model_source': 'test',
        'cost_model': CostModel.default().as_fit_result(),
        'config': {'sampler_is_not_used': True},
        'manifest_sha256': sha256_file(manifest),
        'input_migration': {'contract': 'pass2-input-v1', 'data_root': '/unused'},
    }
    if normalization_location == 'top':
        meta['normalization'] = normalization
    else:
        meta['composition'] = {'normalization': normalization}
    (source / 'meta.yaml').write_text(yaml.safe_dump(meta))
    (source / 'READY.json').write_text('{}\n')
    return source, rows, meta


@pytest.mark.parametrize('normalization_location', ['top', 'composition'])
@pytest.mark.parametrize('workers', [1, 2])
def test_filter_stream_reuses_masks_and_preserves_labeled_payloads(tmp_path, normalization_location, workers):
    source, rows, source_meta = _ready_source(tmp_path, normalization_location=normalization_location)
    output = tmp_path / 'filtered'
    source_bytes = [(source / f'shard_{i:05d}.msgpack').read_bytes() for i in range(2)]

    filter_stream(source_stream=source, output_dir=output, exclude_datasets=['ISLES22'], workers=workers)

    plan = yaml.safe_load((output / 'build.yaml').read_text())
    assert plan['unlabeled_filter'] == {'contract': UNLABELED_FILTER_CONTRACT, 'exclude_datasets': ['ISLES22']}
    assert plan['normalization'] == {'contract': 'fixed-source-v1', 'source_stats_count': 123}
    assert plan['source_stream'] == str(source)
    assert plan['source_meta_sha256'] == sha256_file(source / 'meta.yaml')
    assert 'input_migration' not in plan
    assert 'composition' not in plan
    for name in ('seed', 'config', 'budget_ms', 'label_budget_fraction', 'cost_model_source', 'cost_model'):
        assert plan[name] == source_meta[name]
    assert not (output / 'latents').exists()
    assert not (output / 'READY.json').exists()
    for shard_id in range(2):
        source_path = source / f'shard_{shard_id:05d}.msgpack'
        output_path = output / source_path.name
        before = msgpack.unpackb(source_bytes[shard_id], raw=False)['batches']
        after = msgpack.unpackb(output_path.read_bytes(), raw=False)['batches']
        expected, spans = filter_unlabeled_batches(before, ['ISLES22'])
        assert _msgpack_equal(after, expected)
        assert source_path.read_bytes() == source_bytes[shard_id]
        source_mask = mask_shard_path(source, shard_id)
        output_mask = mask_shard_path(output, shard_id)
        assert output_mask.samefile(source_mask)
        assert sha256_file(output_mask) == rows[shard_id]['mask_sha256']
        old_labeled = [sample for batch in before for sample in batch['samples'] if sample['labeled']]
        new_labeled = [sample for batch in after for sample in batch['samples'] if sample['labeled']]
        assert _msgpack_equal(old_labeled, new_labeled)
        report = json.loads((output / 'reports' / f'shard_{shard_id:05d}.json').read_text())
        assert report['old_labeled_samples'] == report['used_old_labeled_samples'] == report['new_labeled_samples'] == 2
        assert report['generated_labeled_samples'] == report['generated_labeled_patches'] == 0
        assert report['old_unlabeled_samples'] == 7
        assert report['used_old_unlabeled_samples'] == 4
        assert report['dropped_old_unlabeled_samples'] == 3
        assert report['old_unlabeled_patches'] == 58
        assert report['used_old_unlabeled_patches'] == report['logical_latent_rows'] == 31
        assert report['dropped_old_unlabeled_patches'] == 27
        assert report['generated_unlabeled_samples'] == report['generated_unlabeled_patches'] == 0
        assert report['source_latent_row_spans'] == spans
        for name in ('mask_sha256', 'mask_storage_bytes', 'mask_labeled_samples', 'mask_positive_masks', 'n_total_records', 'n_labeled_records'):
            assert report[name] == rows[shard_id][name]


def test_filter_stream_shards_limit_and_completed_resume(tmp_path):
    source, _, _ = _ready_source(tmp_path)
    output = tmp_path / 'filtered'
    kwargs = {'source_stream': source, 'output_dir': output, 'exclude_datasets': ['ISLES22'], 'shards': 1}
    filter_stream(**kwargs)
    shard = output / 'shard_00000.msgpack'
    inode = shard.stat().st_ino
    assert not (output / 'shard_00001.msgpack').exists()
    filter_stream(**kwargs)
    assert shard.stat().st_ino == inode


def test_filter_stream_stops_before_writing_a_shard_if_unlabeled_becomes_empty(tmp_path):
    source, _, _ = _ready_source(tmp_path, empty_after=True)
    output = tmp_path / 'filtered'
    with pytest.raises(ValueError, match='no unlabeled samples remain'):
        filter_stream(source_stream=source, output_dir=output, exclude_datasets=['ISLES22'], shards=1)
    assert not (output / 'shard_00000.msgpack').exists()
    assert not mask_shard_path(output, 0).exists()
    assert not (output / 'reports' / 'shard_00000.json').exists()
