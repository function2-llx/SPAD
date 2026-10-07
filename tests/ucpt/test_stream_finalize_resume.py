"""Resume aggregate publication after an interrupted stream finalization."""

import json

import pytest
import yaml

from pumit.ucpt.mask_store import MASK_STORAGE
from pumit.ucpt.stream import manifest
from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT


def _source(tmp_path):
    plan = {
        'fingerprint': 'fixture', 'algorithm': manifest.ALGORITHM,
        'packing_policy': 'nearest', 'seed': 42, 'batches_per_shard': 1,
        'budget_ms': 2.0, 'label_budget_fraction': 0.5,
        'cost_model_source': 'fixture', 'cost_model': {}, 'config': {}, 'source_stream': None,
        'input_migration': {'contract': 'pass2-input-v1', 'data_root': '/fixture'},
        'normalization': {'contract': 'fixed-source-v1', 'source_stats_count': 10},
    }
    manifest.ensure_build_plan(tmp_path, plan)
    shard_path = tmp_path / 'shard_00000.msgpack'
    shard_path.write_bytes(b'fixture')
    (tmp_path / 'masks').mkdir()
    mask_path = tmp_path / 'masks/shard_00000.bin'
    mask_path.write_bytes(b'')
    row = {key: 0 for key in manifest._SUM_KEYS}
    row.update({
        'shard_id': 0, 'build_fingerprint': plan['fingerprint'],
        'output_sha256': manifest.sha256_file(shard_path),
        'mask_sha256': manifest.sha256_file(mask_path),
        'mask_storage': MASK_STORAGE, 'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'new_labeled_samples': 1, 'new_unlabeled_samples': 1,
        'n_total_records': 2, 'n_labeled_records': 1,
        'batch_size_histogram': {'2': 1}, 'labeled_per_batch_histogram': {'1': 1},
        'unlabeled_per_batch_histogram': {'1': 1},
    })
    manifest.write_report(tmp_path, row)
    return plan


def _interrupt_before_meta(tmp_path, monkeypatch):
    write_bytes = manifest._atomic_write_bytes

    def interrupted(path, data):
        if path.name == 'meta.yaml':
            raise RuntimeError('interrupted before meta')
        return write_bytes(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(manifest, '_atomic_write_bytes', interrupted)
        with pytest.raises(RuntimeError, match='interrupted before meta'):
            manifest.finalize_stream(tmp_path, 1, 1)
    assert (tmp_path / manifest.MANIFEST_NAME).exists()
    assert (tmp_path / manifest.SUMMARY_NAME).exists()
    assert not (tmp_path / 'meta.yaml').exists()


def test_finalize_resumes_identical_aggregate_files(tmp_path, monkeypatch):
    plan = _source(tmp_path)
    _interrupt_before_meta(tmp_path, monkeypatch)

    summary = manifest.finalize_stream(tmp_path, 1, 1)
    meta = yaml.safe_load((tmp_path / 'meta.yaml').read_text())

    assert summary['shards'] == 1
    assert meta['input_migration'] == plan['input_migration']
    assert meta['normalization'] == plan['normalization']


@pytest.mark.parametrize('artifact', [manifest.MANIFEST_NAME, manifest.SUMMARY_NAME])
def test_finalize_refuses_changed_aggregate_files(tmp_path, monkeypatch, artifact):
    _source(tmp_path)
    _interrupt_before_meta(tmp_path, monkeypatch)
    (tmp_path / artifact).write_text(json.dumps({'changed': True}) + '\n')

    with pytest.raises(ValueError, match='differs from validated shard reports'):
        manifest.finalize_stream(tmp_path, 1, 1)
