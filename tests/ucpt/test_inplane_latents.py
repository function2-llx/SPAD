from contextlib import contextmanager
import json
import math
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from safetensors.torch import load_file, save_file
import torch
import yaml

import pumit.ucpt.stream.latent_backend as latent_backend
from pumit.ucpt.affine import _affine_load_extent
from pumit.ucpt.stream.latent_backend import (
    CANONICAL_INPLANE_LATENT_CONTRACT,
    LatentEncodeWork,
    _build_encode_work,
    _canonical_inplane_selection,
    _clean_inplane_temps,
    _json_sha256,
    _materialize_latent_parts,
    _publish_inplane_latents,
    _recover_inplane_rows,
    _requires_canonical_inplane_latent,
    _validate_published_inplane_latents,
    _validate_unselected_latent_rows,
    _write_latent_part,
)
from pumit.ucpt.stream.metadata import canonicalize_sample_metadata


def _sample(matrix: np.ndarray, *, n_patches: int = 1) -> dict:
    affine = np.eye(4)
    affine[:3, :3] = matrix
    crop_size = [1, 16, 16]
    load_extent = _affine_load_extent(matrix, crop_size)
    return {
        'labeled': False,
        'n_patches': n_patches,
        'depth': 1,
        'da_enc': None,
        'params': [{
            'crop_size': crop_size,
            'affine': affine.ravel().tolist(),
            'load_slice_start': [0, 0, 0],
            'load_slice_stop': load_extent.tolist(),
        }],
    }


def test_startup_cleans_only_owned_shard_temps(tmp_path):
    staging = tmp_path / 'staging'
    owned_output = staging / 'shard_00000.safetensors.tmp'
    other_output = staging / 'shard_00001.safetensors.tmp'
    owned_part = staging / '.migration-latent-parts/shard_00000/.tmp-owned'
    other_part = staging / '.migration-latent-parts/shard_00001/.tmp-other'
    for path in (owned_output, other_output, owned_part, other_part):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'interrupted')

    removed = _clean_inplane_temps(staging, [{'shard_id': 0}])

    assert set(removed) == {owned_output, owned_part}
    assert not owned_output.exists()
    assert not owned_part.exists()
    assert other_output.exists()
    assert other_part.exists()


def _inplane_matrix() -> np.ndarray:
    theta = math.radians(31)
    matrix = np.eye(3)
    matrix[1:, 1:] = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]
    matrix[0, 1] = 2e-17
    matrix[2, 0] = -3e-17
    return matrix


def test_metadata_selector_exactly_matches_replay_changes(tmp_path):
    inplane = _sample(_inplane_matrix(), n_patches=2)
    diagonal = _sample(np.eye(3), n_patches=3)
    oblique_matrix = _inplane_matrix()
    oblique_matrix[0, 1] = 1e-3
    oblique = _sample(oblique_matrix, n_patches=4)
    snapped_diagonal = _sample(np.diag([1.0, 15.00001 / 16, 15.00001 / 16]))
    snapped_diagonal['params'][0]['load_slice_stop'] = [1, 16, 16]
    source_samples = [diagonal, inplane, oblique, snapped_diagonal]
    target_samples = [canonicalize_sample_metadata(sample)[0] for sample in source_samples]
    source = tmp_path / 'source'
    stream = tmp_path / 'stream'
    source.mkdir()
    stream.mkdir()
    (source / 'shard_00000.msgpack').write_bytes(
        msgpack.packb({'batches': [{'samples': source_samples}]})
    )
    shard_path = stream / 'shard_00000.msgpack'
    shard_path.write_bytes(msgpack.packb({'batches': [{'samples': target_samples}]}))
    (stream / 'meta.yaml').write_text(yaml.safe_dump({'source_stream': str(source)}))

    unlabeled, selected = _canonical_inplane_selection(
        shard_path,
        {'logical_latent_rows': 10},
    )

    assert selected == [1]
    assert len(unlabeled) == 4
    assert _requires_canonical_inplane_latent(inplane)
    assert not _requires_canonical_inplane_latent(diagonal)
    assert not _requires_canonical_inplane_latent(oblique)


def test_selective_materialization_preserves_every_unselected_row(tmp_path):
    source_path = tmp_path / 'source.safetensors'
    staging = tmp_path / 'staging'
    staging.mkdir()
    output_path = staging / 'shard_00000.safetensors'
    source = torch.arange(6 * 32, dtype=torch.float16).reshape(6, 32)
    save_file({'latents': source}, source_path)
    unlabeled = [{'n_patches': 2}, {'n_patches': 3}, {'n_patches': 1}]
    work = LatentEncodeWork(0, 0, (1,), 3, 1)
    replacement = torch.full((3, 32), 17, dtype=torch.float16)
    plan_sha256 = 'a' * 64
    _write_latent_part(staging, work, replacement, plan_sha256=plan_sha256)

    _materialize_latent_parts(
        source_path,
        unlabeled,
        source_prefix_samples=len(unlabeled),
        work=[work],
        stream_dir=staging,
        output_path=output_path,
        plan_sha256=plan_sha256,
    )
    _validate_unselected_latent_rows(source_path, output_path, unlabeled, [1])

    actual = load_file(output_path)['latents']
    assert torch.equal(actual[:2], source[:2])
    assert torch.equal(actual[2:5], replacement)
    assert torch.equal(actual[5:], source[5:])

    actual[0, 0] += 1
    bad_path = staging / 'bad.safetensors'
    save_file({'latents': actual}, bad_path)
    with pytest.raises(ValueError, match='unselected latent rows differ'):
        _validate_unselected_latent_rows(source_path, bad_path, unlabeled, [1])


def _publication_case(tmp_path, *, write_report: bool = True):
    stream = tmp_path / 'stream'
    staging = stream / '.canonical-inplane-latents.staging'
    staging.mkdir(parents=True)
    source = tmp_path / 'source'
    source.mkdir()
    source_samples = [_sample(_inplane_matrix()), _sample(np.eye(3))]
    samples = [canonicalize_sample_metadata(sample)[0] for sample in source_samples]
    (source / 'shard_00000.msgpack').write_bytes(
        msgpack.packb({'batches': [{'samples': source_samples}]})
    )
    (stream / 'meta.yaml').write_text(yaml.safe_dump({'source_stream': str(source)}))
    (stream / 'shard_00000.msgpack').write_bytes(
        msgpack.packb({'batches': [{'samples': samples}]})
    )
    row = {'shard_id': 0, 'logical_latent_rows': 2}
    output_path = staging / 'shard_00000.safetensors'
    save_file({'latents': torch.zeros(2, 32, dtype=torch.float16)}, output_path)
    plan = {
        'latent_contract': CANONICAL_INPLANE_LATENT_CONTRACT,
        'codec_model': 'flux2',
        'codec_checkpoint': '/checkpoint.pt',
        'source_stream': '/source-stream',
        'source_manifest_sha256': 'b' * 64,
        'source_latent_dir': '/source-latents',
        'source_latent_receipt_sha256': 'c' * 64,
    }
    (staging / 'plan.json').write_text(json.dumps(plan))
    plan_sha256 = _json_sha256(plan)
    if write_report:
        report_dir = staging / 'reports'
        report_dir.mkdir()
        (report_dir / 'shard_00000.json').write_text(json.dumps({
            'plan_sha256': plan_sha256,
            'shard_id': 0,
            'logical_latent_rows': 2,
            'affected_unlabeled_samples': 1,
            'affected_latent_rows': 1,
            'output_size_bytes': output_path.stat().st_size,
        }))
    return stream, staging, source_samples, samples, row, plan, plan_sha256


def test_resume_recovers_output_written_before_shard_report(tmp_path):
    stream, staging, _, samples, row, _, plan_sha256 = _publication_case(
        tmp_path,
        write_report=False,
    )
    source_latents = tmp_path / 'source-latents'
    source_latents.mkdir()
    save_file(
        {'latents': torch.zeros(2, 32, dtype=torch.float16)},
        source_latents / 'shard_00000.safetensors',
    )
    work = _build_encode_work(samples, [0], shard_id=0, memory_budget_gb=0.01)
    _write_latent_part(
        staging,
        work[0],
        torch.zeros(work[0].encoded_rows, 32, dtype=torch.float16),
        plan_sha256=plan_sha256,
    )

    recovered, pending = _recover_inplane_rows(
        stream,
        source_latents,
        staging,
        [row],
        plan_sha256=plan_sha256,
        memory_budget_gb=0.01,
    )

    assert recovered == [0]
    assert pending == []
    report = json.loads((staging / 'reports' / 'shard_00000.json').read_text())
    assert report['affected_unlabeled_samples'] == 1
    assert report['affected_latent_rows'] == 1
    assert not (staging / '.migration-latent-parts').exists()


def test_resume_rejects_unverifiable_output_without_shard_report(tmp_path):
    stream, staging, _, _, row, _, plan_sha256 = _publication_case(
        tmp_path,
        write_report=False,
    )
    source_latents = tmp_path / 'source-latents'
    source_latents.mkdir()
    save_file(
        {'latents': torch.zeros(2, 32, dtype=torch.float16)},
        source_latents / 'shard_00000.safetensors',
    )
    output = load_file(staging / 'shard_00000.safetensors')['latents']
    output[1, 0] = 1
    (staging / 'shard_00000.safetensors').unlink()
    save_file({'latents': output}, staging / 'shard_00000.safetensors')

    with pytest.raises(ValueError, match='unselected latent rows differ'):
        _recover_inplane_rows(
            stream,
            source_latents,
            staging,
            [row],
            plan_sha256=plan_sha256,
            memory_budget_gb=0.01,
        )
    assert not (staging / 'reports' / 'shard_00000.json').exists()


def test_resume_trusts_committed_receipt_without_replaying_selection(tmp_path, monkeypatch):
    stream, staging, _, _, row, _, plan_sha256 = _publication_case(tmp_path)
    source_latents = tmp_path / 'source-latents'
    source_latents.mkdir()
    monkeypatch.setattr(
        latent_backend,
        '_canonical_inplane_selection',
        lambda *args: pytest.fail('completed receipts must not replay sample selection'),
    )

    recovered, pending = _recover_inplane_rows(
        stream,
        source_latents,
        staging,
        [row],
        plan_sha256=plan_sha256,
        memory_budget_gb=0.01,
    )

    assert recovered == []
    assert pending == []


def test_restart_partitions_only_pending_shards(tmp_path, monkeypatch):
    stream = tmp_path / 'stream'
    stream.mkdir()
    rows = [
        {'shard_id': shard_id, 'logical_latent_rows': 1}
        for shard_id in range(8)
    ]
    (stream / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )
    source_latents = tmp_path / 'source-latents'
    source_latents.mkdir()
    checkpoint = tmp_path / 'codec.pt'
    checkpoint.write_bytes(b'checkpoint')
    completed_rows = rows[::2]
    pending_rows = rows[1::2]
    encoded = []
    prepare_calls = []
    distributed = {'initialized': False}
    provenance = {
        'stream_manifest_sha256': 'a' * 64,
        'source_stream': '/source-stream',
        'source_stream_fingerprint': 'fingerprint',
        'source_manifest_sha256': 'b' * 64,
        'source_latent_receipt_sha256': 'c' * 64,
    }
    plan = {
        'latent_contract': CANONICAL_INPLANE_LATENT_CONTRACT,
        'stream': str(stream.resolve()),
        **provenance,
        'source_latent_dir': str(source_latents.resolve()),
        'codec_model': 'flux2',
        'codec_checkpoint': str(checkpoint.resolve()),
        'codec_checkpoint_sha256': latent_backend.sha256_file(checkpoint),
        'compile_mode': 'default',
        'memory_budget_gb': 80,
        'materialized_shards': list(range(8)),
    }
    staging = stream / '.canonical-inplane-latents.staging'
    report_dir = staging / 'reports'
    report_dir.mkdir(parents=True)
    (staging / 'plan.json').write_text(json.dumps(plan))
    plan_sha256 = _json_sha256(plan)
    for row in completed_rows:
        output_path = staging / f"shard_{row['shard_id']:05d}.safetensors"
        save_file({'latents': torch.zeros(1, 32, dtype=torch.float16)}, output_path)
        (report_dir / f"shard_{row['shard_id']:05d}.json").write_text(
            json.dumps(
                {
                    'plan_sha256': plan_sha256,
                    'shard_id': row['shard_id'],
                    'logical_latent_rows': 1,
                    'affected_unlabeled_samples': 0,
                    'affected_latent_rows': 0,
                    'output_size_bytes': output_path.stat().st_size,
                }
            )
        )

    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1')
    monkeypatch.setattr(
        latent_backend,
        '_validate_canonical_inplane_source',
        lambda *args, **kwargs: provenance,
    )

    def prepare(*args, **kwargs):
        prepare_calls.append(None)
        return completed_rows, pending_rows

    monkeypatch.setattr(latent_backend, '_prepare_inplane_rows', prepare)
    monkeypatch.setattr(
        latent_backend.dist,
        'init_process_group',
        lambda *args, **kwargs: distributed.update(initialized=True),
    )
    monkeypatch.setattr(
        latent_backend.dist,
        'is_initialized',
        lambda: distributed['initialized'],
    )
    monkeypatch.setattr(latent_backend.dist, 'broadcast_object_list', lambda *args, **kwargs: None)
    monkeypatch.setattr(latent_backend.dist, 'barrier', lambda: None)
    monkeypatch.setattr(
        latent_backend.dist,
        'destroy_process_group',
        lambda: distributed.update(initialized=False),
    )

    @contextmanager
    def fake_gpu_pool(*args, **kwargs):
        yield object(), 1, None

    monkeypatch.setattr(latent_backend, '_latent_gpu_pool', fake_gpu_pool)
    monkeypatch.setattr(latent_backend, '_canonical_inplane_selection', lambda *args: ([], []))

    def encode(row, *args, **kwargs):
        encoded.append(row['shard_id'])
        return {'samples': 0, 'latent_rows': 0}

    monkeypatch.setattr(latent_backend, '_encode_selected_latent_shard', encode)
    monkeypatch.setattr(latent_backend, '_validate_unselected_latent_rows', lambda *args: None)
    monkeypatch.setattr(latent_backend, '_write_inplane_shard_report', lambda *args: None)
    monkeypatch.setattr(latent_backend, '_validate_inplane_shard', lambda *args: None)

    latent_backend._encode_canonical_inplane_latents(
        SimpleNamespace(
            stream=stream,
            source_latent_dir=source_latents,
            codec_checkpoint=checkpoint,
            codec_model='flux2',
            shard_offset=0,
            shards=None,
            compile_mode='default',
            memory_budget_gb=80,
        ),
        SimpleNamespace(capacity=4),
    )

    assert encoded == [1, 5]
    assert len(prepare_calls) == 1
    assert not distributed['initialized']


def test_migration_gpu_worker_releases_cache_at_shard_boundaries(monkeypatch):
    calls = []
    monkeypatch.setattr(latent_backend.torch.cuda, 'empty_cache', lambda: calls.append(None))
    monkeypatch.setattr(latent_backend, '_LATENT_WORKER_ACTIVE_SHARD', None)

    latent_backend._prepare_worker_shard(3)
    latent_backend._prepare_worker_shard(3)
    latent_backend._prepare_worker_shard(19)

    assert calls == [None, None]


def test_publish_is_one_atomic_directory_rename_after_all_receipts(tmp_path):
    stream, staging, _, _, row, plan, _ = _publication_case(tmp_path)

    _publish_inplane_latents(stream, staging, [row], plan)

    assert not staging.exists()
    assert (stream / 'latents' / 'shard_00000.safetensors').exists()
    inner = stream / 'latents' / 'latent-materialized.json'
    outer = stream / 'latent-materialized.json'
    assert inner.stat().st_ino == outer.stat().st_ino
    receipt = json.loads(outer.read_text())
    assert receipt['latent_contract'] == CANONICAL_INPLANE_LATENT_CONTRACT
    assert receipt['affected_unlabeled_samples'] == 1
    assert receipt['affected_latent_rows'] == 1


def test_resume_finishes_outer_receipt_after_directory_rename(tmp_path, monkeypatch):
    stream, staging, _, _, row, plan, _ = _publication_case(tmp_path)
    real_link = latent_backend.os.link

    def crash_before_outer_receipt(source, destination):
        if destination == stream / 'latent-materialized.json':
            raise RuntimeError('crash')
        return real_link(source, destination)

    monkeypatch.setattr(latent_backend.os, 'link', crash_before_outer_receipt)

    with pytest.raises(RuntimeError, match='crash'):
        _publish_inplane_latents(stream, staging, [row], plan)

    assert not staging.exists()
    assert (stream / 'latents' / 'latent-materialized.json').exists()
    assert not (stream / 'latent-materialized.json').exists()
    monkeypatch.setattr(latent_backend.os, 'link', real_link)

    recovered = _validate_published_inplane_latents(
        stream,
        [row],
        plan,
        recover_missing_outer=True,
    )

    assert recovered
    inner = stream / 'latents' / 'latent-materialized.json'
    outer = stream / 'latent-materialized.json'
    assert inner.stat().st_ino == outer.stat().st_ino


def test_resume_does_not_publish_an_invalid_inner_receipt(tmp_path, monkeypatch):
    stream, staging, _, _, row, plan, _ = _publication_case(tmp_path)
    real_link = latent_backend.os.link

    def crash_before_outer_receipt(source, destination):
        if destination == stream / 'latent-materialized.json':
            raise RuntimeError('crash')
        return real_link(source, destination)

    monkeypatch.setattr(latent_backend.os, 'link', crash_before_outer_receipt)
    with pytest.raises(RuntimeError, match='crash'):
        _publish_inplane_latents(stream, staging, [row], plan)
    inner = stream / 'latents' / 'latent-materialized.json'
    receipt = json.loads(inner.read_text())
    receipt['affected_latent_rows'] = 2
    inner.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match='receipt differs'):
        _validate_published_inplane_latents(
            stream,
            [row],
            plan,
            recover_missing_outer=True,
        )
    assert not (stream / 'latent-materialized.json').exists()
