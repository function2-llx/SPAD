"""Tests for the Universal replay stream generation and reading."""

from __future__ import annotations

import hashlib

import msgpack
import pytest
import yaml

from pumit.spad_unet.replay import (
    CaseLoaderConfiguration,
    COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    ReplayDataLoader,
    ReplayRecord,
    ReplayStreamReader,
    SQRT_DATASET_REPLAY_FORMAT_VERSION,
    SQRT_DATASET_REPLAY_RULE,
    generate_sqrt_replay_stream,
    generate_replay_stream,
    replay_fingerprint,
    replay_group_count_for_budget,
    sqrt_replay_generation_fingerprint,
    verify_legacy_replay_generation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stream_config():
    """Minimal synthetic config for replay stream tests."""
    dataset_ids = ('001', '002', '003', '004')
    training_identifiers = {
        '001': ('case_a', 'case_b', 'case_c', 'case_d'),
        '002': ('case_x', 'case_y'),
        '003': ('case_p', 'case_q', 'case_r'),
        '004': ('case_u', 'case_v'),
    }
    return {
        'seed': 12345,
        'total_steps': 100,
        'dataset_ids': dataset_ids,
        'training_identifiers': training_identifiers,
        'foreground_samples_per_batch': 2,
        'sampling_rule': COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    }


def test_replay_group_count_budget_is_independent_of_optimizer_batch_boundaries():
    assert replay_group_count_for_budget(250_000, 24, 12, group_multiple=2) == 500_000
    assert replay_group_count_for_budget(5, 8, 12, group_multiple=2) == 4

    with pytest.raises(ValueError, match='positive integer'):
        replay_group_count_for_budget(250_000, 0, 12, group_multiple=2)


@pytest.fixture
def generated_stream(tmp_path, stream_config):
    """Generate a stream and return (output_dir, config)."""
    output_dir = tmp_path / 'stream'
    generate_replay_stream(
        output_dir=output_dir,
        steps_per_shard=30,
        **stream_config,
    )
    return output_dir, stream_config


def group_at(reader: ReplayStreamReader, step: int) -> list[ReplayRecord]:
    """Return the records of one emission group."""
    return reader.records_at(
        step * reader.logical_batch_size,
        reader.logical_batch_size,
    )


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


def test_fingerprint_deterministic(stream_config):
    fp1 = replay_fingerprint(**stream_config)
    fp2 = replay_fingerprint(**stream_config)
    assert fp1 == fp2


def test_fingerprint_sensitivity(stream_config):
    """Changing any input changes the fingerprint."""
    base_fp = replay_fingerprint(**stream_config)

    # Change seed
    assert replay_fingerprint(**{**stream_config, 'seed': 99999}) != base_fp
    # Change total_steps
    assert replay_fingerprint(**{**stream_config, 'total_steps': 50}) != base_fp
    # Change foreground count
    assert replay_fingerprint(
        **{**stream_config, 'foreground_samples_per_batch': 1}
    ) != base_fp
    # Change dataset_ids
    reordered_ids = ('002', '001', '003', '004')
    assert replay_fingerprint(**{**stream_config, 'dataset_ids': reordered_ids}) != base_fp
    # Change training identifiers
    modified_ids = {**stream_config['training_identifiers'], '001': ('case_a', 'case_b', 'EXTRA')}
    assert replay_fingerprint(**{**stream_config, 'training_identifiers': modified_ids}) != base_fp


def test_verify_legacy_replay_generation_binds_by_path(generated_stream):
    output_dir, config = generated_stream
    reader = ReplayStreamReader(
        output_dir,
        valid_identifiers=config['training_identifiers'],
    )

    generation = verify_legacy_replay_generation(
        reader,
        config['training_identifiers'],
    )

    assert generation == {
        'seed': config['seed'],
        'replay_total_groups': config['total_steps'],
        'samples_per_dataset': 1,
        'foreground_samples_per_batch': config['foreground_samples_per_batch'],
        'sampling_rule': COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    }


def test_verify_legacy_replay_generation_rejects_foreign_suites(generated_stream):
    output_dir, config = generated_stream
    reader = ReplayStreamReader(output_dir)
    foreign_identifiers = {
        **config['training_identifiers'],
        '001': ('case_a', 'case_b', 'case_c', 'case_d', 'EXTRA'),
    }

    with pytest.raises(ValueError, match='generation contract'):
        verify_legacy_replay_generation(reader, foreign_identifiers)


def test_verify_legacy_replay_generation_rejects_exact_content_streams(
    tmp_path,
    stream_config,
):
    output_dir = tmp_path / 'sqrt'
    generate_sqrt_replay_stream(
        seed=stream_config['seed'],
        total_records=6_000,
        dataset_ids=stream_config['dataset_ids'],
        training_identifiers=stream_config['training_identifiers'],
        output_dir=output_dir,
        records_per_shard=3_000,
        foreground_probability=0.5,
        statistics_prefixes={},
    )
    reader = ReplayStreamReader(output_dir)

    with pytest.raises(ValueError, match='stream_fingerprint'):
        verify_legacy_replay_generation(
            reader,
            stream_config['training_identifiers'],
        )


def test_legacy_generation_fingerprints_and_shard_bytes_are_stable(
    tmp_path,
    stream_config,
):
    contracts = {
        'complementary': (
            {
                'foreground_samples_per_batch': len(stream_config['dataset_ids']) // 2,
                'sampling_rule': COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
            },
            '3014873b38aeefe3aebac03a4a686ee8',
            (
                '43964dcca48a2dfb0c88e1a4abbdfe272168228ab83c458c5ac1a11f7eee8ab3',
                'bd0a8cc063feb2dad3db16919ffc7602f3866ebfdb40a1e6f84932469f45b2fc',
                'b3a1674e5e621d51369685dd391068f8c2cc9768ebd3993761f6cefbaa767f44',
                '9dcb7d98782386e504b909236984b1fcfe1d713c0bad0d59b333aa49df797581',
            ),
        ),
    }
    for name, (overrides, expected_fingerprint, expected_shard_hashes) in contracts.items():
        config = {**stream_config, **overrides}
        output_dir = tmp_path / name
        generate_replay_stream(
            output_dir=output_dir,
            steps_per_shard=30,
            **config,
        )
        assert replay_fingerprint(**config) == expected_fingerprint
        assert tuple(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_dir.glob('shard_*.msgpack'))
        ) == expected_shard_hashes


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------


def test_generate_creates_shards_and_meta(generated_stream):
    output_dir, config = generated_stream
    assert (output_dir / 'meta.yaml').exists()
    # 100 steps / 30 per shard = 4 shards (30+30+30+10)
    for i in range(4):
        assert (output_dir / f'shard_{i:05d}.msgpack').exists()
    assert not (output_dir / 'shard_00004.msgpack').exists()


def test_generate_is_deterministic(tmp_path, stream_config):
    """Two runs with same seed produce identical shards."""
    dir1 = tmp_path / 'run1'
    dir2 = tmp_path / 'run2'
    generate_replay_stream(output_dir=dir1, steps_per_shard=50, **stream_config)
    generate_replay_stream(output_dir=dir2, steps_per_shard=50, **stream_config)

    for shard_file in sorted(dir1.glob('shard_*.msgpack')):
        other = dir2 / shard_file.name
        assert shard_file.read_bytes() == other.read_bytes(), f'{shard_file.name} differs'


def test_generate_sqrt_replay_uses_flat_v6_contract(tmp_path, stream_config):
    output_dir = tmp_path / 'sqrt'
    total_records = 20_000
    meta = generate_sqrt_replay_stream(
        seed=stream_config['seed'],
        total_records=total_records,
        dataset_ids=stream_config['dataset_ids'],
        training_identifiers=stream_config['training_identifiers'],
        output_dir=output_dir,
        records_per_shard=3_000,
        foreground_probability=1 / 3,
        statistics_prefixes={'gb8': 8_000, 'gb16': 16_000},
    )

    assert meta['format_version'] == SQRT_DATASET_REPLAY_FORMAT_VERSION
    assert meta['sampling_rule'] == SQRT_DATASET_REPLAY_RULE
    assert meta['logical_batch_size'] == 1
    assert meta['total_steps'] == total_records
    assert meta['total_records'] == total_records
    assert meta['foreground_probability'] == pytest.approx(1 / 3)
    assert meta['fingerprint'] == meta['generation_fingerprint']
    assert len(meta['stream_fingerprint']) == 64
    assert meta['statistics_prefix_records'] == {
        'full': total_records,
        'gb8': 8_000,
        'gb16': 16_000,
    }

    reader = ReplayStreamReader(
        output_dir,
        expected_fingerprint=meta['generation_fingerprint'],
        valid_identifiers=stream_config['training_identifiers'],
    )
    reader.validate()
    assert reader.total_records == total_records
    assert reader.stream_fingerprint == meta['stream_fingerprint']
    records = reader.records_at(0, total_records)

    expected_weights = {
        '001': 2.0,
        '002': 2 ** 0.5,
        '003': 3 ** 0.5,
        '004': 2 ** 0.5,
    }
    weight_sum = sum(expected_weights.values())
    for dataset_id, weight in expected_weights.items():
        actual_fraction = sum(
            record.dataset_id == dataset_id for record in records
        ) / total_records
        assert actual_fraction == pytest.approx(weight / weight_sum, abs=0.01)
    assert sum(record.force_foreground for record in records) / total_records == pytest.approx(
        1 / 3,
        abs=0.01,
    )
    assert all(
        record.case_id in stream_config['training_identifiers'][record.dataset_id]
        for record in records
    )


def test_sqrt_replay_content_is_independent_of_shard_layout(tmp_path, stream_config):
    common = {
        'seed': stream_config['seed'],
        'total_records': 1_003,
        'dataset_ids': stream_config['dataset_ids'],
        'training_identifiers': stream_config['training_identifiers'],
        'foreground_probability': 1 / 3,
    }
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'
    first = generate_sqrt_replay_stream(
        output_dir=first_dir,
        records_per_shard=97,
        **common,
    )
    second = generate_sqrt_replay_stream(
        output_dir=second_dir,
        records_per_shard=211,
        **common,
    )

    assert first['generation_fingerprint'] == second['generation_fingerprint']
    assert first['stream_fingerprint'] == second['stream_fingerprint']
    assert ReplayStreamReader(first_dir).records_at(0, 1_003) == (
        ReplayStreamReader(second_dir).records_at(0, 1_003)
    )


def test_sqrt_replay_statistics_match_exact_prefix_counts(tmp_path, stream_config):
    output_dir = tmp_path / 'statistics'
    meta = generate_sqrt_replay_stream(
        seed=stream_config['seed'],
        total_records=53,
        dataset_ids=stream_config['dataset_ids'],
        training_identifiers=stream_config['training_identifiers'],
        output_dir=output_dir,
        records_per_shard=11,
        statistics_prefixes={'short': 17, 'medium': 38},
    )
    records = ReplayStreamReader(output_dir).records_at(0, 53)

    for label, prefix_length in meta['statistics_prefix_records'].items():
        prefix = records[:prefix_length]
        statistics = meta['statistics'][label]
        assert statistics['records'] == prefix_length
        assert statistics['foreground_count'] == sum(
            record.force_foreground for record in prefix
        )
        for dataset_id in stream_config['dataset_ids']:
            selected = [
                record for record in prefix if record.dataset_id == dataset_id
            ]
            assert statistics['dataset_counts'][dataset_id] == len(selected)
            assert statistics['dataset_foreground_counts'][dataset_id] == sum(
                record.force_foreground for record in selected
            )


def test_sqrt_generation_fingerprint_tracks_generation_contract(stream_config):
    common = {
        'seed': stream_config['seed'],
        'total_records': 100,
        'dataset_ids': stream_config['dataset_ids'],
        'training_identifiers': stream_config['training_identifiers'],
        'foreground_probability': 1 / 3,
    }
    fingerprint = sqrt_replay_generation_fingerprint(**common)
    assert sqrt_replay_generation_fingerprint(**common) == fingerprint
    assert sqrt_replay_generation_fingerprint(
        **{**common, 'total_records': 101}
    ) != fingerprint
    assert sqrt_replay_generation_fingerprint(
        **{**common, 'foreground_probability': 0.5}
    ) != fingerprint
    assert sqrt_replay_generation_fingerprint(
        **{
            **common,
            'training_identifiers': {
                **stream_config['training_identifiers'],
                '001': ('case_a',),
            },
        }
    ) != fingerprint


def test_generate_total_records(generated_stream):
    """Total records = total_steps * logical_batch_size."""
    output_dir, config = generated_stream
    total_records = 0
    for shard_file in sorted(output_dir.glob('shard_*.msgpack')):
        with open(shard_file, 'rb') as f:
            data = msgpack.unpack(f, raw=False)
        total_records += len(data['records'])
    expected = config['total_steps'] * len(config['dataset_ids'])
    assert total_records == expected


def test_generate_complementary_foreground_replay_manages_independent_step_pairs(
    tmp_path,
    stream_config,
):
    output_dir = tmp_path / 'complementary'
    config = {
        **stream_config,
        'foreground_samples_per_batch': len(stream_config['dataset_ids']) // 2,
        'samples_per_dataset': 1,
        'sampling_rule': COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    }
    generate_replay_stream(
        output_dir=output_dir,
        # An odd shard size exercises a complementary pair spanning shards.
        steps_per_shard=3,
        **config,
    )

    reader = ReplayStreamReader(output_dir)
    reader.validate()
    first_batch_foreground_sets = []
    copied_case_pairs = 0
    for first_step in range(0, stream_config['total_steps'], 2):
        first_batch = group_at(reader, first_step)
        second_batch = group_at(reader, first_step + 1)
        first_foreground = frozenset(
            record.dataset_id for record in first_batch if record.force_foreground
        )
        second_foreground = frozenset(
            record.dataset_id for record in second_batch if record.force_foreground
        )
        assert first_foreground.isdisjoint(second_foreground)
        assert first_foreground | second_foreground == set(
            stream_config['dataset_ids']
        )
        first_batch_foreground_sets.append(first_foreground)
        copied_case_pairs += tuple(record.case_id for record in first_batch) == tuple(
            record.case_id for record in second_batch
        )

    # Each step pair draws a new foreground subset and draws both batches' cases
    # independently instead of copying the first batch into the second.
    assert len(set(first_batch_foreground_sets)) > 1
    assert copied_case_pairs < stream_config['total_steps'] // 2


def test_complementary_foreground_survives_reading_the_stream_in_smaller_batches(
    tmp_path,
    stream_config,
):
    """A batch of two halves each group, so a complementary pair spans four batches."""
    output_dir = tmp_path / 'complementary_batches'
    generate_replay_stream(
        output_dir=output_dir,
        steps_per_shard=7,
        **{
            **stream_config,
            'foreground_samples_per_batch': len(stream_config['dataset_ids']) // 2,
            'samples_per_dataset': 1,
            'sampling_rule': COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        },
    )
    reader = ReplayStreamReader(output_dir)
    group_size = len(stream_config['dataset_ids'])
    batch_size = 2
    batches_per_group = group_size // batch_size

    group_foreground = []
    group_datasets = []
    for group in range(stream_config['total_steps']):
        records = [
            record
            for batch in range(batches_per_group)
            for record in reader.records_at(
                group * group_size + batch * batch_size,
                batch_size,
            )
        ]
        group_datasets.append(sorted(record.dataset_id for record in records))
        group_foreground.append(
            frozenset(
                record.dataset_id for record in records if record.force_foreground
            )
        )

    assert all(
        datasets == sorted(stream_config['dataset_ids'])
        for datasets in group_datasets
    )
    for first, second in zip(
        group_foreground[::2],
        group_foreground[1::2],
        strict=True,
    ):
        assert first.isdisjoint(second)
        assert first | second == set(stream_config['dataset_ids'])


def test_complementary_foreground_stream_can_be_batched_across_group_bounds(
    tmp_path,
):
    dataset_ids = tuple(f'{index:03d}' for index in range(12))
    output_dir = tmp_path / 'cross_group_batches'
    generate_replay_stream(
        seed=12345,
        total_steps=4,
        dataset_ids=dataset_ids,
        training_identifiers={
            dataset_id: (f'case_{dataset_id}',)
            for dataset_id in dataset_ids
        },
        foreground_samples_per_batch=6,
        samples_per_dataset=1,
        sampling_rule=COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        output_dir=output_dir,
        steps_per_shard=3,
    )
    reader = ReplayStreamReader(output_dir)

    # GB8 does not align with a 12-record group: the middle batches cross group boundaries.
    flat_records = [
        record
        for optimizer_step in range(reader.total_records // 8)
        for record in reader.records_at(optimizer_step * 8, 8)
    ]
    assert len(flat_records) == 48
    groups = [flat_records[offset:offset + 12] for offset in range(0, 48, 12)]
    assert all(
        sorted(record.dataset_id for record in group) == sorted(dataset_ids)
        for group in groups
    )
    foreground_sets = [
        {record.dataset_id for record in group if record.force_foreground}
        for group in groups
    ]
    for first, second in zip(
        foreground_sets[::2],
        foreground_sets[1::2],
        strict=True,
    ):
        assert first.isdisjoint(second)
        assert first | second == set(dataset_ids)


def test_generate_rejects_non_empty_output_dir(tmp_path, stream_config):
    """Generation fails if output directory already has shards."""
    output_dir = tmp_path / 'existing'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=100, **stream_config)
    with pytest.raises(FileExistsError, match='already contains'):
        generate_replay_stream(output_dir=output_dir, steps_per_shard=100, **stream_config)


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


def test_reader_returns_correct_batch_size(generated_stream):
    output_dir, config = generated_stream
    reader = ReplayStreamReader(output_dir)
    batch = group_at(reader, 0)
    assert len(batch) == len(config['dataset_ids'])
    assert all(isinstance(r, ReplayRecord) for r in batch)


def test_reader_slices_the_flat_stream_independently_of_group_size(generated_stream):
    output_dir, config = generated_stream
    reader = ReplayStreamReader(output_dir)
    group_size = len(config['dataset_ids'])
    assert reader.total_records == config['total_steps'] * group_size

    flat = [
        record
        for step in range(3)
        for record in group_at(reader, step)
    ]
    # Batches of two tile the flat stream, including across a group boundary.
    assert [
        record
        for start in range(0, 3 * group_size, 2)
        for record in reader.records_at(start, 2)
    ] == flat
    with pytest.raises(IndexError, match='out of range'):
        reader.records_at(reader.total_records - 1, 2)


def test_reader_all_records_valid(generated_stream):
    """Every record names a valid training case from its dataset."""
    output_dir, config = generated_stream
    reader = ReplayStreamReader(output_dir)
    training_ids = config['training_identifiers']
    for step in range(config['total_steps']):
        for record in group_at(reader, step):
            assert record.dataset_id in training_ids, f'unknown dataset {record.dataset_id}'
            assert record.case_id in training_ids[record.dataset_id], (
                f'case {record.case_id} not in dataset {record.dataset_id}'
            )


def test_reader_validates_fingerprint(generated_stream):
    output_dir, _ = generated_stream
    reader = ReplayStreamReader(output_dir)
    ReplayStreamReader(output_dir, expected_fingerprint=reader.fingerprint)
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        ReplayStreamReader(output_dir, expected_fingerprint='wrong' * 4)


def test_reader_validates_exact_stream_fingerprint(tmp_path, stream_config):
    output_dir = tmp_path / 'exact_stream_fingerprint'
    meta = generate_sqrt_replay_stream(
        seed=stream_config['seed'],
        total_records=100,
        dataset_ids=stream_config['dataset_ids'],
        training_identifiers=stream_config['training_identifiers'],
        output_dir=output_dir,
    )

    ReplayStreamReader(
        output_dir,
        expected_stream_fingerprint=meta['stream_fingerprint'],
    )
    with pytest.raises(ValueError, match='stream fingerprint mismatch'):
        ReplayStreamReader(
            output_dir,
            expected_stream_fingerprint='wrong' * 16,
        )


def test_legacy_reader_rejects_expected_stream_fingerprint(generated_stream):
    output_dir, _ = generated_stream
    with pytest.raises(ValueError, match='does not provide stream_fingerprint'):
        ReplayStreamReader(
            output_dir,
            expected_stream_fingerprint='wrong' * 16,
        )


def test_reader_missing_shard_fails(tmp_path, stream_config):
    output_dir = tmp_path / 'incomplete'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=50, **stream_config)
    (output_dir / 'shard_00001.msgpack').unlink()
    with pytest.raises(FileNotFoundError, match='missing shard'):
        ReplayStreamReader(output_dir)


def test_reader_out_of_range_fails(generated_stream):
    output_dir, config = generated_stream
    reader = ReplayStreamReader(output_dir)
    with pytest.raises(IndexError):
        group_at(reader, config['total_steps'])
    with pytest.raises(IndexError):
        group_at(reader, -1)


def test_reader_sequential_consistency(generated_stream):
    """Reading steps 0..N sequentially matches random access."""
    output_dir, _ = generated_stream
    reader = ReplayStreamReader(output_dir)
    batch_10 = group_at(reader, 10)
    batch_50 = group_at(reader, 50)
    batch_99 = group_at(reader, 99)
    assert group_at(reader, 10) == batch_10
    assert group_at(reader, 50) == batch_50
    assert group_at(reader, 99) == batch_99


def test_reader_rejects_corrupted_shard(tmp_path, stream_config):
    """Reader rejects a shard with deleted records (wrong record count)."""
    output_dir = tmp_path / 'corrupt'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=50, **stream_config)

    # Corrupt shard 0: remove one record
    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    data['records'] = data['records'][:-1]  # drop last record
    with open(shard_path, 'wb') as f:
        msgpack.pack(data, f)

    reader = ReplayStreamReader(output_dir)
    with pytest.raises(ValueError, match='records.*expected'):
        group_at(reader, 0)


def test_reader_rejects_inflated_shard_steps(tmp_path, stream_config):
    """Reader rejects a shard that self-declares more steps than metadata dictates."""
    output_dir = tmp_path / 'inflated'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=50, **stream_config)

    # Inflate shard 0: change steps to 60 and add matching extra records
    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    extra_records = [
        {'d': '001', 'c': 'case_a', 'f': False}
    ] * (10 * len(stream_config['dataset_ids']))
    data['steps'] = 60
    data['records'] = data['records'] + extra_records
    with open(shard_path, 'wb') as f:
        msgpack.pack(data, f)

    reader = ReplayStreamReader(output_dir)
    with pytest.raises(ValueError, match='declares.*steps.*expected'):
        group_at(reader, 0)


def test_reader_rejects_extra_shard_files(tmp_path, stream_config):
    """Reader rejects a stream directory with more shard files than metadata declares."""
    output_dir = tmp_path / 'extra'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=100, **stream_config)

    # Create an extra shard file
    extra_path = output_dir / 'shard_00001.msgpack'
    extra_path.write_bytes(b'\x80')

    with pytest.raises(ValueError, match='unexpected extra shards'):
        ReplayStreamReader(output_dir)


def test_validate_eager_preflight(tmp_path, stream_config):
    """validate() catches shard corruption before any stream read."""
    output_dir = tmp_path / 'preflight'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=30, **stream_config)

    # Corrupt the last shard (shard 3, which holds steps 90-99)
    shard_path = output_dir / 'shard_00003.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    data['records'] = data['records'][:-1]
    with open(shard_path, 'wb') as f:
        msgpack.pack(data, f)

    reader = ReplayStreamReader(output_dir)
    # Lazy: reading group 0 succeeds since shard 0 is fine
    assert group_at(reader, 0) is not None
    # Eager: validate() catches the corruption in shard 3
    with pytest.raises(ValueError):
        reader.validate()


def test_validate_checks_shard_hashes(tmp_path, stream_config):
    """validate() detects content changes via SHA-256 mismatch."""
    output_dir = tmp_path / 'hash_check'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=100, **stream_config)

    # Flip one byte in the shard without changing structure
    shard_path = output_dir / 'shard_00000.msgpack'
    content = bytearray(shard_path.read_bytes())
    content[-1] ^= 0xFF
    shard_path.write_bytes(bytes(content))

    reader = ReplayStreamReader(output_dir)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        reader.validate()


def test_validate_checks_v6_ordered_stream_fingerprint(tmp_path, stream_config):
    output_dir = tmp_path / 'ordered_hash_check'
    generate_sqrt_replay_stream(
        seed=stream_config['seed'],
        total_records=100,
        dataset_ids=stream_config['dataset_ids'],
        training_identifiers=stream_config['training_identifiers'],
        output_dir=output_dir,
        records_per_shard=100,
    )

    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    data['records'][0]['f'] = not data['records'][0]['f']
    with open(shard_path, 'wb') as f:
        msgpack.pack(data, f)

    meta_path = output_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['shard_hashes'][shard_path.name] = hashlib.sha256(
        shard_path.read_bytes()
    ).hexdigest()
    meta_path.write_text(yaml.dump(meta, sort_keys=False))

    reader = ReplayStreamReader(output_dir)
    with pytest.raises(ValueError, match='ordered stream SHA-256 mismatch'):
        reader.validate()


def test_validate_checks_v6_statistics(tmp_path, stream_config):
    output_dir = tmp_path / 'statistics_check'
    generate_sqrt_replay_stream(
        seed=stream_config['seed'],
        total_records=100,
        dataset_ids=stream_config['dataset_ids'],
        training_identifiers=stream_config['training_identifiers'],
        output_dir=output_dir,
        records_per_shard=100,
    )

    meta_path = output_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['statistics']['full']['foreground_count'] += 1
    meta_path.write_text(yaml.dump(meta, sort_keys=False))

    with pytest.raises(ValueError, match='statistics do not match'):
        ReplayStreamReader(output_dir).validate()


def test_reader_rejects_invalid_schema(tmp_path, stream_config):
    """Reader rejects records with unexpected keys."""
    output_dir = tmp_path / 'bad_schema'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=100, **stream_config)

    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    # Add an extra key to the first record
    data['records'][0]['force_fg'] = True
    with open(shard_path, 'wb') as f:
        msgpack.pack(data, f)

    reader = ReplayStreamReader(output_dir)
    with pytest.raises(ValueError, match='unexpected keys'):
        group_at(reader, 0)


def test_reader_validates_case_membership(tmp_path, stream_config):
    """Reader with valid_identifiers rejects unknown cases."""
    output_dir = tmp_path / 'bad_case'
    generate_replay_stream(output_dir=output_dir, steps_per_shard=100, **stream_config)

    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    # Replace first case with an invalid one
    data['records'][0] = {'d': '001', 'c': 'INVALID_CASE', 'f': False}
    with open(shard_path, 'wb') as f:
        msgpack.pack(data, f)

    reader = ReplayStreamReader(
        output_dir, valid_identifiers=stream_config['training_identifiers'],
    )
    with pytest.raises(ValueError, match='not in.*training set'):
        group_at(reader, 0)


def test_global_batch_contains_each_dataset_exactly_once(tmp_path, stream_config):
    output_dir = tmp_path / 'global'
    generate_replay_stream(
        output_dir=output_dir,
        steps_per_shard=100,
        **stream_config,
    )
    reader = ReplayStreamReader(output_dir)

    counts = {dataset_id: 0 for dataset_id in stream_config['dataset_ids']}
    for step in range(stream_config['total_steps']):
        global_dataset_ids = tuple(
            record.dataset_id
            for record in group_at(reader, step)
        )
        assert set(global_dataset_ids) == set(stream_config['dataset_ids'])
        assert len(global_dataset_ids) == len(set(global_dataset_ids))
        for dataset_id in global_dataset_ids:
            counts[dataset_id] += 1
        assert sum(
            record.force_foreground for record in group_at(reader, step)
        ) == stream_config['foreground_samples_per_batch']

    assert counts == {
        dataset_id: stream_config['total_steps']
        for dataset_id in stream_config['dataset_ids']
    }


def test_schema_has_no_extra_fields(generated_stream):
    """Records contain only dataset, case, and foreground-crop decisions."""
    output_dir, _ = generated_stream
    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    for record in data['records']:
        assert set(record.keys()) == {'d', 'c', 'f'}


# ---------------------------------------------------------------------------
# ReplayDataLoader tests
# ---------------------------------------------------------------------------


def test_replay_data_loader_returns_prescribed_case():
    """get_indices() returns the prescribed case."""
    loader = object.__new__(ReplayDataLoader)
    loader.indices = ['case_a', 'case_b', 'case_c']
    loader._prescribed_case = None

    loader.set_next_case('case_b')
    result = loader.get_indices()
    assert result == ['case_b']
    assert loader._prescribed_case is None


def test_replay_data_loader_rejects_unknown_case():
    loader = object.__new__(ReplayDataLoader)
    loader.indices = ['case_a', 'case_b']
    loader._prescribed_case = None

    with pytest.raises(ValueError, match='not in loader identifiers'):
        loader.set_next_case('case_z')


def test_replay_data_loader_requires_set_next_case():
    loader = object.__new__(ReplayDataLoader)
    loader.indices = ['case_a']
    loader._prescribed_case = None

    with pytest.raises(RuntimeError, match='set_next_case'):
        loader.get_indices()


def test_replay_data_loader_activates_the_prescribed_case_geometry():
    loader = object.__new__(ReplayDataLoader)
    loader.indices = ['thin', 'isotropic']
    loader._prescribed_case = None
    loader._case_loader_configurations = {
        'thin': CaseLoaderConfiguration(
            patch_size=(40, 225, 225),
            final_patch_size=(40, 192, 192),
            need_to_pad=(0, 33, 33),
            transforms='thin-transform',
            continuous_da=2.321928,
        ),
        'isotropic': CaseLoaderConfiguration(
            patch_size=(308, 308, 308),
            final_patch_size=(192, 192, 192),
            need_to_pad=(116, 116, 116),
            transforms='isotropic-transform',
            continuous_da=0.0,
        ),
    }

    loader.set_next_case('thin')

    assert loader.get_indices() == ['thin']
    assert tuple(loader.patch_size) == (40, 225, 225)
    assert loader.final_patch_size == (40, 192, 192)
    assert tuple(loader.need_to_pad) == (0, 33, 33)
    assert loader.transforms == 'thin-transform'
    assert loader._active_continuous_da == pytest.approx(2.321928)
