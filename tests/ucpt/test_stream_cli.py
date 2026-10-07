"""Stream command dispatch and filtering completion checks."""

import argparse
import json

import pytest
import yaml

from pumit.ucpt.stream import __main__ as cli


@pytest.fixture
def build_args(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'meta.yaml').write_text(yaml.safe_dump({
        'composition': {'normalization': {'contract': 'fixed-source-v1'}},
    }))
    config = tmp_path / 'config.yaml'
    config.write_text('size_xy_choices: [128]\n')
    cost_model = tmp_path / 'cost-model.json'
    cost_model.write_text('{"coefficient": 1}\n')
    monkeypatch.setattr(cli, 'distributed_context', lambda: (0, 1))
    return argparse.Namespace(
        config=config,
        shards=2,
        shard_offset=0,
        output_stream=tmp_path / 'output',
        seed=42,
        batches_per_shard=4,
        budget_ms=600.0,
        label_budget_fraction=0.75,
        cost_model=cost_model,
        use_default_cost_model=False,
        source_stream=source,
        shard_workers=8,
        sample_workers=2,
        sample_prefetch_factor=8,
        force=False,
    )


def _mock_stages(monkeypatch):
    events = []
    monkeypatch.setattr(cli, 'build_stream', lambda **kwargs: events.append(('build', kwargs)))
    monkeypatch.setattr(cli, 'finalize_stream', lambda *args: events.append(('finalize', args)))
    monkeypatch.setattr(cli, 'cmd_link_latents', lambda args: events.append(('link', args)))
    monkeypatch.setattr(cli, 'cmd_verify', lambda args: events.append(('verify', args)))
    return events


def test_plain_build_forwards_arguments(build_args, monkeypatch):
    events = _mock_stages(monkeypatch)
    cli._cmd_build(build_args)
    assert [event[0] for event in events] == ['build']
    assert events[0][1]['config_path'] == build_args.config
    assert events[0][1]['output_dir'] == build_args.output_stream
    assert events[0][1]['source_stream'] == build_args.source_stream


def test_filter_unlabeled_cli_passes_source_and_dataset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'filter_stream', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr('sys.argv', [
        'stream', 'filter-unlabeled',
        '--source-stream', str(tmp_path / 'source'),
        '--output-stream', str(tmp_path / 'filtered'),
        '--exclude-dataset', 'ISLES22', '--shards', '1', '--workers', '2',
    ])

    cli.main()

    assert calls == [{
        'source_stream': tmp_path / 'source', 'output_dir': tmp_path / 'filtered',
        'exclude_datasets': ['ISLES22'], 'shards': 1, 'workers': 2,
    }]


@pytest.fixture
def filter_args(build_args):
    source_meta_path = build_args.source_stream / 'meta.yaml'
    source_meta = yaml.safe_load(source_meta_path.read_text())
    source_meta.update({
        'n_shards': 2,
        'cost_model_source': str(build_args.cost_model),
    })
    source_meta_path.write_text(yaml.safe_dump(source_meta))
    return argparse.Namespace(
        source_stream=build_args.source_stream,
        output_stream=build_args.output_stream,
        exclude_datasets=['ISLES22'],
        shards=None,
        workers=32,
        complete=True,
    )


def _finalized_filtered_stream(args, *, ready=False):
    output = args.output_stream
    output.mkdir()
    source_meta = yaml.safe_load((args.source_stream / 'meta.yaml').read_text())
    shard_count = source_meta['n_shards'] if args.shards is None else args.shards
    plan = {
        'fingerprint': 'filter-build-fingerprint',
        'source_stream': str(args.source_stream.resolve()),
        'source_meta_sha256': cli.sha256_file(args.source_stream / 'meta.yaml'),
        'unlabeled_filter': {
            'contract': cli.UNLABELED_FILTER_CONTRACT,
            'exclude_datasets': args.exclude_datasets,
        },
    }
    (output / cli.BUILD_PLAN_NAME).write_text(yaml.safe_dump(plan))
    (output / 'manifest.jsonl').write_text('{}\n')
    meta = {
        **plan,
        'build_fingerprint': plan['fingerprint'],
        'fingerprint': 'filtered-fingerprint',
        'n_shards': shard_count,
        'stream_complete': True,
        'manifest_sha256': cli.sha256_file(output / 'manifest.jsonl'),
    }
    (output / 'meta.yaml').write_text(yaml.safe_dump(meta))
    if ready:
        (output / 'READY.json').write_text(json.dumps({
            'fingerprint': meta['fingerprint'],
            'verified_shards': shard_count,
            'manifest_sha256': meta['manifest_sha256'],
        }))


def _mock_filter_stages(monkeypatch):
    events = _mock_stages(monkeypatch)
    monkeypatch.setattr(cli, 'filter_stream', lambda **kwargs: events.append(('filter', kwargs)))
    return events


@pytest.mark.parametrize('shards', [None, 1])
def test_complete_filter_runs_cpu_stages_in_order(filter_args, monkeypatch, shards):
    events = _mock_filter_stages(monkeypatch)
    filter_args.shards = shards
    cli._cmd_filter_unlabeled(filter_args)
    assert [event[0] for event in events] == ['filter', 'finalize', 'link', 'verify']
    assert events[1][1] == (filter_args.output_stream, 2 if shards is None else 1, 32)
    assert events[2][1].source_latent_dir == filter_args.source_stream / 'latents'
    assert events[2][1].workers == events[3][1].workers == 32
    assert events[3][1].cost_model.is_file()


def test_filter_direct_call_without_complete_stays_metadata_only(filter_args, monkeypatch):
    events = _mock_filter_stages(monkeypatch)
    del filter_args.complete
    cli._cmd_filter_unlabeled(filter_args)
    assert [event[0] for event in events] == ['filter']


@pytest.mark.parametrize('ready', [False, True])
def test_complete_filter_resumes_finalized_stream(filter_args, monkeypatch, ready):
    events = _mock_filter_stages(monkeypatch)
    _finalized_filtered_stream(filter_args, ready=ready)
    cli._cmd_filter_unlabeled(filter_args)
    assert [event[0] for event in events] == ([] if ready else ['link', 'verify'])


@pytest.mark.parametrize('changed', ['exclude', 'source', 'source_meta', 'shards', 'identity', 'ready'])
def test_complete_filter_rejects_changed_resume(filter_args, monkeypatch, changed):
    events = _mock_filter_stages(monkeypatch)
    _finalized_filtered_stream(filter_args, ready=True)
    if changed == 'exclude':
        filter_args.exclude_datasets = ['AMOS22']
    elif changed == 'source':
        replacement = filter_args.source_stream.parent / 'replacement-source'
        replacement.mkdir()
        (replacement / 'meta.yaml').write_text((filter_args.source_stream / 'meta.yaml').read_text())
        filter_args.source_stream = replacement
    elif changed == 'source_meta':
        path = filter_args.source_stream / 'meta.yaml'
        path.write_text(path.read_text() + 'changed: true\n')
    elif changed == 'shards':
        filter_args.shards = 1
    elif changed == 'identity':
        path = filter_args.output_stream / 'meta.yaml'
        meta = yaml.safe_load(path.read_text())
        meta['build_fingerprint'] = 'different-build'
        path.write_text(yaml.safe_dump(meta))
    else:
        path = filter_args.output_stream / 'READY.json'
        ready = json.loads(path.read_text())
        ready['verified_shards'] = 1
        path.write_text(json.dumps(ready))
    with pytest.raises(ValueError):
        cli._cmd_filter_unlabeled(filter_args)
    assert events == []


def test_complete_filter_rejects_multi_node_before_writing(filter_args, monkeypatch):
    events = _mock_filter_stages(monkeypatch)
    monkeypatch.setattr(cli, 'distributed_context', lambda: (0, 2))
    with pytest.raises(ValueError, match='requires a single node'):
        cli._cmd_filter_unlabeled(filter_args)
    assert events == []
    assert not filter_args.output_stream.exists()


def test_complete_filter_rejects_missing_cost_model_before_writing(filter_args, monkeypatch):
    events = _mock_filter_stages(monkeypatch)
    path = filter_args.source_stream / 'meta.yaml'
    meta = yaml.safe_load(path.read_text())
    meta['cost_model_source'] = 'embedded-default'
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(FileNotFoundError, match='source cost model is not a file'):
        cli._cmd_filter_unlabeled(filter_args)
    assert events == []
    assert not filter_args.output_stream.exists()


def test_filter_progress_repeats_flushed_artifact_counts(tmp_path, monkeypatch):
    from threading import Event

    repeated = Event()
    messages = []

    def capture(message, *, flush):
        assert flush is True
        messages.append(message)
        if 'reports=1 latents=1' in message:
            repeated.set()

    monkeypatch.setattr(cli, '_FILTER_HEARTBEAT_SECONDS', 0.01)
    monkeypatch.setattr(cli, 'print', capture, raising=False)
    with cli._filter_phase_progress(tmp_path, 'latent-reuse'):
        (tmp_path / 'reports').mkdir()
        (tmp_path / 'reports' / 'shard_00000.json').write_text('{}\n')
        (tmp_path / 'latents').mkdir()
        (tmp_path / 'latents' / 'shard_00000.safetensors').touch()
        (tmp_path / 'latents' / 'shard_00001.safetensors.tmp').touch()
        assert repeated.wait(timeout=2)
    assert 'reports=0 latents=0' in messages[0]
    assert len(messages) >= 2
    assert all('phase=latent-reuse elapsed=' in message for message in messages)


def test_filter_complete_reports_phase_transitions(filter_args, monkeypatch, capsys):
    _mock_filter_stages(monkeypatch)
    cli._cmd_filter_unlabeled(filter_args)
    messages = capsys.readouterr().out.splitlines()
    phases = [message.split('phase=', 1)[1].split()[0] for message in messages if 'phase=' in message]
    assert phases == ['prepare', 'filter', 'finalize', 'latent-reuse', 'verify']
    assert messages[-1].startswith('[ucpt] filter-unlabeled complete:')


@pytest.mark.parametrize('failed', [False, True])
def test_filter_progress_stops_thread_and_preserves_exception(tmp_path, monkeypatch, failed):
    threads = []
    thread_type = cli.Thread

    def create_thread(**kwargs):
        thread = thread_type(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(cli, 'Thread', create_thread)
    error = RuntimeError('original phase error')
    try:
        with cli._filter_phase_progress(tmp_path, 'verify'):
            assert threads[0].is_alive()
            if failed:
                raise error
    except RuntimeError as caught:
        assert caught is error
    else:
        assert not failed
    assert len(threads) == 1
    assert not threads[0].is_alive()
