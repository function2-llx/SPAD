# tests/ucpt/test_analyze_stream.py
"""analyze_stream verdict tests: exit 1 on any zero-labeled batch, else pass."""
import msgpack
import pytest
import yaml


def _sample(labeled, n_patches=100):
    s = {
        'img': '/fake/x.npy',
        'n_patches': n_patches,
        'da_enc': 0,
        'depth': 16,
        'labeled': labeled,
    }
    if labeled:
        s['seg_cost_queries'] = 1
        s['dataset'] = 'd1'
        s['key'] = 'k1'
        s['modality'] = 'CT'
        s['classes'] = [
            {
                'source': 'src1',
                'name': 'liver',
                'is_positive': True,
                'target_voxels': 16,
            },
        ]
    return s


def _write_stream(tmp_path, batches):
    with open(tmp_path / 'shard_00000.msgpack', 'wb') as f:
        msgpack.pack({'batches': batches, 'total_patches': 1}, f)
    (tmp_path / 'meta.yaml').write_text(yaml.safe_dump(
        {'label_budget_fraction': 0.5, 'budget_ms': 6000.0, 'seed': 42}))


def _run(tmp_path, monkeypatch):
    import scripts.ucpt.analyze_stream as az
    monkeypatch.setattr(
        'sys.argv',
        ['analyze_stream', str(tmp_path), '--use-default-cost-model', '--workers', '1'],
    )
    az.main()


def test_zero_labeled_batch_exits_1(tmp_path, monkeypatch):
    batches = [
        {'step_idx': 0, 'samples': [_sample(True), _sample(False)]},
        {'step_idx': 1, 'samples': [_sample(False), _sample(False)]},
    ]
    _write_stream(tmp_path, batches)
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, monkeypatch)
    assert exc.value.code != 0
    assert 'FAIL' in str(exc.value)
    assert '1 zero-labeled' in str(exc.value)


def test_all_batches_labeled_passes(tmp_path, monkeypatch, capsys):
    batches = [
        {'step_idx': 0, 'samples': [_sample(True), _sample(False)]},
        {'step_idx': 1, 'samples': [_sample(True)]},
    ]
    _write_stream(tmp_path, batches)
    _run(tmp_path, monkeypatch)  # no SystemExit
    out = capsys.readouterr().out
    assert 'PASS' in out
    assert 'realized' in out  # compute-fraction report present
