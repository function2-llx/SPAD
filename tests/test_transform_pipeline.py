import numpy as np
import pytest

from pumit.transforms.pipeline import TransformPipeline, RandomizableTransform


class FakeRandom:
    """Random transform that records rng draws."""
    def sample_params(self, state, rng):
        val = float(rng.random())
        return {'enabled': True, 'value': val}

    def __call__(self, data, *, enabled, value):
        data['result'] = data.get('result', 0) + value
        return data


class FakeDeterministic:
    """Deterministic transform as a plain function-like class (no sample_params)."""
    def __call__(self, data, **kwargs):
        data['result'] = data.get('result', 0) * 2
        return data


class FakeDrop:
    """Transform that always drops."""
    def sample_params(self, state, rng):
        return None

    def __call__(self, data, **kwargs):
        return data


def _plain_function(data: dict, **kwargs) -> dict:
    """Plain function usable as a pipeline step."""
    data['func_called'] = True
    return data


def test_sample_params_returns_list():
    pipeline = TransformPipeline([FakeRandom(), FakeDeterministic()])
    rng = np.random.default_rng(42)
    params = pipeline.sample_params({}, rng)
    assert isinstance(params, list)
    assert len(params) == 2
    assert 'value' in params[0]
    assert params[1] == {}


def test_replay_applies_transforms_in_order():
    pipeline = TransformPipeline([FakeRandom(), FakeDeterministic()])
    rng = np.random.default_rng(42)
    params = pipeline.sample_params({}, rng)
    data = {'result': 1}
    data = pipeline.replay(data, params)
    expected = (1 + params[0]['value']) * 2
    assert data['result'] == pytest.approx(expected)


def test_drop_propagates_none():
    pipeline = TransformPipeline([FakeRandom(), FakeDrop(), FakeRandom()])
    rng = np.random.default_rng(42)
    params = pipeline.sample_params({}, rng)
    assert params is None


def test_replay_asserts_length_mismatch():
    pipeline = TransformPipeline([FakeRandom(), FakeDeterministic()])
    with pytest.raises(AssertionError):
        pipeline.replay({}, [{'enabled': True, 'value': 0.5}])


def test_deterministic_only_pipeline():
    pipeline = TransformPipeline([FakeDeterministic(), FakeDeterministic()])
    rng = np.random.default_rng(0)
    params = pipeline.sample_params({}, rng)
    assert params == [{}, {}]
    data = pipeline.replay({'result': 3}, params)
    assert data['result'] == 12  # 3 * 2 * 2


def test_plain_function_as_transform():
    pipeline = TransformPipeline([_plain_function, FakeDeterministic()])
    rng = np.random.default_rng(0)
    params = pipeline.sample_params({}, rng)
    assert params == [{}, {}]
    data = pipeline.replay({'result': 5}, params)
    assert data['func_called'] is True
    assert data['result'] == 10


def test_isinstance_check():
    assert isinstance(FakeRandom(), RandomizableTransform)
    assert not isinstance(FakeDeterministic(), RandomizableTransform)
    assert not isinstance(_plain_function, RandomizableTransform)
