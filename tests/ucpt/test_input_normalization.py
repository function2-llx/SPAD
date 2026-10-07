from copy import deepcopy

import numpy as np
import orjson
import pytest
import torch
import yaml

from pumit.transforms.patchify import patchify
from pumit.ucpt.input import InputNormalizer, normalize_input
from pumit.ucpt.transforms import build_ucpt_pipeline


def _input_tree(tmp_path, records, *, dataset='Mixed'):
    data_root = tmp_path / 'data' / 'preprocess'
    physical_root = tmp_path / 'physical-preprocess'
    physical_root.mkdir()
    data_root.parent.mkdir()
    data_root.symlink_to(physical_root, target_is_directory=True)
    metadata_path = data_root.parent / 'fingerprint' / dataset / 'samples.json'
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_bytes(orjson.dumps(records))
    image_dir = data_root / dataset / 'images'
    image_dir.mkdir(parents=True)
    return data_root, image_dir, metadata_path


def _pipeline(normalizer=None):
    return build_ucpt_pipeline(
        size_xy_choices=[16], size_xy_choices_2d=[16], max_depth_per_da={0: 16},
        input_normalizer=normalizer,
    )


def _params(image):
    shape = list(image.shape[1:])
    return [
        {
            'da_enc': None,
            'affine': np.eye(4).ravel().tolist(),
            'load_slice_start': [0, 0, 0],
            'load_slice_stop': shape,
            'crop_size': shape,
            'n_patches': 1,
            'spacing_label': [1.0, 1.0, 1.0],
        },
        {'enabled': False},
        {'enabled': False},
        {'enabled': False},
        {},
    ]


def test_cache_dtype_routes_ambiguous_modalities_and_caches_compact_index(tmp_path):
    data_root, image_dir, metadata_path = _input_tree(
        tmp_path,
        {
            'converted_display': {'modality': 'US', 'image_dtype': {'original': 'int16', 'cache': 'uint8'}},
            'medical': {'modality': 'US', 'image_dtype': {'original': 'uint8', 'cache': 'float32'}},
            'display': {'modality': 'US', 'image_dtype': 'uint8'},
            'fundus': {'modality': 'fundus', 'image_dtype': 'uint8'},
            'rgb_prefix': {'modality': 'RGB/fundus', 'image_dtype': 'float32'},
            'gray_prefix': {'modality': 'gray/X-ray', 'image_dtype': 'uint16'},
            'histology': {'modality': 'histopathology', 'image_dtype': {'original': 'float64', 'cache': 'uint8'}},
            'float_histology': {'modality': 'histopathology', 'image_dtype': 'float64'},
        },
    )
    normalizer = InputNormalizer(data_root)
    assert normalizer.scheme(image_dir / 'medical.npy') == 'zscore'
    metadata_path.unlink()
    assert normalizer.scheme_for('Mixed', 'converted_display') == 'rgb'
    assert normalizer.scheme_for('Mixed', 'display') == 'rgb'
    assert normalizer.scheme(image_dir / 'fundus.npy') == 'rgb'
    assert normalizer.scheme_for('Mixed', 'rgb_prefix') == 'zscore'
    assert normalizer.scheme_for('Mixed', 'gray_prefix') == 'zscore'
    assert normalizer.scheme_for('Mixed', 'histology') == 'rgb'
    assert normalizer.scheme_for('Mixed', 'float_histology') == 'zscore'
    assert normalizer._dataset_schemes('Mixed') == {
        'converted_display': 'rgb', 'medical': 'zscore', 'display': 'rgb', 'fundus': 'rgb',
        'rgb_prefix': 'zscore', 'gray_prefix': 'zscore', 'histology': 'rgb', 'float_histology': 'zscore',
    }


def test_medical_sampling_uses_weak_recipe_without_changing_spatial_parameters(tmp_path):
    data_root, image_dir, _ = _input_tree(
        tmp_path, {'case': {'modality': 'US', 'image_dtype': 'float32'}},
    )
    state = {
        'img': str(image_dir / 'case.npy'),
        'shape': [1, 16, 16],
        'spacing': [np.nan, 1.0, 1.0],
        'num_channels': 1,
    }
    pipeline = _pipeline(InputNormalizer(data_root))
    legacy = _pipeline()
    fresh = pipeline.sample_params(state, np.random.default_rng(42))
    old = legacy.sample_params(state, np.random.default_rng(42))
    assert len(fresh) == len(old) == 5
    np.testing.assert_array_equal(fresh[0]['affine'], old[0]['affine'])
    assert fresh[0]['load_slice_start'] == old[0]['load_slice_start']
    assert fresh[0]['crop_size'] == old[0]['crop_size']

    scale, contrast, gamma = pipeline.transforms[1:4]
    assert scale.prob == 0.1
    assert gamma.prob == 0.15
    rng = np.random.default_rng(21)
    scale_samples = [scale.sample_params(state, rng) for _ in range(200)]
    gamma_samples = [gamma.sample_params(state, rng) for _ in range(200)]
    assert any(sample['enabled'] for sample in scale_samples)
    assert any(sample['enabled'] for sample in gamma_samples)
    assert contrast.sample_params(state, rng) == {'enabled': False}
    for sample in scale_samples:
        if sample['enabled']:
            assert all(-0.2 <= factor <= 0.2 for factor in sample['factors'])
    for sample in gamma_samples:
        if sample['enabled']:
            assert not sample['invert']
            assert all(0.8 <= value <= 1.2 for value in sample['gammas'])


def test_legacy_image_symlink_uses_canonical_source_metadata(tmp_path):
    key = 'dataset-verse19training_sub-verse001_ct'
    data_root, image_dir, _ = _input_tree(
        tmp_path, {key: {'modality': 'CT', 'image_dtype': 'int16'}},
        dataset='VerSe',
    )
    image = np.linspace(-3, 5, 256, dtype=np.float16).reshape(1, 1, 16, 16)
    canonical = image_dir / f'{key}.npy'
    np.save(canonical, image)
    alias = image_dir / 'sub-verse001_ct.npy'
    alias.symlink_to(canonical.name)
    normalizer = InputNormalizer(data_root)

    assert normalizer.scheme(alias) == 'zscore'
    result = _pipeline(normalizer).replay({'img': str(alias)}, _params(image))['img']
    expected = torch.from_numpy(image).float().repeat(3, 1, 1, 1)
    torch.testing.assert_close(result, expected)


def test_medical_replay_keeps_zscores_and_gamma_retains_scaled_statistics(tmp_path):
    data_root, image_dir, _ = _input_tree(
        tmp_path, {'case': {'modality': 'MRI', 'image_dtype': 'float32'}},
    )
    image = np.linspace(-3, 5, 256, dtype=np.float32).reshape(1, 1, 16, 16)
    path = image_dir / 'case.npy'
    np.save(path, image)
    pipeline = _pipeline(InputNormalizer(data_root))
    params = _params(image)
    unchanged = pipeline.replay({'img': str(path)}, params)['img']
    torch.testing.assert_close(unchanged, torch.from_numpy(image).repeat(3, 1, 1, 1))

    params[1] = {'enabled': True, 'factors': [0.2]}
    params[2] = {'enabled': True, 'factor': 0.0}
    params[3] = {'enabled': True, 'gammas': [0.8], 'invert': False}
    frozen = deepcopy(params)
    augmented = pipeline.replay({'img': str(path)}, params)['img']
    scaled = torch.from_numpy(image) * 1.2
    torch.testing.assert_close(augmented[0].mean(), scaled.mean())
    torch.testing.assert_close(augmented[0].std(correction=0), scaled.std(correction=0))
    assert augmented.min() < 0 and augmented.max() > 1
    assert not torch.allclose(augmented[0], scaled[0])
    assert params == frozen


@pytest.mark.parametrize(
    ('channels', 'modality', 'dtype'),
    [
        (1, 'US', 'uint8'),
        (3, 'fundus', 'uint8'),
        (3, 'histopathology', {'original': 'float64', 'cache': 'uint8'}),
    ],
)
def test_display_replay_preserves_pass2_normalization_and_ignores_frozen_intensity(tmp_path, channels, modality, dtype):
    data_root, image_dir, _ = _input_tree(
        tmp_path, {'case': {'modality': modality, 'image_dtype': dtype}},
    )
    source = np.linspace(0, 255, channels * 256, dtype=np.uint8).reshape(channels, 1, 16, 16)
    rgb = np.repeat(source, 3, axis=0) if channels == 1 else source
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
    image = ((rgb.astype(np.float32) / 255 - mean) / std).astype(np.float16)
    path = image_dir / 'case.npy'
    np.save(path, image)
    pipeline = _pipeline(InputNormalizer(data_root))
    for transform in pipeline.transforms[1:4]:
        assert transform.sample_params({'img': str(path)}, np.random.default_rng(0)) == {'enabled': False}
    params = _params(source)
    params[1] = {'enabled': True, 'factors': [0.2]}
    params[2] = {'enabled': True, 'factor': 0.0}
    params[3] = {'enabled': True, 'gammas': [1.5], 'invert': True}
    frozen = deepcopy(params)
    result = pipeline.replay({'img': str(path)}, params)['img']
    torch.testing.assert_close(result, torch.from_numpy(image).float(), rtol=0, atol=0)
    assert result.shape == (3, 1, 16, 16)
    assert result.min() < 0 and result.max() > 1
    if channels == 1:
        assert not torch.equal(result[0], result[1])
    patches, grid = patchify(result, da=4)
    assert patches.shape[0] == params[0]['n_patches'] == 1
    assert grid == (1, 1, 1)
    assert params == frozen


@pytest.mark.parametrize('scheme,channels', [('zscore', 1), ('zscore', 3), ('rgb', 3)])
@pytest.mark.parametrize('batched', [False, True])
def test_runtime_preparation_only_casts_expands_medical_and_makes_contiguous(scheme, channels, batched):
    shape = (2, channels, 1, 3, 4) if batched else (channels, 1, 3, 4)
    image = torch.linspace(-3, 7, int(np.prod(shape)), dtype=torch.float16).reshape(shape).transpose(-1, -2)
    expected = image.float()
    if channels == 1:
        expected = expected.repeat_interleave(3, dim=int(batched))
    actual = normalize_input(image, scheme, batched=batched)
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_runtime_preparation_rejects_invalid_scheme_or_unexpanded_display():
    with pytest.raises(ValueError, match='unknown UCPT input normalization scheme'):
        normalize_input(torch.ones(3, 1, 2, 2), 'unknown')
    with pytest.raises(ValueError, match='expected three channels'):
        normalize_input(torch.ones(1, 1, 2, 2), 'rgb')


def test_default_pipeline_preserves_legacy_input_behavior(tmp_path):
    image = np.full((1, 1, 16, 16), 0.75, dtype=np.float32)
    path = tmp_path / 'image.npy'
    np.save(path, image)
    result = _pipeline().replay({'img': str(path)}, _params(image))['img']
    torch.testing.assert_close(result, torch.full((3, 1, 16, 16), 0.5))


def test_training_builder_explicitly_enables_input_routing(tmp_path, monkeypatch):
    import pumit.ucpt.train.data as training_data
    from pumit.ucpt.train.config import UCPTDataConfig

    stream = tmp_path / 'stream'
    stream.mkdir()
    (stream / 'meta.yaml').write_text(yaml.safe_dump({
        'config': {'size_xy_choices': [16], 'size_xy_choices_2d': [16], 'max_depth_per_da': {0: 16}},
    }))
    monkeypatch.setattr(training_data, 'UCPTReplayDataset', lambda **kwargs: kwargs)
    dataset = training_data.build_replay_dataset(
        UCPTDataConfig(stream_dir=str(stream), data_root=str(tmp_path / 'preprocess')),
        rank=0, world_size=1, start_offset=0, n_prefix=5, augment_threads=1, tcmalloc_release_every=0,
    )
    assert isinstance(dataset['pipeline'].transforms[-1], InputNormalizer)
    assert dataset['pipeline'].transforms[-1].fingerprint_root == tmp_path / 'fingerprint'


@pytest.mark.parametrize('dtype,scheme', [('float32', 'zscore'), ('uint8', 'rgb')])
def test_eval_and_training_preserve_the_same_pass2_inputs(tmp_path, monkeypatch, dtype, scheme):
    import pumit.ucpt.seg.evaluation as evaluation

    data_root, image_dir, _ = _input_tree(
        tmp_path, {'case': {'modality': 'US', 'image_dtype': dtype}},
    )
    image = np.full((1, 1, 16, 16), -0.75, dtype=np.float32)
    if scheme == 'rgb':
        image = np.broadcast_to(
            np.array([-1.5, 0.25, 2.5], dtype=np.float16).reshape(3, 1, 1, 1),
            (3, 1, 16, 16),
        ).copy()
    path = image_dir / 'case.npy'
    np.save(path, image)
    normalizer = InputNormalizer(data_root)
    expected = _pipeline(normalizer).replay({'img': str(path)}, _params(image))['img']
    sample = {
        'dataset': 'Mixed', 'key': 'case', 'img': str(path), 'modality': 'US', 'shape': [1, 16, 16],
        'classes': [{'source': 'source', 'name': 'class'}], 'da': None,
        'sliding_window': {
            'padded_shape': [1, 16, 16], 'roi_size': [1, 16, 16], 'overlap': 0.0,
        },
    }
    panel_path = tmp_path / 'panel.json'
    panel_path.write_bytes(orjson.dumps({'format_version': evaluation.PANEL_FORMAT_VERSION, 'samples': [sample]}))

    class TextResources:
        def __init__(self, path):
            pass

        def get_batch(self, keys):
            return torch.ones(1, 2, 4)

        def get_mask_batch(self, keys):
            return torch.ones(1, 2, dtype=torch.bool)

    monkeypatch.setattr(evaluation, 'TextEmbeddingCache', TextResources)
    monkeypatch.setattr(evaluation, 'SegmentationPromptResolver', TextResources)
    dataset = evaluation.SegEvalDataset(
        panel_path, data_root=data_root, text_cache_path=tmp_path, class_captions_dir=tmp_path,
        input_normalizer=normalizer,
    )
    item = dataset[0]
    assert item['input_scheme'] == scheme
    seen = []

    def artifact(windows, da, embeddings, valid):
        seen.append(windows)
        return windows[:, :1].unsqueeze(1)

    evaluation.sliding_window_logits(
        artifact, item, device=torch.device('cpu'), sw_batch_size=1,
        interpolation_order='interpolate-then-stitch',
    )
    torch.testing.assert_close(seen[0][0], expected)
