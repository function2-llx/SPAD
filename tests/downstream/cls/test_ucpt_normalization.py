import numpy as np
import pytest
import torch

from pumit.downstream.cls.backbones.vit_spad import ucpt
from pumit.downstream.cls.data import normalize_to_encoder
from pumit.downstream.cls.registry import BACKBONES


@pytest.mark.parametrize(
    'shape,is_3d',
    [
        ((2, 5, 7), False),
        ((2, 5, 7, 1), False),
        ((2, 5, 7, 3), False),
        ((2, 3, 5, 7), True),
    ],
)
def test_ucpt_imagenet_matches_pretraining_uint8_path(shape, is_3d):
    images = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    backbone = BACKBONES['ucpt-imagenet']
    assert backbone.create_model is ucpt
    actual = backbone.transform_batch(images, is_3d=is_3d)['x']
    expected = []
    for sample in images:
        if is_3d:
            sample = sample[None]
        elif sample.ndim == 2:
            sample = sample[None, None]
        else:
            sample = sample.transpose(2, 0, 1)[:, None]
        if sample.shape[0] == 1:
            sample = np.repeat(sample, 3, axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]
        expected.append((sample.astype(np.float32) / 255.0 - mean) / std)
    torch.testing.assert_close(actual, torch.from_numpy(np.stack(expected)))
    assert actual.is_contiguous()


def test_legacy_ucpt_normalization_unchanged():
    images = np.random.default_rng(0).integers(0, 256, size=(2, 3, 5, 7), dtype=np.uint8)
    backbone = BACKBONES['ucpt']
    assert backbone.create_model is ucpt
    torch.testing.assert_close(
        backbone.transform_batch(images, is_3d=True)['x'],
        normalize_to_encoder(images, is_3d=True),
        rtol=0,
        atol=0,
    )
