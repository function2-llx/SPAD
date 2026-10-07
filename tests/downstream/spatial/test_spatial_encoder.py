"""Tests for spatial-evaluation encoder input contracts."""

import pytest
import torch

from pumit.downstream.spatial.encoder import normalize_volumes, slice_encoder_state


def test_normalize_volumes_expands_grayscale_for_pumit():
    volumes = torch.tensor([[[[[0.0, 0.5, 1.0]]]]])

    normalized = normalize_volumes(volumes, 'pumit')

    assert normalized.shape == (1, 3, 1, 1, 3)
    torch.testing.assert_close(normalized[:, 0], torch.tensor([[[[-1.0, 0.0, 1.0]]]]))
    torch.testing.assert_close(normalized[:, 1], normalized[:, 0])
    torch.testing.assert_close(normalized[:, 2], normalized[:, 0])


def test_normalize_volumes_rejects_implicit_normalization():
    with pytest.raises(ValueError, match='unsupported normalization'):
        normalize_volumes(torch.zeros(1, 1, 1, 1, 1), 'auto')


@pytest.mark.parametrize(
    ('checkpoint_format', 'key'),
    [
        ('ucpt', '_orig_mod.teacher_vit.layer.weight'),
        ('legacy-ssl', '_orig_mod.encoder.layer.weight'),
    ],
)
def test_slice_encoder_state_requires_explicit_checkpoint_format(checkpoint_format, key):
    weight = torch.ones(1)

    sliced = slice_encoder_state({key: weight}, checkpoint_format)

    assert sliced == {'layer.weight': weight}
