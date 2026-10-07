import numpy as np
import torch

from pumit.downstream.cls.data import normalize_to_encoder


def test_normalize_2d_grayscale_to_5d_rgb():
    # (N, H, W, 1) uint8 single-channel 2D
    x = np.zeros((2, 224, 224, 1), dtype=np.uint8)
    x[...] = 255
    out = normalize_to_encoder(x, is_3d=False)
    assert out.shape == (2, 3, 1, 224, 224)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out))  # 255/255*2-1 == 1


def test_normalize_2d_rgb_preserved():
    x = np.zeros((2, 224, 224, 3), dtype=np.uint8)  # already RGB
    out = normalize_to_encoder(x, is_3d=False)
    assert out.shape == (2, 3, 1, 224, 224)
    assert torch.allclose(out, -torch.ones_like(out))  # 0/255*2-1 == -1


def test_normalize_2d_grayscale_3d_input():
    # (N, H, W) uint8, no channel axis — the ndim==3 branch (live path for
    # grayscale 2D MedMNIST sets like OCT/Pneumonia/Tissue).
    x = np.zeros((2, 28, 28), dtype=np.uint8)
    x[...] = 255
    out = normalize_to_encoder(x, is_3d=False)
    assert out.shape == (2, 3, 1, 28, 28)
    assert torch.allclose(out, torch.ones_like(out))


def test_normalize_3d_to_5d_rgb():
    x = np.zeros((2, 64, 64, 64), dtype=np.uint8)  # (N, D, H, W)
    out = normalize_to_encoder(x, is_3d=True)
    assert out.shape == (2, 3, 64, 64, 64)
    assert torch.allclose(out, -torch.ones_like(out))
