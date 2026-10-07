import math
import torch
from pumit.model.rope import SpatialRotaryEmbedding3D


def test_freq_matrix_matches_compute_angles():
    """freq_matrix matmul must produce identical angles to _compute_angles."""
    rope = SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)

    for shape in [(4, 24, 24), (1, 24, 24), (8, 12, 12), (16, 24, 24)]:
        D, H, W = shape
        old_result = rope._compute_angles(shape, torch.device('cpu'))

        coords_d = (2 * (torch.arange(D, dtype=torch.float32) + 0.5) / D) - 1
        coords_h = (2 * (torch.arange(H, dtype=torch.float32) + 0.5) / H) - 1
        coords_w = (2 * (torch.arange(W, dtype=torch.float32) + 0.5) / W) - 1

        grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')
        coords = torch.stack([grid_d.flatten(), grid_h.flatten(), grid_w.flatten()], dim=1)

        new_angles = coords @ rope.freq_matrix
        new_cos = new_angles.cos()
        new_sin = new_angles.sin()
        assert torch.allclose(old_result[:, 0, :], new_cos, atol=1e-5), \
            f"cos mismatch for shape {shape}: max diff {(old_result[:, 0, :] - new_cos).abs().max()}"
        assert torch.allclose(old_result[:, 1, :], new_sin, atol=1e-5), \
            f"sin mismatch for shape {shape}: max diff {(old_result[:, 1, :] - new_sin).abs().max()}"


def test_freq_matrix_prefix_gives_identity_rope():
    """Coords (0,0,0) must produce cos=1, sin=0 (identity RoPE)."""
    rope = SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)
    prefix_coords = torch.zeros(5, 3)
    angles = prefix_coords @ rope.freq_matrix
    cos = angles.cos()
    sin = angles.sin()
    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin))


def test_freq_matrix_with_rescale():
    """Rescaling coords by r should match _compute_angles_augmented semantics."""
    rope = SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)
    shape = (4, 12, 12)
    D, H, W = shape
    r = 1.5

    coords_d = ((2 * (torch.arange(D, dtype=torch.float32) + 0.5) / D) - 1) * r
    coords_h = ((2 * (torch.arange(H, dtype=torch.float32) + 0.5) / H) - 1) * r
    coords_w = ((2 * (torch.arange(W, dtype=torch.float32) + 0.5) / W) - 1) * r

    grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')
    coords = torch.stack([grid_d.flatten(), grid_h.flatten(), grid_w.flatten()], dim=1)

    new_angles = coords @ rope.freq_matrix
    new_cos = new_angles.cos()
    new_sin = new_angles.sin()

    inv_freq = rope.inv_freq
    angles_2d = (2 * math.pi * torch.stack(
        torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1
    ).flatten(0, 1)[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
    angles_d_manual = (2 * math.pi * coords_d[:, None] * rope.inv_freq_depth[None, :]).tile(4)
    angles_manual = (angles_d_manual[:, None, :] + angles_2d[None, :, :]).reshape(D * H * W, 64)

    assert torch.allclose(new_angles, angles_manual, atol=1e-5)
