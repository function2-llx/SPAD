from torch import Tensor

__all__ = ['patchify']


def patchify(
    img: Tensor,
    da: int,
    patch_size: int = 16,
    max_adapt: int = 4,
) -> tuple[Tensor, tuple[int, int, int]]:
    """Chunk a 3D volume into uniform (C, patch_size, patch_size, patch_size) patches.

    Depth slices thinner than patch_size are repeated via repeat_interleave to fill
    the full patch depth. This is mathematically equivalent to applying SPADPatchEmbed
    with the given da directly on the original image.

    Args:
        img: (C, D, H, W) volume tensor.
        da: depth adaptation level (number of right-shifts on kernel depth).
        patch_size: spatial patch size (isotropic for H, W; depth before repetition).
        max_adapt: maximum adaptation level (clamps da).

    Returns:
        patches: (n_patches, C, patch_size, patch_size, patch_size) tensor.
        spatial_shape: (D_patches, H_patches, W_patches) grid dimensions.
    """
    C, D, H, W = img.shape
    patch_d = patch_size >> min(da, max_adapt)
    D_patches = D // patch_d
    H_patches = H // patch_size
    W_patches = W // patch_size
    n_patches = D_patches * H_patches * W_patches

    patches = img.reshape(C, D_patches, patch_d, H_patches, patch_size, W_patches, patch_size)
    patches = patches.permute(1, 3, 5, 0, 2, 4, 6).contiguous()
    patches = patches.reshape(n_patches, C, patch_d, patch_size, patch_size)

    if patch_d < patch_size:
        patches = patches.repeat_interleave(patch_size // patch_d, dim=2)

    return patches, (D_patches, H_patches, W_patches)
