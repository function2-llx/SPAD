import torch

from pumit.transforms.patchify import patchify


def test_patchify_shapes_da0():
    img = torch.randn(3, 32, 384, 384)
    patches, shape = patchify(img, da=0)
    assert patches.shape == (2 * 24 * 24, 3, 16, 16, 16)
    assert shape == (2, 24, 24)


def test_patchify_shapes_da2():
    img = torch.randn(3, 16, 384, 384)
    patches, shape = patchify(img, da=2)
    assert patches.shape == (4 * 24 * 24, 3, 16, 16, 16)
    assert shape == (4, 24, 24)


def test_patchify_shapes_2d():
    img = torch.randn(3, 1, 384, 384)
    patches, shape = patchify(img, da=5)  # da >= max_adapt, patch_d = 1
    assert patches.shape == (1 * 24 * 24, 3, 16, 16, 16)
    assert shape == (1, 24, 24)


def test_patchify_equivalence_with_spad():
    """repeat_interleave + conv(da=0) == SPADPatchEmbed(da=N)."""
    from pumit.model.vit import ViT, ViTConfig

    config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=768,
    )
    vit = ViT(config)
    vit.eval()

    for da in [0, 1, 2, 3, 4]:
        patch_d = 16 >> min(da, 4)
        depth = patch_d * 4  # 4 patches in depth
        img = torch.randn(1, 3, depth, 128, 128)

        # Path A: direct SPADPatchEmbed with da
        with torch.no_grad():
            embed_a = vit.embeddings.patch_embeddings(img, da=da)
            tokens_a = embed_a.flatten(2).transpose(1, 2)  # (1, n_patches, 192)

        # Path B: patchify then SPADPatchEmbed with da=0 on each patch
        patches, spatial_shape = patchify(img[0], da=da)
        with torch.no_grad():
            # patches: (n_patches, 3, 16, 16, 16), treated as batch
            # conv3d with kernel=stride=16 on 16x16x16 input -> (n_patches, 192, 1, 1, 1)
            embed_b = vit.embeddings.patch_embeddings(patches, da=0)
            tokens_b = embed_b.reshape(1, -1, 192)  # (1, n_patches, 192)

        assert torch.allclose(tokens_a, tokens_b, atol=1e-4), (
            f"Equivalence failed at da={da}: max diff={torch.max(torch.abs(tokens_a - tokens_b))}"
        )
