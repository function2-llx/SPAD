"""Tests for ViT components with DINOv3 HF-compatible key naming."""
from pathlib import Path

import pytest
import torch

from pumit.model.vit import (
    Attention,
    Block,
    Embeddings,
    LayerScale,
    Mlp,
    SPADPatchEmbed,
    ViT,
    ViTConfig,
)
from pumit.spadop.conv import uniform_inflator


class _IdentityBlock(torch.nn.Module):
    def forward(self, x, rope, attn_bias=None):
        return x


class TestLayerScale:
    def test_key_name(self):
        ls = LayerScale(768)
        assert list(ls.state_dict().keys()) == ['lambda1']

    def test_forward_shape(self):
        ls = LayerScale(768)
        x = torch.randn(2, 10, 768)
        out = ls(x)
        assert out.shape == x.shape

    def test_init_ones(self):
        ls = LayerScale(768)
        assert torch.allclose(ls.lambda1, torch.ones(768))


class TestMlp:
    def test_key_names(self):
        mlp = Mlp(768, 3072)
        assert sorted(mlp.state_dict().keys()) == [
            'down_proj.bias', 'down_proj.weight', 'up_proj.bias', 'up_proj.weight',
        ]

    def test_forward_shape(self):
        mlp = Mlp(768, 3072)
        x = torch.randn(2, 10, 768)
        out = mlp(x)
        assert out.shape == (2, 10, 768)

    def test_no_bias(self):
        mlp = Mlp(768, 3072, bias=False)
        assert sorted(mlp.state_dict().keys()) == ['down_proj.weight', 'up_proj.weight']


class TestAttention:
    def test_key_names(self):
        attn = Attention(
            dim=768, num_heads=12,
            query_bias=True, key_bias=False, value_bias=True, proj_bias=True,
        )
        expected = [
            'k_proj.weight', 'o_proj.bias', 'o_proj.weight',
            'q_proj.bias', 'q_proj.weight',
            'v_proj.bias', 'v_proj.weight',
        ]
        assert sorted(attn.state_dict().keys()) == expected

    def test_key_names_no_bias(self):
        attn = Attention(
            dim=768, num_heads=12,
            query_bias=False, key_bias=False, value_bias=False, proj_bias=False,
        )
        expected = ['k_proj.weight', 'o_proj.weight', 'q_proj.weight', 'v_proj.weight']
        assert sorted(attn.state_dict().keys()) == expected


class TestBlock:
    def test_key_prefixes_match_checkpoint(self):
        block = Block(
            dim=768, num_heads=12, intermediate_size=3072,
            query_bias=True, key_bias=False, value_bias=True,
            proj_bias=True, mlp_bias=True,
        )
        prefixes = {k.rsplit('.', 1)[0] for k in block.state_dict().keys()}
        expected = {
            'attention.k_proj', 'attention.o_proj', 'attention.q_proj', 'attention.v_proj',
            'layer_scale1', 'layer_scale2',
            'mlp.down_proj', 'mlp.up_proj',
            'norm1', 'norm2',
        }
        assert prefixes == expected


class TestSPADPatchEmbed:
    def test_key_names(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        assert sorted(pe.state_dict().keys()) == ['bias', 'weight']

    def test_forward_da0(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        x = torch.randn(1, 3, 16, 32, 32)
        out = pe(x, da=0)
        assert out.shape == (1, 768, 1, 2, 2)

    def test_forward_da1(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        x = torch.randn(1, 3, 32, 32, 32)
        out = pe(x, da=1)
        # da=1 -> new_depth = 16 >> 1 = 8, stride_d = 8, 32/8 = 4
        assert out.shape == (1, 768, 4, 2, 2)

    def test_2d_inflation(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        weight_2d = torch.randn(768, 3, 16, 16)
        fake_2d = {'weight': weight_2d, 'bias': torch.randn(768)}
        pe.load_state_dict(fake_2d, strict=True)
        expected = uniform_inflator(weight_2d, 16).permute(1, 2, 0, 3, 4)
        torch.testing.assert_close(pe.weight, expected)
        assert pe.inflator is uniform_inflator

    def test_2d_inflation_with_resize(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        fake_2d = {'weight': torch.randn(768, 3, 14, 14), 'bias': torch.randn(768)}
        pe.load_state_dict(fake_2d, strict=True)
        assert pe.weight.shape == (768, 3, 16, 16, 16)

    def test_3d_weight_no_inflation(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        fake_3d = {'weight': torch.randn(768, 3, 16, 16, 16), 'bias': torch.randn(768)}
        pe.load_state_dict(fake_3d, strict=True)
        assert pe.weight.shape == (768, 3, 16, 16, 16)

    def test_max_adapt(self):
        pe = SPADPatchEmbed(patch_size=16, in_channels=3, embed_dim=768)
        assert pe.max_adapt == 4  # 16 = 2^4

    def test_power_of_2_assert(self):
        with pytest.raises(AssertionError):
            SPADPatchEmbed(patch_size=(15, 16, 16), in_channels=3, embed_dim=768)


class TestEmbeddings:
    def test_key_names(self):
        emb = Embeddings(embed_dim=768, num_register_tokens=4, patch_size=16, in_channels=3)
        assert sorted(emb.state_dict().keys()) == [
            'cls_token', 'patch_embeddings.bias', 'patch_embeddings.weight', 'register_tokens',
        ]

    def test_mask_token_popped_on_load(self):
        emb = Embeddings(embed_dim=768, num_register_tokens=4, patch_size=16, in_channels=3)
        sd = emb.state_dict()
        sd['mask_token'] = torch.randn(1, 1, 768)
        # Should load without error (mask_token is popped)
        missing, unexpected = emb.load_state_dict(sd, strict=False)
        assert 'mask_token' not in [k for k in unexpected]


class TestViTWeightLoading:
    def test_load_dinov3_strict(self):
        """DINOv3 ViT-B checkpoint loads with no missing keys."""
        p = Path('pretrained/dinov3-vitb16/model.safetensors')
        if not p.exists():
            pytest.skip('checkpoint not available')
        import safetensors.torch as st
        config = ViTConfig()
        model = ViT(config)
        sd = st.load_file(str(p))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        allowed_unexpected = {'embeddings.mask_token', 'rope_embeddings.inv_freq'}
        assert set(unexpected) <= allowed_unexpected, f'Unexpected: {set(unexpected) - allowed_unexpected}'
        assert len(missing) == 0, f'Missing: {missing}'

    def test_config_defaults_match_vitb(self):
        """Default ViTConfig matches ViT-B/16 architecture."""
        config = ViTConfig()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072
        assert config.patch_size == 16
        assert config.num_register_tokens == 4

    def test_vit_key_naming(self):
        """ViT state dict keys match DINOv3 HF checkpoint naming convention."""
        config = ViTConfig(num_hidden_layers=2)
        model = ViT(config)
        keys = set(model.state_dict().keys())

        # Check embeddings keys
        assert 'embeddings.cls_token' in keys
        assert 'embeddings.register_tokens' in keys
        assert 'embeddings.patch_embeddings.weight' in keys
        assert 'embeddings.patch_embeddings.bias' in keys

        # Check block keys (layer.0.*)
        assert 'layer.0.norm1.weight' in keys
        assert 'layer.0.norm1.bias' in keys
        assert 'layer.0.norm2.weight' in keys
        assert 'layer.0.norm2.bias' in keys
        assert 'layer.0.attention.q_proj.weight' in keys
        assert 'layer.0.attention.q_proj.bias' in keys
        assert 'layer.0.attention.k_proj.weight' in keys
        assert 'layer.0.attention.v_proj.weight' in keys
        assert 'layer.0.attention.v_proj.bias' in keys
        assert 'layer.0.attention.o_proj.weight' in keys
        assert 'layer.0.attention.o_proj.bias' in keys
        assert 'layer.0.layer_scale1.lambda1' in keys
        assert 'layer.0.layer_scale2.lambda1' in keys
        assert 'layer.0.mlp.up_proj.weight' in keys
        assert 'layer.0.mlp.up_proj.bias' in keys
        assert 'layer.0.mlp.down_proj.weight' in keys
        assert 'layer.0.mlp.down_proj.bias' in keys

        # k_proj should NOT have bias (key_bias=False)
        assert 'layer.0.attention.k_proj.bias' not in keys

        # Check final norm
        assert 'norm.weight' in keys
        assert 'norm.bias' in keys

    def test_vit_properties(self):
        config = ViTConfig(num_register_tokens=4)
        model = ViT(config)
        assert model.embed_dim == 768
        assert model.n_prefix == 5  # 1 CLS + 4 registers
        assert model.patch_embed is model.embeddings.patch_embeddings

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
    def test_forward_with_bdm(self):
        """forward() with BDM attn_bias produces same results as sequential encode_image calls."""
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        from pumit.model.rope import build_rope
        import einops

        config = ViTConfig(num_hidden_layers=2, hidden_size=192, num_attention_heads=3, intermediate_size=768)
        model = ViT(config).cuda().eval()

        torch.manual_seed(42)
        B, D, H, W = 2, 1, 8, 8
        x = torch.randn(B, 3, D * 16, H * 16, W * 16, device='cuda')
        num_patches = D * H * W
        n_prefix = model.n_prefix

        idx1 = torch.stack([torch.randperm(num_patches, device='cuda')[:45] for _ in range(B)]).sort(dim=1).values
        idx2 = torch.stack([torch.randperm(num_patches, device='cuda')[:16] for _ in range(B)]).sort(dim=1).values

        with torch.no_grad(), torch.amp.autocast('cuda'):
            cls1_seq, patches1_seq = model.encode_image(x, da=0, visible_idx=idx1)
            cls2_seq, patches2_seq = model.encode_image(x, da=0, visible_idx=idx2)

            # Manually pack and call forward()
            patch_tokens = model.embeddings.patch_embeddings(x, da=0)
            shape = patch_tokens.shape[2:]
            patch_tokens = einops.rearrange(patch_tokens, 'n c ... -> n (...) c')
            prefix = torch.cat([
                model.embeddings.cls_token.expand(B, -1, -1),
                model.embeddings.register_tokens.expand(B, -1, -1),
            ], dim=1)

            packed_tokens, packed_rope, seq_lens = [], [], []
            for idx in [idx1, idx2]:
                gather = einops.repeat(idx, 'n l -> n l d', d=model.embed_dim)
                vis = patch_tokens.gather(dim=1, index=gather)
                vt = torch.cat([prefix, vis], dim=1)
                patch_rope = model.rope.compute(shape, idx, training=False)
                view_rope = build_rope(patch_rope, n_prefix=n_prefix)
                packed_tokens.append(vt.reshape(-1, model.embed_dim))
                packed_rope.append(view_rope.reshape(-1, 2, model.rope.head_dim))
                seq_lens.extend([vt.shape[1]] * B)

            x_packed = torch.cat(packed_tokens, dim=0).unsqueeze(0)
            rope = torch.cat(packed_rope, dim=0).unsqueeze(0)
            bdm = BlockDiagonalMask.from_seqlens(seq_lens)

            out = model.forward(x_packed, rope, bdm)

            n1 = n_prefix + 45
            n2 = n_prefix + 16
            dino_out = out[0, :B * n1].reshape(B, n1, -1)
            mim_out = out[0, B * n1:].reshape(B, n2, -1)

        torch.testing.assert_close(dino_out[:, 0], cls1_seq, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(mim_out[:, 0], cls2_seq, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(dino_out[:, n_prefix:], patches1_seq, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(mim_out[:, n_prefix:], patches2_seq, atol=1e-2, rtol=1e-2)

    def test_rope_buffers_popped_on_load(self):
        """rope.* keys in state_dict are silently popped during loading."""
        config = ViTConfig(num_hidden_layers=1)
        model = ViT(config)
        sd = model.state_dict()
        # Add fake rope keys
        sd['rope.freqs_cos'] = torch.randn(10)
        sd['rope.freqs_sin'] = torch.randn(10)
        sd['rope_embeddings.inv_freq'] = torch.randn(10)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # These should be silently handled (not appear in unexpected)
        assert 'rope.freqs_cos' not in unexpected
        assert 'rope.freqs_sin' not in unexpected
        assert 'rope_embeddings.inv_freq' not in unexpected


class TestViTGradientCheckpointing:
    def test_partial_gradient_checkpointing_uses_first_n_layers(self, monkeypatch):
        config = ViTConfig(
            num_hidden_layers=4,
            hidden_size=192,
            num_attention_heads=3,
            intermediate_size=768,
            grad_ckpt=True,
            grad_ckpt_first_n_layers=2,
        )
        model = ViT(config, skip_embed=True).train()
        model.layer = torch.nn.ModuleList([
            _IdentityBlock() for _ in range(config.num_hidden_layers)
        ])
        checkpointed = []

        def fake_checkpoint(function, *args, use_reentrant):
            assert use_reentrant is False
            checkpointed.append(function)
            return function(*args)

        monkeypatch.setattr('pumit.model.vit.torch_checkpoint.checkpoint', fake_checkpoint)
        tokens = torch.randn(1, 4, 192)
        rope = torch.zeros(1, 4, 2, 64)
        rope[..., 0, :] = 1.0

        model(tokens, rope)

        assert checkpointed == list(model.layer[:2])

    @pytest.mark.parametrize('grad_ckpt, first_n_layers', [(False, 1), (True, 5), (True, -1)])
    def test_partial_gradient_checkpointing_rejects_invalid_config(self, grad_ckpt, first_n_layers):
        config = ViTConfig(
            num_hidden_layers=4,
            hidden_size=192,
            num_attention_heads=3,
            intermediate_size=768,
            grad_ckpt=grad_ckpt,
            grad_ckpt_first_n_layers=first_n_layers,
        )
        with pytest.raises(ValueError):
            ViT(config, skip_embed=True)


class TestViTSkipEmbed:
    def test_vit_skip_embed(self):
        """ViT with skip_embed=True has no embeddings attribute; forward() still works."""
        config = ViTConfig(num_hidden_layers=2, hidden_size=192, num_attention_heads=3, intermediate_size=768)
        model = ViT(config, skip_embed=True)
        assert not hasattr(model, 'embeddings')
        assert model.skip_embed is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers')
    def test_vit_skip_embed_forward(self):
        """ViT with skip_embed=True: forward() still works on pre-embedded tokens."""
        config = ViTConfig(num_hidden_layers=2, hidden_size=192, num_attention_heads=3, intermediate_size=768)
        model = ViT(config, skip_embed=True).cuda().eval()

        B, L, C = 2, 10, 192
        tokens = torch.randn(B, L, C, device='cuda')
        rope = torch.zeros(B, L, 2, 64, device='cuda')
        rope[..., 0, :] = 1.0  # cos=1, sin=0 (identity)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.forward(tokens, rope)
        assert out.shape == (B, L, C)

    def test_embeddings_forward(self):
        """Embeddings.forward() returns (B, n_patches, embed_dim) at da=0."""
        config = ViTConfig(num_hidden_layers=1, hidden_size=192, num_attention_heads=3, intermediate_size=768)
        model = ViT(config)
        B, D, H, W = 1, 1, 2, 2
        x = torch.randn(B, 3, D * 16, H * 16, W * 16)
        out = model.embeddings(x, da=0)
        expected_patches = D * H * W
        assert out.shape == (B, expected_patches, 192)

    def test_embeddings_forward_da2(self):
        """Embeddings.forward() at da=2 produces correct patch count for reduced depth stride."""
        config = ViTConfig(num_hidden_layers=1, hidden_size=192, num_attention_heads=3, intermediate_size=768)
        model = ViT(config)
        # da=2 -> depth_stride = 16 >> 2 = 4, so D_in=64 -> 64/4 = 16 depth patches
        B, D_in, H, W = 1, 64, 32, 32
        x = torch.randn(B, 3, D_in, H, W)
        out = model.embeddings(x, da=2)
        expected_patches = (D_in // 4) * (H // 16) * (W // 16)  # 16 * 2 * 2 = 64
        assert out.shape == (B, expected_patches, 192)
