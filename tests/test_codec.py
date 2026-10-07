"""Tests for SPAD KL-VAE."""

from pathlib import Path

import torch
import pytest
from pumit.codec.flux1 import ResnetBlock, AttnBlock, Downsample, Upsample, Encoder, Decoder


class TestResnetBlock:
    def test_same_channels(self):
        block = ResnetBlock(128, 128)
        x = torch.randn(1, 128, 8, 16, 16)
        y = block(x, da=0)
        assert y.shape == (1, 128, 8, 16, 16)

    def test_channel_change(self):
        block = ResnetBlock(128, 256)
        x = torch.randn(1, 128, 8, 16, 16)
        y = block(x, da=0)
        assert y.shape == (1, 256, 8, 16, 16)

    def test_da_none(self):
        block = ResnetBlock(128, 128)
        x = torch.randn(1, 128, 8, 16, 16)
        y = block(x, da=None)
        assert y.shape == (1, 128, 8, 16, 16)

    def test_conv_shortcut_is_plain_conv3d(self):
        """conv_shortcut should be plain nn.Conv3d, not a SPAD subclass."""
        block = ResnetBlock(128, 256)
        assert type(block.conv_shortcut) is torch.nn.Conv3d


class TestAttnBlock:
    @pytest.fixture(autouse=True)
    def cuda_skip(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required for memory_efficient_attention')

    def test_shape_preserved(self):
        block = AttnBlock(512).cuda().bfloat16()
        x = torch.randn(1, 512, 4, 8, 8, device='cuda', dtype=torch.bfloat16)
        y = block(x)
        assert y.shape == x.shape

    def test_batched_shape_preserved(self):
        block = AttnBlock(512).cuda().bfloat16()
        x = torch.randn(4, 512, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        y = block(x)
        assert y.shape == x.shape

    def test_batch_independence(self):
        """Each sample in the batch should be processed independently."""
        block = AttnBlock(512).cuda().bfloat16()
        block.eval()
        x = torch.randn(2, 512, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        y_batched = block(x)
        y0 = block(x[0:1])
        y1 = block(x[1:2])
        torch.testing.assert_close(y_batched[0:1], y0, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y_batched[1:2], y1, atol=1e-2, rtol=1e-2)


class TestDownsample:
    def test_isotropic_halves_all(self):
        ds = Downsample(128)
        x = torch.randn(1, 128, 8, 16, 16)
        y = ds(x, da=0)
        assert y.shape == (1, 128, 4, 8, 8)

    def test_anisotropic_skips_depth(self):
        ds = Downsample(128)
        x = torch.randn(1, 128, 4, 16, 16)
        y = ds(x, da=2)
        assert y.shape == (1, 128, 4, 8, 8)

    def test_da_none_skips_depth(self):
        ds = Downsample(128)
        x = torch.randn(1, 128, 4, 16, 16)
        y = ds(x, da=None)
        assert y.shape == (1, 128, 4, 8, 8)


class TestUpsample:
    def test_doubles_all_with_depth(self):
        us = Upsample(128)
        x = torch.randn(1, 128, 4, 8, 8)
        y = us(x, da=0, upsample_depth=True)
        assert y.shape == (1, 128, 8, 16, 16)

    def test_hw_only_upsample(self):
        us = Upsample(128)
        x = torch.randn(1, 128, 4, 8, 8)
        y = us(x, da=0, upsample_depth=False)
        assert y.shape == (1, 128, 4, 16, 16)

    def test_da_none_hw_only(self):
        us = Upsample(128)
        x = torch.randn(1, 128, 4, 8, 8)
        y = us(x, da=None, upsample_depth=False)
        assert y.shape == (1, 128, 4, 16, 16)


class TestEncoder:
    @pytest.fixture(autouse=True)
    def cuda_skip(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required for memory_efficient_attention')

    def test_output_shape_isotropic(self):
        enc = Encoder().cuda().bfloat16()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        y = enc(x, da=0)
        # 8x downsample, double_z=True -> 32 channels
        assert y.shape == (1, 32, 2, 4, 4)

    def test_output_shape_anisotropic(self):
        enc = Encoder().cuda().bfloat16()
        # DA=2: depth should only downsample once
        x = torch.randn(1, 3, 4, 32, 32, device='cuda', dtype=torch.bfloat16)
        y = enc(x, da=2)
        assert y.shape == (1, 32, 2, 4, 4)  # D: 4->2 (one downsample)

    def test_da_none(self):
        enc = Encoder().cuda().bfloat16()
        x = torch.randn(1, 3, 1, 32, 32, device='cuda', dtype=torch.bfloat16)
        y = enc(x, da=None)
        # da=None: depth never downsampled, D stays 1
        assert y.shape == (1, 32, 1, 4, 4)

    def test_with_quant_conv(self):
        enc = Encoder(latent_channels=32, use_quant_conv=True).cuda().bfloat16()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        y = enc(x, da=0)
        assert y.shape == (1, 64, 2, 4, 4)
        assert hasattr(enc, 'quant_conv')


class TestDecoder:
    @pytest.fixture(autouse=True)
    def cuda_skip(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required for memory_efficient_attention')

    def test_output_shape_isotropic(self):
        dec = Decoder().cuda().bfloat16()
        # Isotropic input after 3 downsamples from original (da=0, nds=3)
        x = torch.randn(1, 16, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        y = dec(x, da=0)
        # 8x upsample
        assert y.shape == (1, 3, 16, 32, 32)

    def test_da_none(self):
        dec = Decoder().cuda().bfloat16()
        x = torch.randn(1, 16, 1, 4, 4, device='cuda', dtype=torch.bfloat16)
        y = dec(x, da=None)
        # da=None: nds=0, only HW upsampled
        assert y.shape == (1, 3, 1, 32, 32)

    def test_with_post_quant_conv(self):
        dec = Decoder(latent_channels=32, use_post_quant_conv=True).cuda().bfloat16()
        x = torch.randn(1, 32, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        y = dec(x, da=0)
        assert y.shape == (1, 3, 16, 32, 32)
        assert hasattr(dec, 'post_quant_conv')


class TestSPADKLVAE:
    @pytest.fixture(autouse=True)
    def cuda_skip(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required for memory_efficient_attention')

    def _make_vae(self):
        from pumit.codec import SPADKLVAE
        return SPADKLVAE().cuda().bfloat16()

    def test_encode_shape(self):
        vae = self._make_vae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        mean, logvar = vae.encode(x, da=0)
        assert mean.shape == (1, 16, 2, 4, 4)
        assert logvar.shape == (1, 16, 2, 4, 4)

    def test_decode_shape(self):
        vae = self._make_vae()
        z = torch.randn(1, 16, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        x = vae.decode(z, da=0)
        assert x.shape == (1, 3, 16, 32, 32)

    def test_roundtrip(self):
        vae = self._make_vae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        mean, logvar = vae.encode(x, da=0)
        recon = vae.decode(mean, da=0)  # deterministic decode from mean
        assert recon.shape == x.shape

    def test_forward(self):
        vae = self._make_vae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        output = vae(x, da=0, da_dec=0)
        assert output.recon.shape == x.shape

    def test_forward_da_none(self):
        vae = self._make_vae()
        x = torch.randn(1, 3, 1, 32, 32, device='cuda', dtype=torch.bfloat16)
        output = vae(x, da=None, da_dec=None)
        assert output.recon.shape == x.shape

    def test_forward_assertions(self):
        vae = self._make_vae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        # da=None, da_dec=0 should fail assertion
        with pytest.raises(AssertionError):
            vae(x, da=None, da_dec=0)
        # da=0, da_dec=None should fail assertion
        with pytest.raises(AssertionError):
            vae(x, da=0, da_dec=None)
        # |da - da_dec| > 1 should fail assertion
        with pytest.raises(AssertionError):
            vae(x, da=0, da_dec=2)


class TestSPADFlux2AE:
    @pytest.fixture(autouse=True)
    def cuda_skip(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required for memory_efficient_attention')

    def _make_ae(self):
        from pumit.codec.flux2 import SPADFlux2AE
        return SPADFlux2AE().cuda().bfloat16()

    def test_encode_shape(self):
        ae = self._make_ae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        z = ae.encode(x, da=0)
        assert z.shape == (1, 32, 2, 4, 4)

    def test_decode_shape(self):
        ae = self._make_ae()
        z = torch.randn(1, 32, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        x = ae.decode(z, da=0)
        assert x.shape == (1, 3, 16, 32, 32)

    def test_roundtrip(self):
        ae = self._make_ae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        z = ae.encode(x, da=0)
        recon = ae.decode(z, da=0)
        assert recon.shape == x.shape

    def test_forward_returns_codec_output(self):
        from pumit.codec.loss import CodecOutput
        ae = self._make_ae()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        output = ae(x, da=0, da_dec=0)
        assert isinstance(output, CodecOutput)
        assert output.recon.shape == x.shape
        assert output.mean.shape == (1, 32, 2, 4, 4)
        assert output.logvar is None

    def test_forward_da_none(self):
        ae = self._make_ae()
        x = torch.randn(1, 3, 1, 32, 32, device='cuda', dtype=torch.bfloat16)
        output = ae(x, da=None, da_dec=None)
        assert output.recon.shape == x.shape

    def test_encode_deterministic(self):
        ae = self._make_ae()
        ae.eval()
        x = torch.randn(1, 3, 16, 32, 32, device='cuda', dtype=torch.bfloat16)
        z1 = ae.encode(x, da=0)
        z2 = ae.encode(x, da=0)
        torch.testing.assert_close(z1, z2)

    def test_has_quant_convs(self):
        ae = self._make_ae()
        assert ae.encoder.quant_conv is not None
        assert ae.decoder.post_quant_conv is not None


class TestWeightLoading:
    PRETRAINED_PATH = Path(__file__).resolve().parent.parent / 'pretrained' / 'flux1-vae.pt'

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / 'pretrained' / 'flux1-vae.pt').exists(),
        reason='pretrained/flux1-vae.pt not found',
    )
    def test_load_pretrained_state_dict(self):
        """Verify our model can load the cached FLUX.1 VAE state_dict."""
        from pumit.codec import SPADKLVAE

        sd = torch.load(self.PRETRAINED_PATH, map_location='cpu', weights_only=True)
        ours = SPADKLVAE()

        # Verify key names match (every pretrained key should exist in our model)
        our_keys = set(ours.state_dict().keys())
        missing = set(sd.keys()) - our_keys
        assert not missing, f"Missing keys in our model: {missing}"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / 'pretrained' / 'flux1-vae.pt').exists(),
        reason='pretrained/flux1-vae.pt not found',
    )
    def test_inflation_changes_weights(self):
        """Loading 2D weights into 3D model should produce 5D conv weights."""
        from pumit.codec import SPADKLVAE

        sd = torch.load(self.PRETRAINED_PATH, map_location='cpu', weights_only=True)
        ours = SPADKLVAE()
        # Load 2D weights -- SPAD Conv3d._load_from_state_dict inflates them
        ours.load_state_dict(sd, strict=False)

        # Conv weights should now be 5D (inflated from 4D)
        w = ours.encoder.conv_in.weight
        assert w.ndim == 5, f"Expected 5D weight after inflation, got {w.ndim}D"


    FLUX2_PRETRAINED_PATH = Path(__file__).resolve().parent.parent / 'pretrained' / 'flux2-vae.pt'

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / 'pretrained' / 'flux2-vae.pt').exists(),
        reason='pretrained/flux2-vae.pt not found',
    )
    def test_load_flux2_pretrained(self):
        from pumit.codec.flux2 import SPADFlux2AE
        sd = torch.load(self.FLUX2_PRETRAINED_PATH, map_location='cpu', weights_only=True)
        model = SPADFlux2AE()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"Unexpected keys: {unexpected}"
        w = model.encoder.conv_in.weight
        assert w.ndim == 5

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / 'pretrained' / 'flux2-vae.pt').exists(),
        reason='pretrained/flux2-vae.pt not found',
    )
    def test_flux2_quant_conv_loaded(self):
        from pumit.codec.flux2 import SPADFlux2AE
        sd = torch.load(self.FLUX2_PRETRAINED_PATH, map_location='cpu', weights_only=True)
        model = SPADFlux2AE()
        model.load_state_dict(sd, strict=False)
        w = model.encoder.quant_conv.weight
        assert w.ndim == 5
        assert w.shape == (64, 64, 1, 1, 1)
