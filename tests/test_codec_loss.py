"""Tests for VAE loss."""

import torch
import pytest


class TestKLDivergence:
    def test_zero_for_standard_normal(self):
        from pumit.codec.loss import kl_divergence
        # mean=0, logvar=0 -> std=1 -> standard normal -> KL=0
        mean = torch.zeros(1, 16, 2, 4, 4)
        logvar = torch.zeros(1, 16, 2, 4, 4)
        kl = kl_divergence(mean, logvar)
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_non_standard(self):
        from pumit.codec.loss import kl_divergence
        mean = torch.ones(1, 16, 2, 4, 4)
        logvar = torch.zeros(1, 16, 2, 4, 4)
        kl = kl_divergence(mean, logvar)
        assert kl.item() > 0


class TestVAELoss:
    def test_forward(self):
        from pumit.codec.loss import VAELoss, CodecOutput
        loss_fn = VAELoss()
        x = torch.randn(1, 3, 4, 64, 64).clamp(-1, 1)
        output = CodecOutput(
            recon=torch.randn(1, 3, 4, 64, 64).clamp(-1, 1).requires_grad_(True),
            mean=torch.randn(1, 16, 1, 8, 8).requires_grad_(True),
            logvar=torch.zeros(1, 16, 1, 8, 8).requires_grad_(True),
        )
        losses = loss_fn(x, output)
        assert 'loss' in losses
        assert 'l1' in losses
        assert 'kl' in losses
        assert losses['loss'].requires_grad


class TestCodecLoss:
    def test_with_logvar(self):
        from pumit.codec.loss import CodecLoss, CodecOutput
        loss_fn = CodecLoss()
        x = torch.randn(1, 3, 4, 64, 64).clamp(-1, 1)
        output = CodecOutput(
            recon=torch.randn(1, 3, 4, 64, 64).clamp(-1, 1).requires_grad_(True),
            mean=torch.randn(1, 16, 1, 8, 8).requires_grad_(True),
            logvar=torch.zeros(1, 16, 1, 8, 8).requires_grad_(True),
        )
        losses = loss_fn(x, output)
        assert 'loss' in losses
        assert 'l1' in losses
        assert 'kl' in losses
        assert losses['loss'].requires_grad

    def test_without_logvar(self):
        from pumit.codec.loss import CodecLoss, CodecOutput
        loss_fn = CodecLoss()
        x = torch.randn(1, 3, 4, 64, 64).clamp(-1, 1)
        output = CodecOutput(
            recon=torch.randn(1, 3, 4, 64, 64).clamp(-1, 1).requires_grad_(True),
            mean=torch.randn(1, 16, 1, 8, 8).requires_grad_(True),
        )
        losses = loss_fn(x, output)
        assert 'loss' in losses
        assert 'l1' in losses
        assert 'kl' not in losses
        assert losses['loss'].requires_grad
