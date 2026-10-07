"""End-to-end integration tests for the plans-constructible SPAD network."""

import pytest
import torch
from torch.nn import functional as F

from pumit.spad_unet.architecture import SPADResEncUNet
from pumit.spad_unet.geometry import sample_da


@pytest.fixture
def model():
    return SPADResEncUNet(input_channels=1, num_classes=4, inference_da=0).cuda()


class TestForwardBackward:
    def test_da0_isotropic(self, model):
        """Full forward + loss + backward at da=0."""
        x = torch.randn(1, 1, 128, 128, 128, device='cuda')
        target = torch.randint(0, 4, (1, 128, 128, 128), device='cuda')

        logits = model(x, da=0)

        loss = F.cross_entropy(logits[0], target)
        loss.backward()

        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_da2_anisotropic(self, model):
        """Full forward + loss + backward at da=2 (anisotropic)."""
        x = torch.randn(1, 1, 32, 128, 128, device='cuda')
        target = torch.randint(0, 4, (1, 32, 128, 128), device='cuda')

        logits = model(x, da=2)

        loss = F.cross_entropy(logits[0], target)
        loss.backward()

        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_deep_supervision_loss(self, model):
        """DS loss with per-level weights."""
        x = torch.randn(1, 1, 64, 64, 64, device='cuda')

        logits = model(x, da=0)

        # Compute per-level targets (nearest downsampled)
        target_full = torch.randint(0, 4, (1, 64, 64, 64), device='cuda')
        losses = []
        for i, logit in enumerate(logits):
            if i == 0:
                t = target_full
            else:
                t = (
                    F.interpolate(
                        target_full.float().unsqueeze(1),
                        size=logit.shape[2:],
                        mode='nearest',
                    )
                    .squeeze(1)
                    .long()
                )
            losses.append(F.cross_entropy(logit, t))

        # Weighted sum (exponential decay)
        weights = [1 / (2**i) for i in range(len(losses))]
        weights[-1] = 0
        total = sum(w * loss for w, loss in zip(weights, losses))
        total.backward()

        assert any(p.grad is not None for p in model.decoder.seg_layers.parameters())


class TestStochasticDA:
    def test_various_continuous_da(self, model):
        """sample_da produces values that produce valid outputs for all datasets."""
        x = torch.randn(1, 1, 96, 128, 128, device='cuda')

        for continuous_da in [0.0, 0.36, 1.49, 1.98]:
            da = sample_da(continuous_da)
            logits = model(x, da=da)
            assert len(logits) == 5
            assert logits[0].shape[2:] == x.shape[2:]

    def test_different_da_same_fullres_shape(self, model):
        """Full-res output always matches input spatial dims."""
        x = torch.randn(1, 1, 64, 128, 128, device='cuda')

        for da in [0, 1, 2, 3]:
            logits = model(x, da=da)
            assert logits[0].shape == (1, 4, 64, 128, 128)


class TestSegHeadChannels:
    def test_seg_head_output_matches_num_classes(self, model):
        """Seg heads produce correct number of output channels."""
        x = torch.randn(1, 1, 64, 64, 64, device='cuda')
        logits = model(x, da=0)
        for logit in logits:
            assert logit.shape[1] == 4  # num_classes
