"""Codec construction and batching utilities for UCPT latent encoding."""

from __future__ import annotations

from pathlib import Path

import einops
import torch
from torch.nn import functional as F

from pumit.codec import SPADFlux2AE, SPADKLVAE

_MAX_COMPILED_ENCODER_SIGNATURES = 24


class _StaticHotEncoder:
    """Compile only recurring input shape and DA signatures within a worker."""

    def __init__(self, encoder, compiled, *, max_signatures: int):
        self.encoder = encoder
        self.compiled = compiled
        self.max_signatures = max_signatures
        self.seen_signatures = set()
        self.compiled_signatures = set()

    def __call__(self, x: torch.Tensor, *, da: int | None):
        signature = (tuple(x.shape), da)
        if signature in self.compiled_signatures:
            return self.compiled(x, da=da)
        if (
            signature in self.seen_signatures
            and len(self.compiled_signatures) < self.max_signatures
        ):
            self.compiled_signatures.add(signature)
            return self.compiled(x, da=da)
        self.seen_signatures.add(signature)
        return self.encoder(x, da=da)


def build_encoder(
    codec_model: str,
    codec_checkpoint: str | Path,
    device: torch.device,
    compile_mode: str = 'default',
):
    if codec_model == 'flux2':
        codec = SPADFlux2AE(grad_ckpt=False)
    else:
        codec = SPADKLVAE(grad_ckpt=False)

    codec_ckpt = torch.load(codec_checkpoint, map_location='cpu', weights_only=False)
    codec_sd = codec_ckpt.get('ema', codec_ckpt.get('model', codec_ckpt))
    codec_sd = {key.removeprefix('_orig_mod.'): value for key, value in codec_sd.items()}
    codec.load_state_dict(codec_sd)

    encoder = codec.encoder
    encoder.requires_grad_(False)
    encoder.eval()
    encoder = encoder.to(device)
    torch._dynamo.config.recompile_limit = 32
    compiled = torch.compile(encoder, dynamic=False, mode=compile_mode)
    return _StaticHotEncoder(
        encoder,
        compiled,
        max_signatures=_MAX_COMPILED_ENCODER_SIGNATURES,
    )


def get_latent_targets(
    encoder,
    x: torch.Tensor,
    da: int | None,
    max_adapt: int = 4,
) -> torch.Tensor:
    """Encode images into flattened mean latent targets."""
    h = encoder(x, da=da)
    targets = h.chunk(2, dim=1)[0]
    pool_d = 2 if da is not None and da < max_adapt else 1
    targets = F.avg_pool3d(targets, (pool_d, 2, 2))
    return einops.rearrange(targets, 'n c ... -> n (...) c')


def get_batch_size(
    da_enc: int | None,
    depth: int,
    size_xy: int,
    memory_budget_gb: float,
) -> int:
    """Compute a batch size from the empirical encoder memory model."""
    del da_enc
    voxels_per_sample = depth * size_xy * size_xy
    mem_per_sample_gb = voxels_per_sample * 1.05e-6
    return max(int(memory_budget_gb / mem_per_sample_gb), 1)
