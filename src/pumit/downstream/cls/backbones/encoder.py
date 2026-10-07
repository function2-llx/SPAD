from __future__ import annotations

from pathlib import Path

import safetensors.torch as st
import torch
from torch import Tensor

from pumit.model.vit import ViT, ViTConfig

# DINOv3 safetensors carries a couple of keys the bare ViT does not use.
_DINOV3_ALLOWED_UNEXPECTED = {'embeddings.mask_token', 'rope_embeddings.inv_freq'}


def slice_teacher_vit(model_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Extract the EMA teacher encoder weights from a UCPTModel state_dict.

    Keys look like 'teacher_vit.<name>' (optionally '_orig_mod.'-prefixed from
    torch.compile). Returns a dict keyed by bare '<name>' loadable into ViT.
    """
    out: dict[str, Tensor] = {}
    for key, val in model_sd.items():
        key = key.removeprefix('_orig_mod.')
        if key.startswith('teacher_vit.'):
            out[key.removeprefix('teacher_vit.')] = val
    if not out:
        raise KeyError('no teacher_vit.* keys found in checkpoint model state_dict')
    return out


def _finalize(vit: ViT, device: str, trainable: bool) -> ViT:
    vit.to(device)
    if trainable:
        vit.train()
    else:
        vit.eval()
        vit.requires_grad_(False)
    return vit


def _teacher_vit_cache_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_suffix('.teacher_vit.safetensors')


def _load_teacher_vit_sd(ckpt_path: Path) -> dict[str, Tensor]:
    """Load only the teacher_vit slice, caching to a sidecar safetensors file.

    On first call for a given checkpoint, loads the full .pt (5.5 GB), slices
    teacher_vit (~1.2 GB), writes the sidecar, then returns the slice. Subsequent
    calls load only the sidecar. Invalidates if the checkpoint is newer.
    """
    cache = _teacher_vit_cache_path(ckpt_path)
    if cache.exists() and cache.stat().st_mtime >= ckpt_path.stat().st_mtime:
        return st.load_file(str(cache))
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sliced = slice_teacher_vit(ckpt['model'])
    del ckpt
    st.save_file(sliced, str(cache))
    return sliced


def load_encoder(ckpt_path: str | Path, vit_cfg: ViTConfig, device: str,
                 trainable: bool = False) -> ViT:
    """Load a ViT encoder (EMA teacher) from a UCPT-format checkpoint."""
    sliced = _load_teacher_vit_sd(Path(ckpt_path))
    vit = ViT(vit_cfg)
    vit.load_state_dict(sliced, strict=True)
    return _finalize(vit, device, trainable)


def load_dinov3_encoder(safetensors_path: str | Path, vit_cfg: ViTConfig, device: str,
                        trainable: bool = False) -> ViT:
    """Load the official DINOv3 checkpoint directly (2D->3D inflated on load).

    The 2D conv patch-embed weight is interpolated + inflated to 3D by
    SPADPatchEmbed._load_from_state_dict, so one encoder serves 2D (da=max_adapt)
    and 3D (da=0). This is the from-2D-init baseline, no medical pretraining.
    """
    sd = st.load_file(str(safetensors_path))
    vit = ViT(vit_cfg)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    actual_unexpected = set(unexpected) - _DINOV3_ALLOWED_UNEXPECTED
    if actual_unexpected:
        raise RuntimeError(f'unexpected DINOv3 keys: {actual_unexpected}')
    if missing:
        raise RuntimeError(f'missing keys loading DINOv3: {missing}')
    return _finalize(vit, device, trainable)
