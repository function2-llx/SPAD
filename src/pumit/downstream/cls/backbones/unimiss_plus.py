"""UniMiSS+ baseline as a backbone module (3D-only, cross-dimensional self-supervised MiT).

Contract: unimiss_plus(*, dims=3, device, weights, trainable) -> nn.Module with
forward(x)->(global,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

UniMiSS+ (Xie et al., IEEE-TPAMI 2024; CC BY-NC-ND 4.0) is a cross-dimensional (2D+3D) DINO-style
self-supervised encoder pretrained on DeepLesion CT. The encoder is a pyramid MiT (Mix Transformer)
with switchable 2D/3D patch embedding. This adapter uses the 3D-only classification-downstream
variant (Downstream/3D/RICORD/nets/MiTPlus.py), vendored under _unimiss_plus/, with the exact
`model_plus` config (embed_dims=[32,64,128,256,320,320], depths=[1,2,4,2], num_heads=[1,2,4,8],
sr_ratios=[8,4,2,1]) and the official student-branch loading contract -- no key-guessing.

Weights: the released `UniMissPlus.pth` (GDrive; see README), a DINO dump under `{'student',
'teacher', ...}`. We load the official-default **student** branch (`module.backbone.transformer.*`
stripped to bare keys), matching the official downstream `pre_type='student'`.

3D-only; single-channel. The pretraining crop is anisotropic [16,96,96] (16 depth x 96 in-plane),
but MedMNIST3D volumes are isotropic 64^3, so the input is an `img_size` cube and the three
in-plane-only strides are made cubic to match (`isotropize_strides`). The pyramid then halves
uniformly and its /16 stage is 12^3 at 192^3, level with the flat ViTs; the readout is one stage
deeper at 6^3 = 216. Unlike UniMiSS, the first two conv blocks run at full input resolution, so
their activations are recomputed in backward (`_Recompute`) to reach the protocol batch size.
The encoder's first block is InstanceNorm, which absorbs input scale, so min-max
[0,1] is used (MedMNIST3D is uint8-scaled, not raw HU; the native HU windowing does not transfer).
Readout = the official pre-head CLS/GAP average `0.5*(norm_new(CLS) + mean(norm_new(patches)))`
(320-dim), so the harness's linear head reproduces the official `0.5*[head_new(norm(CLS)) +
head_new(mean(norm(patches)))]` logits exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ._unimiss_plus.mitplus import model_plus
from .unimiss import isotropize_strides

NATIVE_SHAPE = (16, 96, 96)   # UniMiSS+ 3D pretraining crop (depth, in-plane, in-plane)
EMBED_DIM = 320              # stage-4 (embed_dims[-1]) dim


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} in [0,1] at native res (CPU). Shape-blind.

    3D-only; raises on 2D. Min-max [0,1] (MedMNIST3D is uint8-scaled; the encoder's first
    InstanceNorm block absorbs input scale, so the exact range is non-critical).
    """
    if not is_3d:
        raise ValueError('UniMiSS+ is a 3D-only baseline; cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0   # [0,1]
    return {'x': x.unsqueeze(1)}                                         # (N,1,D,H,W)


def _build_encoder(
    input_channels: int = 1, *, use_cls_tokens: bool = True, drop_path_rate: float = 0.0,
) -> nn.Module:
    """Construct the UniMiSS+ MiT_encoder (3D-only) with the official `model_plus` config (random init)."""
    return model_plus(
        norm_cfg3D='IN3', activation_cfg='LeakyReLU', num_classes=2,
        img_size3D=list(NATIVE_SHAPE), in_chans=input_channels, pretrain=False,
        use_cls_tokens=use_cls_tokens, drop_path_rate=drop_path_rate,
    )


def _load_encoder_state_dict(weights: str) -> dict[str, Tensor]:
    """Load the official-default student branch from UniMissPlus.pth and return bare encoder keys.

    The checkpoint is `{'student': {'module.backbone.transformer.<k>': v, ...}, ...}` (the official
    downstream loader defaults to `pre_type='student'`); unwrap `student` and strip the
    `module.backbone.transformer.` prefix so keys match the 3D-only MiT_encoder state_dict.
    `weights_only=False` because the checkpoint carries numpy scalars (trusted official source).
    """
    sd = torch.load(weights, map_location='cpu', weights_only=False)
    if 'student' not in sd:
        raise RuntimeError(f'{weights}: expected a DINO dump with a `student` key')
    student = sd['student']
    return {k[len('module.backbone.transformer.'):] if k.startswith('module.backbone.transformer.') else k: v
            for k, v in student.items() if k.startswith('module.backbone.transformer.')}


# The pretrained MM model carries a 2D branch absent from the 3D-only downstream encoder; its
# keys are the only acceptable unexpected entries after a strict-ish load.
_2D_BRANCH_PREFIXES = ('ConvBlock2D', 'patch_embed2D', 'pos_embed2D', 'cls_tokens2D',
                       'block2D', 'Decblock2D', 'DecEmbed2D', 'DecPosEmbed2D',
                       'TransposeConv2D', 'DeConvBlock2D', 'recon_conv2D')


class _Recompute(nn.Module):
    """Wrap a submodule so its activations are recomputed in backward instead of stored.

    UniMiSS+ runs its first two conv blocks at full input resolution, which is what makes 192^3
    unaffordable; recomputing just those two is enough for bs 32. Attribute access and extra
    forward arguments pass through, so the wrapper is invisible to the rest of the pyramid.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        return checkpoint(self.block, *args, use_reentrant=False, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules['block'], name)


class UniMissPlusEncoder(nn.Module):
    """UniMiSS+ MiT_encoder as a MedMNIST3D baseline; 3D-only.

    forward(x) with x: (B,1,D,H,W) at native res -> resampled to an img_size cube ->
    forward3d_features -> (global[B,320], patch_tokens[B,N,320]); global = the pre-head
    norm_new(CLS) token.
    """

    def __init__(
        self, weights: str, img_size: int = NATIVE_SHAPE[1],
        isotropic: bool = True, recompute_stem: bool = True, drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.model = _build_encoder(drop_path_rate=drop_path_rate)
        if isotropic:
            isotropize_strides(self.model)
        sd = _load_encoder_state_dict(weights)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # norm_new + head_new are downstream-added (the classification head + its norm); they are
        # absent from the pretrained backbone and kept random-init (official behavior), so their
        # presence in `missing` is expected. Any other missing key is a real topology mismatch.
        missing_critical = [k for k in missing if not k.startswith(('norm_new', 'head_new'))]
        if missing_critical:
            raise RuntimeError(f'UniMiSS+ student checkpoint missing encoder keys: {missing_critical}')
        bad_unexpected = [k for k in unexpected
                          if not k.startswith(_2D_BRANCH_PREFIXES) and '.sr2D.' not in k]
        if bad_unexpected:
            raise RuntimeError(
                f'UniMiSS+ student checkpoint has unexpected non-2D-branch keys: {bad_unexpected}')
        # after loading: the wrapper would otherwise prefix these blocks' keys with `.block`
        if recompute_stem:
            self.model.ConvBlock3D0 = _Recompute(self.model.ConvBlock3D0)
            self.model.ConvBlock3D1 = _Recompute(self.model.ConvBlock3D1)
        self.embed_dim = EMBED_DIM

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Group the stems, stage embeddings and blocks, folding readout parameters into the deepest block."""
        model = self.model
        layers = [tuple(model.ConvBlock3D0.parameters()), tuple(model.ConvBlock3D1.parameters())]
        for stage in range(1, 5):
            cls_tokens = getattr(model, f'cls_tokens{stage}')
            embedding_parameters = (
                *getattr(model, f'patch_embed3D{stage}').parameters(),
                getattr(model, f'pos_embed3D{stage}'),
                *((cls_tokens,) if stage == 1 else cls_tokens.parameters()),
            )
            layers.append(embedding_parameters)
            layers.extend(tuple(block.parameters()) for block in getattr(model, f'block{stage}'))
        layers[-1] = (*layers[-1], *model.norm_new.parameters(), *model.head_new.parameters())
        return tuple(layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Isotropic cube: MedMNIST3D volumes are 64^3, so resampling to the anisotropic pretraining
        # crop would squash depth 4x relative to in-plane. The pyramid's per-stage pos-embeds are
        # interpolated to whatever grid the input produces, so a cube is a supported shape.
        x = F.interpolate(x, size=(self.img_size,) * 3, mode='trilinear', align_corners=False)
        return self.model.forward3d_features(x)


def unimiss_plus(
    *, dims: int, device: str, weights: str, img_size: int,
    trainable: bool = False, isotropic: bool = True,
    recompute_stem: bool = True, drop_path_rate: float = 0.0,
):
    """Construct the UniMiSS+ 3D encoder on `device` from the released student branch.

    `weights` is required (path to the GDrive-released `UniMissPlus.pth`; see the UniMiSS+
    README). `img_size` is required and gives an isotropic cube: it sets the pyramid's stage
    grids, so the caller states the resolution rather than inheriting the anisotropic pretraining
    crop. `recompute_stem` trades compute for memory on the two full-resolution conv blocks,
    which is what lets 192^3 run at the protocol batch size; it is mathematically transparent.
    Set `isotropic=False` / `recompute_stem=False` to recover the pretrained schedule and the
    store-everything backward. 3D-only: dims must be 3.
    """
    if dims != 3:
        raise ValueError('UniMiSS+ is a 3D-only baseline; pass dims=3')
    enc = UniMissPlusEncoder(
        weights=weights, img_size=img_size, isotropic=isotropic,
        recompute_stem=recompute_stem, drop_path_rate=drop_path_rate,
    ).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc
