"""UniMiSS (ECCV 2022) baseline as a backbone module (3D-only, cross-dimensional SSL MiT).

Contract: unimiss(*, dims=3, device, weights, trainable) -> nn.Module with
forward(x)->(global,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

UniMiSS (Xie et al., ECCV 2022; CC BY-NC-ND 4.0) is the predecessor to UniMiSS+ -- the original
cross-dimensional (2D+3D) DINO-style SSL encoder pretrained on DeepLesion CT. Same MiT (Mix
Transformer) family as UniMiSS+; including it lets the table show whether the TPAMI "+" extension
improved transfer. This adapter uses the 3D classification-downstream `model_small` config
(embed_dims=[48,128,256,512], depths=[2,3,4,3], sr_ratios=[6,4,2,1]), vendored under _unimiss/
alongside its nets/utils.py. The official-default **student** branch is loaded
(`module.backbone.transformer.*` stripped to bare).

3D-only; single-channel; min-max [0,1] (InstanceNorm first block absorbs input scale). The
pretraining crop is anisotropic [16,96,96], but MedMNIST3D volumes are isotropic 64^3, so the
input is an `img_size` cube and the stem stride is made cubic to match (`isotropize_strides`).
The pyramid then halves uniformly: at 192^3 its /16 stage is 12^3, level with the flat ViTs, and
the stage-4 readout is 6^3 = 216. Readout = the official pre-head CLS/GAP average (512-dim);
see unimiss_plus.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._unimiss.MiT import model_small

NATIVE_SHAPE = (16, 96, 96)
EMBED_DIM = 512              # stage-4 (embed_dims[-1]) dim for the `small` variant


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} in [0,1] at native res (CPU). Shape-blind. 3D-only."""
    if not is_3d:
        raise ValueError('UniMiSS is a 3D-only baseline; cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0
    return {'x': x.unsqueeze(1)}


def _build_encoder(input_channels: int = 1, *, drop_path_rate: float = 0.0) -> nn.Module:
    """Construct the UniMiSS VisionTransformer (3D-only) with the `model_small` config (random init)."""
    return model_small(
        norm_cfg3D='IN3', activation_cfg='LeakyReLU',
        img_size3D=list(NATIVE_SHAPE), in_chans=input_channels,
        num_classes=2, pretrain=False, drop_path_rate=drop_path_rate,
    )


def isotropize_strides(model: nn.Module) -> int:
    """Make every anisotropic Conv3d stride isotropic, in place; returns how many were changed.

    The pyramid was pretrained on anisotropic CT crops, so its stem strides in-plane only
    ((1,2,2)). Depth then trails in-plane by 2x at every later stage, and no stage is a cube. Our
    volumes are isotropic 64^3, so the reduction should be too: with cubic strides the pyramid
    halves uniformly and its /16 stage is 12^3 at a 192^3 input, matching the flat ViTs.

    Stride is configuration, not a parameter, and each affected kernel is already cubic, so every
    pretrained tensor still loads unchanged. This adapts the downsampling schedule -- a larger
    change than resampling a position table, and worth stating as such.
    """
    changed = 0
    for module in model.modules():
        if isinstance(module, nn.Conv3d) and len(set(module.stride)) > 1:
            if len(set(module.kernel_size)) != 1:
                raise RuntimeError(
                    f'cannot isotropize a non-cubic kernel: stride {module.stride}, '
                    f'kernel {module.kernel_size}'
                )
            module.stride = (max(module.stride),) * 3
            changed += 1
    return changed


def _load_encoder_state_dict(weights: str) -> dict[str, Tensor]:
    """Load the official-default student branch from UniMiss_small.pth and return bare encoder keys.

    The checkpoint is `{'student': {'module.backbone.transformer.<k>': v, ...}, ...}` (the official
    downstream loader defaults to `pre_type='student'`); unwrap `student` and strip the
    `module.backbone.transformer.` prefix so keys match the 3D-only VisionTransformer state_dict.
    `weights_only=False` (numpy scalars; trusted official source).
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


class UniMissEncoder(nn.Module):
    """UniMiSS VisionTransformer as a MedMNIST3D baseline; 3D-only.

    forward(x) with x: (B,1,D,H,W) -> resampled to an img_size cube -> forward3d_features ->
    (global[B,512], patch_tokens[B,N,512]); global = pre-head stage-4 norm_new(CLS).
    """

    def __init__(
        self, weights: str, img_size: int = NATIVE_SHAPE[1],
        isotropic: bool = True, drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.model = _build_encoder(drop_path_rate=drop_path_rate)
        if isotropic:
            isotropize_strides(self.model)
        sd = _load_encoder_state_dict(weights)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        missing_critical = [k for k in missing if not k.startswith(('norm_new', 'head_new'))]
        if missing_critical:
            raise RuntimeError(f'UniMiSS student checkpoint missing encoder keys: {missing_critical}')
        bad_unexpected = [k for k in unexpected
                          if not k.startswith(_2D_BRANCH_PREFIXES) and '.sr2D.' not in k]
        if bad_unexpected:
            raise RuntimeError(
                f'UniMiSS student checkpoint has unexpected non-2D-branch keys: {bad_unexpected}')
        self.embed_dim = EMBED_DIM

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Group the stem, stage embeddings and blocks, folding readout parameters into the deepest block."""
        model = self.model
        layers = [tuple(model.patch_embed3D0.parameters())]
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


def unimiss(
    *, dims: int, device: str, weights: str, img_size: int, trainable: bool = False,
    isotropic: bool = True, drop_path_rate: float = 0.0,
):
    """Construct the UniMiSS (ECCV) 3D encoder on `device` (released student weights, atomic).

    `weights` is required (path to the GDrive-released UniMiss_small.pth). `img_size` is required
    and gives an isotropic cube: it sets the pyramid's stage grids, so the caller states the
    resolution rather than inheriting the anisotropic pretraining crop. `isotropic=False` keeps
    the pretrained in-plane-only stem stride (archival; the /16 stage is then not a cube).
    """
    if dims != 3:
        raise ValueError('UniMiSS is a 3D-only baseline; pass dims=3')
    enc = UniMissEncoder(
        weights=weights, img_size=img_size, isotropic=isotropic, drop_path_rate=drop_path_rate,
    ).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc
