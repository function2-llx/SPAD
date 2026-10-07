"""VoCo-L's pretrained global embedding for MedMNIST3D classification.

The official pretraining backbone concatenates GAP over encoder1(x), encoder2(h0), encoder3(h1), encoder4(h2), and encoder10(h4), giving 2304 features at width 96.
All five convolutional branches and the Swin transformer are loaded from the released checkpoint; no downstream decoder or SSL projection head is used.
The shared Swin implementation preserves the MONAI 1.3.0 patch-merging order used during pretraining.
Patch tokens retain the /32 Swin grid for the downstream token-count contract, independently of the concatenated global embedding.
Input is single-channel uint8 scaled to [0,1].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pumit.downstream.seg.backbones._voco_swin import SwinTransformer, UnetrBasicBlock

FEATURE_SIZE_L = 96
PATCH = 2


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} in [0,1] at native res (CPU). Shape-blind."""
    if not is_3d:
        raise ValueError('VoCo is a 3D-only baseline; cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0
    return {'x': x.unsqueeze(1)}


def build_swin(
    feature_size: int, *, drop_path_rate: float = 0.0, use_checkpoint: bool = False,
    attention_chunk_size: int | None = None,
) -> SwinTransformer:
    """The official VoCo downstream topology (model.py of Luffy03/Large-Scale-Medical)."""
    return SwinTransformer(
        in_chans=1,
        embed_dim=feature_size,
        window_size=(7, 7, 7),
        patch_size=(PATCH, PATCH, PATCH),
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        attention_chunk_size=attention_chunk_size,
    )


class VoCoEncoder(nn.Module):
    """Return the pretrained five-branch global embedding and /32 Swin patch tokens."""

    def __init__(
        self, weights: str, img_size: int, feature_size: int = FEATURE_SIZE_L,
        *, drop_path_rate: float = 0.0, use_checkpoint: bool = False,
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        if img_size % 32:
            raise ValueError(f'img_size {img_size} not divisible by 32 (the readout stride)')
        self.img_size = img_size
        self.use_checkpoint = use_checkpoint
        self.swin = build_swin(
            feature_size, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
            attention_chunk_size=attention_chunk_size,
        )
        self.encoder1 = UnetrBasicBlock(1, feature_size)
        self.encoder2 = UnetrBasicBlock(feature_size, feature_size)
        self.encoder3 = UnetrBasicBlock(2 * feature_size, 2 * feature_size)
        self.encoder4 = UnetrBasicBlock(4 * feature_size, 4 * feature_size)
        self.encoder10 = UnetrBasicBlock(16 * feature_size, 16 * feature_size)
        self.embed_dim = 24 * feature_size
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        self.load_state_dict(
            {
                'swin.' + key.removeprefix('swinViT.') if key.startswith('swinViT.') else key: value
                for key, value in state.items()
            },
            strict=True,
        )

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Match the segmentation encoder's shallow-to-deep pretrained parameter groups."""
        vit = self.swin
        return (
            tuple(self.encoder1.parameters()),
            tuple(vit.patch_embed.parameters()),
            (*vit.layers1c.parameters(), *vit.layers1.parameters(), *self.encoder2.parameters()),
            (*vit.layers2c.parameters(), *vit.layers2.parameters(), *self.encoder3.parameters()),
            (*vit.layers3c.parameters(), *vit.layers3.parameters(), *self.encoder4.parameters()),
            (*vit.layers4c.parameters(), *vit.layers4.parameters(), *self.encoder10.parameters()),
        )

    @staticmethod
    def _pool_encoder(module: nn.Module, x: Tensor) -> Tensor:
        return module(x).mean(dim=(2, 3, 4))

    def _readout(self, module: nn.Module, x: Tensor) -> Tensor:
        # Include pooling in recomputation so full-resolution branch outputs are not saved.
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._pool_encoder, module, x, use_reentrant=False)
        return self._pool_encoder(module, x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.interpolate(x, size=(self.img_size,) * 3, mode='trilinear', align_corners=False)
        hidden = self.swin(x)
        global_features = torch.cat(
            (
                self._readout(self.encoder1, x),
                self._readout(self.encoder2, hidden[0]),
                self._readout(self.encoder3, hidden[1]),
                self._readout(self.encoder4, hidden[2]),
                self._readout(self.encoder10, hidden[4]),
            ),
            dim=1,
        )
        patch_tokens = hidden[4].flatten(2).transpose(1, 2)
        return global_features, patch_tokens


def voco_l(
    *, dims: int, device: str, weights: str, img_size: int, trainable: bool = False,
    drop_path_rate: float = 0.0,
    attention_chunk_size: int | None = 2048,
):
    """Construct the VoCo-L encoder on `device` (weights from the released VoCo_L_SSL_head.pt).

    `weights` and `img_size` are required: the checkpoint is a local file, and img_size sets the token grid (192 -> 12^3 at /16, patch tokens 6^3 at /32).
    3D-only: dims must be 3.
    Finetuning recomputes Swin blocks, window attention, and pooled convolutional branches to reduce activation memory.
    """
    if dims != 3:
        raise ValueError('VoCo is a 3D-only baseline; pass dims=3')
    enc = VoCoEncoder(
        weights=weights, img_size=img_size,
        drop_path_rate=drop_path_rate, use_checkpoint=trainable,
        attention_chunk_size=attention_chunk_size if trainable else None,
    ).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc
