"""Load SAM 3 detector weights into UCPT segmentation modules.

Loads ``detector_model.*`` weights into ``SPADNeck``, ``FusionEncoder``, and ``SemanticHead``, with optional text
projection initialization. SPAD layers inflate 2D weights automatically; plain ``nn.Conv3d`` layers are inflated here.

Source prefixes are ``detector_model.detr_encoder.layers.*``, ``detector_model.mask_decoder.*``, and
``detector_model.vision_encoder.neck.fpn_layers.*``.
"""

from safetensors.torch import load_file
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remap(sd: dict[str, torch.Tensor], src_prefix: str, dst_prefix: str) -> dict[str, torch.Tensor]:
    """Strip *src_prefix* from matching keys, prepend *dst_prefix*."""
    out: dict[str, torch.Tensor] = {}
    n = len(src_prefix)
    for k, v in sd.items():
        if k.startswith(src_prefix):
            out[dst_prefix + k[n:]] = v
    return out


def _inflate_conv2d_weight(tensor: torch.Tensor) -> torch.Tensor:
    """Inflate a 2D convolution weight along depth.

    Args:
        tensor: Source weight tensor.

    Returns:
        ``[O, I, 1, H, W]`` for a 4D source, otherwise the input unchanged.
    """
    if tensor.dim() == 4:
        return tensor.unsqueeze(2)
    return tensor


# Keys in the neck state dict that map to plain nn.Conv3d (kernel_size=1,
# returned by spadop.Conv3d factory) and need manual 2D->3D inflation.
# SPADConvTranspose3d_K2S2 and SPADConv3d_K3S1 auto-inflate via
# SPADWeightInflationMixin._load_from_state_dict.
_NECK_PLAIN_CONV3D_WEIGHT_KEYS: frozenset[str] = frozenset({
    'proj40_1.weight',
    'proj20_1.weight',
    'proj10_1.weight',
})


def _load_module(module: nn.Module, sub_sd: dict[str, torch.Tensor]) -> dict:
    """Load a remapped state dict with ``strict=False``.

    Args:
        module: Target module.
        sub_sd: Remapped source state dict.

    Returns:
        Matched count plus missing and unexpected keys.
    """
    result = module.load_state_dict(sub_sd, strict=False)
    matched = len(sub_sd) - len(result.unexpected_keys)
    return {
        'matched': matched,
        'missing': list(result.missing_keys),
        'unexpected': list(result.unexpected_keys),
    }


# ---------------------------------------------------------------------------
# Neck key mapping
# ---------------------------------------------------------------------------
#
# Mapping:
#   fpn_layers.0 (4x) -> up40 branch (up40_0/1 + proj40_1/2)
#   fpn_layers.1 (2x) -> up20 branch (up20_0 + proj20_1/2)
#   fpn_layers.2 (1x) -> proj10 branch (proj10_1/2)
#   fpn_layers.3 (1x) -> unmapped (SAM 3 has a 4th branch we don't use)
#
# Transposed-convolution biases are omitted because the SPAD targets have no bias.
# ---------------------------------------------------------------------------

_NECK_PREFIX = 'detector_model.vision_encoder.neck.'

NECK_KEY_MAP: dict[str, str] = {
    # --- fpn_layers.0 (scale_factor=4.0) -> up40 branch ---
    'fpn_layers.0.scale_layers.0.weight': 'up40_0.weight',
    'fpn_layers.0.scale_layers.2.weight': 'up40_1.weight',
    'fpn_layers.0.proj1.weight': 'proj40_1.weight',
    'fpn_layers.0.proj1.bias': 'proj40_1.bias',
    'fpn_layers.0.proj2.weight': 'proj40_2.weight',
    'fpn_layers.0.proj2.bias': 'proj40_2.bias',
    # --- fpn_layers.1 (scale_factor=2.0) -> up20 branch ---
    'fpn_layers.1.scale_layers.0.weight': 'up20_0.weight',
    'fpn_layers.1.proj1.weight': 'proj20_1.weight',
    'fpn_layers.1.proj1.bias': 'proj20_1.bias',
    'fpn_layers.1.proj2.weight': 'proj20_2.weight',
    'fpn_layers.1.proj2.bias': 'proj20_2.bias',
    # --- fpn_layers.2 (scale_factor=1.0) -> proj10 branch ---
    'fpn_layers.2.proj1.weight': 'proj10_1.weight',
    'fpn_layers.2.proj1.bias': 'proj10_1.bias',
    'fpn_layers.2.proj2.weight': 'proj10_2.weight',
    'fpn_layers.2.proj2.bias': 'proj10_2.bias',
}


def _build_neck_sd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build per-key remapped neck state dict from the full checkpoint."""
    neck_sd: dict[str, torch.Tensor] = {}
    for ckpt_suffix, our_key in NECK_KEY_MAP.items():
        full_key = _NECK_PREFIX + ckpt_suffix
        if full_key in sd:
            neck_sd[our_key] = sd[full_key]
    return neck_sd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sam3_weights(
    ckpt_path: str,
    *,
    neck: nn.Module,
    fusion: nn.Module,
    head: nn.Module,
    text_projection: nn.Module | None = None,
) -> dict:
    """Load SAM 3 detector weights into PUMIT seg modules.

    Args:
        ckpt_path: Path to ``model.safetensors``.
        neck: ``SPADNeck`` instance.
        fusion: ``FusionEncoder`` instance.
        head: ``SemanticHead`` instance.
        text_projection: Optional text projection to initialize from SAM 3.

    Returns:
        Total loaded tensor count, unmatched target keys, and per-module load details.
    """
    sd = load_file(ckpt_path)

    # ---- fusion encoder ----
    fusion_sd = _remap(sd, 'detector_model.detr_encoder.layers.', 'layers.')
    if not fusion_sd:
        raise RuntimeError(
            'Fusion encoder prefix remap produced empty dict. '
            'Expected prefix: detector_model.detr_encoder.layers.'
        )

    # ---- semantic head ----
    head_sd = _remap(sd, 'detector_model.mask_decoder.', '')
    if not head_sd:
        raise RuntimeError(
            'Head prefix remap produced empty dict. '
            'Expected prefix: detector_model.mask_decoder.'
        )
    # nn.Conv3d semantic_projection needs manual 2D->3D inflation.
    # SPAD convs (pixel_decoder.conv_layers) handle their own inflation.
    if 'semantic_projection.weight' in head_sd:
        head_sd['semantic_projection.weight'] = _inflate_conv2d_weight(
            head_sd['semantic_projection.weight']
        )

    # ---- neck ----
    neck_sd = _build_neck_sd(sd)
    if not neck_sd:
        raise RuntimeError(
            'Neck remap produced empty dict. '
            'Expected prefix: detector_model.vision_encoder.neck.'
        )
    # Inflate 1x1 conv weights going to plain nn.Conv3d (not SPAD subclasses).
    for key in _NECK_PLAIN_CONV3D_WEIGHT_KEYS:
        if key in neck_sd:
            neck_sd[key] = _inflate_conv2d_weight(neck_sd[key])

    # ---- load ----
    fusion_info = _load_module(fusion, fusion_sd)
    head_info = _load_module(head, head_sd)
    neck_info = _load_module(neck, neck_sd)
    text_projection_info = None
    if text_projection is not None:
        text_projection_info = _load_module(
            text_projection,
            {
                'weight': sd['detector_model.text_projection.weight'],
                'bias': sd['detector_model.text_projection.bias'],
            },
        )

    unmatched_target = (
        fusion_info['missing'] + head_info['missing'] + neck_info['missing']
    )
    loaded = fusion_info['matched'] + head_info['matched'] + neck_info['matched']
    detail = {
        'fusion': fusion_info,
        'head': head_info,
        'neck': neck_info,
    }
    if text_projection_info is not None:
        unmatched_target += text_projection_info['missing']
        loaded += text_projection_info['matched']
        detail['text_projection'] = text_projection_info

    return {
        'loaded': loaded,
        'unmatched_target': unmatched_target,
        'detail': detail,
    }
