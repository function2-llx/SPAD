"""Optimizer parameter-group construction for cls finetuning, including LLRD.

LLRD (layer-wise learning-rate decay) mirrors the downstream/seg contract: the from-scratch
head trains at `lr_head`, while backbone layer `i` of `n` trains at
`lr_encoder * layer_decay ** (n - 1 - i)` -- the top block keeps the full encoder LR and
shallower (more general) layers decay geometrically. `layer_decay == 1` reproduces the original
two-group behavior exactly, so existing results stay comparable.

The layer partition comes from the encoder's own `parameter_layers()` (shallow-to-deep), which
must cover every trainable backbone parameter exactly once; a mismatch raises rather than
silently dropping parameters from the optimizer.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from torch import nn

from pumit.types import NoWeightDecayParameter

# Name parts that conventionally skip weight decay in ViT finetuning (mirrors
# downstream/seg's vit_standard policy).
_NO_WEIGHT_DECAY_NAME_PARTS = (
    'cls_token',
    'mask_token',
    'pos_embed',
    'position_embed',
    'register_token',
    'rel_pos',
    'relative_position',
    'relative_position_bias',
)


def make_parameter_layers(
    embedding_parameters: Iterable[nn.Parameter],
    blocks: Sequence[nn.Module],
    top_parameters: Iterable[nn.Parameter],
) -> tuple[tuple[nn.Parameter, ...], ...]:
    """Return ViT parameters as shallow-to-deep architecture layers.

    Layer 0 is the patch/position embedding; each Transformer block is its own layer; the final
    norm (and any pretrained head-side parameters) fold into the deepest block so they train at
    the undecayed encoder LR. Mirrors `pumit.downstream.seg.adapters.vit.make_parameter_layers`.
    """
    groups = [tuple(embedding_parameters)]
    groups.extend(tuple(block.parameters()) for block in blocks)
    if not groups or not groups[-1]:
        raise RuntimeError('parameter layers require at least one non-empty Transformer block')
    groups[-1] = (*groups[-1], *tuple(top_parameters))
    return tuple(groups)


def _trainable_layer_ids(encoder: nn.Module) -> tuple[set[int], ...]:
    """Validate and return the encoder's trainable parameter ids grouped shallow-to-deep."""
    trainable = {id(p) for p in encoder.parameters() if p.requires_grad}
    grouping = getattr(encoder, 'parameter_layers', None)
    if not callable(grouping):
        raise TypeError(
            f'{type(encoder).__name__} must define parameter_layers() to use layer_decay < 1'
        )
    layers = tuple(
        tuple(id(p) for p in params if p.requires_grad)
        for params in grouping()
    )
    if not layers or any(not layer for layer in layers):
        raise RuntimeError('encoder parameter layers must be non-empty')
    flat = [pid for layer in layers for pid in layer]
    if len(flat) != len(set(flat)):
        raise RuntimeError('encoder parameter layers contain duplicates')
    if set(flat) != trainable:
        missing = trainable - set(flat)
        unexpected = set(flat) - trainable
        raise RuntimeError(
            f'encoder parameter layers do not cover trainable parameters: '
            f'missing={len(missing)}, unexpected={len(unexpected)}'
        )
    return tuple(set(layer) for layer in layers)


def is_no_decay(name: str, parameter: nn.Parameter) -> bool:
    """Whether a parameter should skip weight decay under the `vit_standard` policy.

    True for the explicit `NoWeightDecayParameter` marker (cls/register tokens), for any
    1-D-or-scalar tensor (norm weights, biases, layer-scale gammas), and for conventional
    position/token names. Mirrors downstream/seg and UCPT pretraining, which both exclude these.
    """
    if isinstance(parameter, NoWeightDecayParameter) or parameter.ndim <= 1:
        return True
    lowered = name.lower()
    return any(part in lowered for part in _NO_WEIGHT_DECAY_NAME_PARTS)


def build_param_groups(
    encoder: nn.Module,
    head: nn.Module,
    *,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    layer_decay: float = 1.0,
    weight_decay_policy: str = 'all',
) -> list[dict]:
    """Build AdamW parameter groups, optionally with layer-wise LR decay.

    Args:
        encoder: the backbone; must expose `parameter_layers()` when `layer_decay < 1`.
        head: the from-scratch classification head (the `scratch` scope, at `lr_head`).
        lr_encoder: base encoder LR, applied undecayed to the deepest backbone layer.
        lr_head: LR for the from-scratch head.
        weight_decay: applied to every decaying group.
        layer_decay: geometric per-layer factor in (0, 1]; 1.0 disables LLRD.
        weight_decay_policy: `all` decays every parameter (the original cls behavior);
            `vit_standard` additionally splits out a no-decay group for norms, biases,
            layer-scale gammas and cls/register/position tokens.

    Returns:
        AdamW-ready group dicts, each with `name`, `params`, `lr`, `weight_decay`.
    """
    if not isinstance(layer_decay, int | float) or not 0 < layer_decay <= 1:
        raise ValueError(f'layer_decay must be in (0, 1], got {layer_decay}')
    if weight_decay_policy not in {'all', 'vit_standard'}:
        raise ValueError(
            f"weight_decay_policy must be 'all' or 'vit_standard', got {weight_decay_policy!r}"
        )

    split_decay = weight_decay_policy == 'vit_standard'
    head_named = [(n, p) for n, p in head.named_parameters() if p.requires_grad]
    if not head_named:
        raise RuntimeError('the classification head has no trainable parameters')

    def emit(scope: str, lr: float, named: list[tuple[str, nn.Parameter]]) -> list[dict]:
        """Turn one scope's parameters into 1 group (`all`) or 2 (`vit_standard`)."""
        if not split_decay:
            return [{'name': scope, 'params': [p for _, p in named],
                     'lr': lr, 'weight_decay': weight_decay}] if named else []
        decayed = [p for n, p in named if not is_no_decay(n, p)]
        skipped = [p for n, p in named if is_no_decay(n, p)]
        groups = []
        if decayed:
            groups.append({'name': f'{scope}_decay', 'params': decayed,
                           'lr': lr, 'weight_decay': weight_decay})
        if skipped:
            groups.append({'name': f'{scope}_no_decay', 'params': skipped,
                           'lr': lr, 'weight_decay': 0.0})
        return groups

    if layer_decay == 1:
        encoder_named = [(n, p) for n, p in encoder.named_parameters() if p.requires_grad]
        return [*emit('encoder', lr_encoder, encoder_named), *emit('scratch', lr_head, head_named)]

    layers = _trainable_layer_ids(encoder)
    n_layers = len(layers)
    layer_of = {pid: index for index, layer in enumerate(layers) for pid in layer}
    buckets: dict[int, list[tuple[str, nn.Parameter]]] = {i: [] for i in range(n_layers)}
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            buckets[layer_of[id(parameter)]].append((name, parameter))

    groups: list[dict] = []
    for index, named in buckets.items():
        lr = lr_encoder * layer_decay ** (n_layers - 1 - index)
        groups.extend(emit(f'backbone_layer_{index:02d}', lr, named))
    groups.extend(emit('scratch', lr_head, head_named))
    return groups


def warmup_steps_for(total_steps: int, fraction: float = 0.05) -> int:
    """Warmup length as a fraction of the run: 5% of 330 steps -> 16.

    Returns at least 1 step for any positive fraction, so a short run still ramps rather than
    starting at full LR; a fraction of 0 disables warmup entirely.
    """
    if not 0 <= fraction < 1:
        raise ValueError(f'warmup fraction must be in [0, 1), got {fraction}')
    if fraction == 0:
        return 0
    return max(1, int(round(total_steps * fraction)))


def build_warmup_cosine(optimizer, *, total_steps: int, warmup_steps: int):
    """Linear warmup into cosine annealing, preserving each group's own base LR.

    Both phases are multiplicative on the per-group base LR, so the encoder/head ratio -- and any
    LLRD per-layer ratios -- hold at every step. Cosine spans the post-warmup remainder, ending at
    ~0. Without warmup the full LR lands while the randomly initialized head is still untrained,
    which is what drove the observed peak-at-epoch-1-then-decline behavior.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(f'warmup_steps must be in [0, {total_steps}), got {warmup_steps}')
    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )
