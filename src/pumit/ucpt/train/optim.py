"""UCPT optimizer parameter grouping and learning-rate scheduling."""

import math

import torch

from pumit.ucpt.model import UCPTModel

from .config import UCPTOptimConfig


_SSL_PREFIXES = ('recon_decoder.', 'patch_distill_decoder.', 'cls_predictor.')


def build_param_groups(model: UCPTModel, cfg: UCPTOptimConfig) -> list[dict]:
    """Build optimizer groups by subsystem, ViT depth, and weight-decay policy.

    Segmentation and SSL prefixes are classified before the catch-all no-decay names.
    This keeps one-dimensional parameters in the corresponding subsystem groups.
    The ``requires_grad`` filter excludes EMA twins and frozen decoder parameters.
    Embeddings use ``layer_decay ** depth``; the last ViT block and final norm use the full LR.

    Args:
        model: UCPT model whose trainable parameters are grouped.
        cfg: Training and optimizer configuration.

    Returns:
        Non-empty groups, with LLRD applied only to the ViT and all groups split by weight-decay policy.
    """
    if not 0 < cfg.layer_decay <= 1:
        raise ValueError(f'layer_decay must be in (0, 1], got {cfg.layer_decay}')
    no_decay_names = model.no_weight_decay()
    vit_layer_ids = {}
    if cfg.layer_decay < 1:
        depth = len(model.vit.layer)
        vit_layer_ids.update((id(p), 0) for p in model.vit.embeddings.parameters())
        for layer_id, block in enumerate(model.vit.layer, start=1):
            vit_layer_ids.update((id(p), layer_id) for p in block.parameters())
        vit_layer_ids.update((id(p), depth) for p in model.vit.norm.parameters())
    groups: dict[str, dict] = {
        'vit': {
            'params': [], 'params_names': [], 'group_name': 'vit',
            'lr': cfg.lr, 'weight_decay': cfg.weight_decay,
        },
        'vit_no_decay': {
            'params': [], 'params_names': [], 'group_name': 'vit_no_decay',
            'lr': cfg.lr, 'weight_decay': 0.0,
        },
        'ssl': {
            'params': [], 'params_names': [], 'group_name': 'ssl',
            'lr': cfg.ssl_lr_fraction * cfg.lr, 'weight_decay': cfg.weight_decay,
        },
        'ssl_no_decay': {
            'params': [], 'params_names': [], 'group_name': 'ssl_no_decay',
            'lr': cfg.ssl_lr_fraction * cfg.lr, 'weight_decay': 0.0,
        },
        'seg': {
            'params': [], 'params_names': [], 'group_name': 'seg',
            'lr': cfg.seg_lr_fraction * cfg.lr, 'weight_decay': cfg.weight_decay,
        },
        'seg_no_decay': {
            'params': [], 'params_names': [], 'group_name': 'seg_no_decay',
            'lr': cfg.seg_lr_fraction * cfg.lr, 'weight_decay': 0.0,
        },
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('seg.') and name in no_decay_names:
            g = 'seg_no_decay'
        elif name.startswith('seg.'):
            g = 'seg'
        elif name.startswith(_SSL_PREFIXES) and name in no_decay_names:
            g = 'ssl_no_decay'
        elif name.startswith(_SSL_PREFIXES):
            g = 'ssl'
        elif name in no_decay_names:
            g = 'vit_no_decay'
        else:
            g = 'vit'
        if g.startswith('vit') and cfg.layer_decay < 1:
            layer_id = vit_layer_ids[id(param)]
            no_decay_suffix = '_no_decay' if name in no_decay_names else ''
            g = f'vit_layer_{layer_id}{no_decay_suffix}'
            if g not in groups:
                groups[g] = {
                    'params': [], 'params_names': [], 'group_name': g,
                    'lr': cfg.lr * cfg.layer_decay ** (depth - layer_id),
                    'weight_decay': 0.0 if no_decay_suffix else cfg.weight_decay,
                }
        groups[g]['params'].append(param)
        groups[g]['params_names'].append(name)
    return [g for g in groups.values() if g['params']]


def make_optimizer_scheduler(
    cfg: UCPTOptimConfig,
    param_groups: list[dict],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build AdamW with linear warmup, a constant phase, and cosine cooldown.

    Args:
        cfg: Training and optimizer configuration.
        param_groups: Parameter groups with subsystem-specific learning rates.

    Returns:
        The optimizer and its learning-rate scheduler.
    """
    from torch.optim.lr_scheduler import ConstantLR, LambdaLR, LinearLR, SequentialLR

    constant_steps = cfg.steps - cfg.warmup_steps - cfg.cooldown_steps
    if constant_steps < 0:
        raise ValueError(
            f'warmup_steps + cooldown_steps exceeds total steps: '
            f'{cfg.warmup_steps} + {cfg.cooldown_steps} > {cfg.steps}',
        )
    if not 0 <= cfg.min_lr <= cfg.lr:
        raise ValueError(f'min_lr must be in [0, lr], got min_lr={cfg.min_lr}, lr={cfg.lr}')

    # fused=True: single multi-tensor kernel for the whole step. The default (foreach)
    # dispatch falls back to _single_tensor_adam for groups containing the
    # NoWeightDecayParameter subclass (exact-type check); explicit fused bypasses that
    # heuristic and accepts any CUDA floating-point param. CPU (tests) keeps the default.
    optimizer = torch.optim.AdamW(
        param_groups, lr=cfg.lr, betas=cfg.betas,
        fused=torch.cuda.is_available() or None,
    )
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
        LinearLR(optimizer, start_factor=1e-2, total_iters=cfg.warmup_steps),
    ]
    milestones: list[int] = []
    if constant_steps > 0:
        schedulers.append(ConstantLR(optimizer, factor=1.0, total_iters=constant_steps))
        milestones.append(cfg.warmup_steps)
    if cfg.cooldown_steps > 0:
        min_factor = cfg.min_lr / cfg.lr

        def cosine_factor(step: int) -> float:
            progress = min(step / cfg.cooldown_steps, 1.0)
            return min_factor + (1 - min_factor) * (1 + math.cos(math.pi * progress)) / 2

        schedulers.append(LambdaLR(optimizer, lr_lambda=cosine_factor))
        milestones.append(cfg.warmup_steps + constant_steps)

    scheduler = (
        schedulers[0]
        if len(schedulers) == 1
        else SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    )
    return optimizer, scheduler
