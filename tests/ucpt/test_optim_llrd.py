"""UCPT layer learning rates, weight decay, and schedule ratios."""

import pytest
import torch
from torch import nn

from pumit.model.vit import ViT, ViTConfig
from pumit.ucpt.model import UCPTModel
from pumit.ucpt.train.config import UCPTOptimConfig, load_train_config
from pumit.ucpt.train.optim import build_param_groups, make_optimizer_scheduler


def _model():
    vit_config = ViTConfig(
        hidden_size=32, num_hidden_layers=24, num_attention_heads=2,
        intermediate_size=64, patch_size=2, num_register_tokens=1, drop_path_rate=0.1,
    )
    vit = ViT(vit_config)
    seg = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32).requires_grad_(False))
    return UCPTModel(
        vit=vit, recon_decoder=nn.Linear(32, 32), patch_distill_decoder=nn.Linear(32, 32),
        cls_predictor=nn.Linear(32, 32), seg=seg,
    )


def test_llrd_groups_cover_trainable_parameters_and_preserve_decay_policy():
    model = _model()
    cfg = UCPTOptimConfig(lr=1e-4, layer_decay=0.95, ssl_lr_fraction=0.7, seg_lr_fraction=0.3)
    groups = build_param_groups(model, cfg)
    names = [name for group in groups for name in group['params_names']]
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(names) == trainable
    assert len(names) == len(set(names))
    param_ids = [id(p) for group in groups for p in group['params']]
    assert len(param_ids) == len(set(param_ids))
    by_name = {name: group for group in groups for name in group['params_names']}
    for name in trainable:
        expected_decay = 0 if name in model.no_weight_decay() else cfg.weight_decay
        assert by_name[name]['weight_decay'] == expected_decay
        if name.startswith('vit.embeddings.'):
            expected_lr = 1e-4 * 0.95 ** 24
        elif name.startswith('vit.layer.'):
            block_index = int(name.split('.')[2])
            expected_lr = 1e-4 * 0.95 ** (23 - block_index)
        elif name.startswith('vit.norm.'):
            expected_lr = 1e-4
        elif name.startswith('seg.'):
            expected_lr = 3e-5
        else:
            expected_lr = 7e-5
        assert by_name[name]['lr'] == pytest.approx(expected_lr)
    assert model.vit.patch_embed.weight.requires_grad
    model.train()
    assert model.vit.training
    assert not model.teacher_vit.training
    assert not model.ema_seg.training


def test_llrd_ratios_survive_warmup_plateau_cooldown_and_optimizer_step(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model = _model()
    cfg = UCPTOptimConfig(
        lr=1e-4, layer_decay=0.95, ssl_lr_fraction=0.7, seg_lr_fraction=0.3,
        warmup_steps=2, cooldown_steps=2, steps=6,
    )
    groups = build_param_groups(model, cfg)
    peaks = [group['lr'] for group in groups]
    optimizer, scheduler = make_optimizer_scheduler(cfg, groups)
    initial_patch = model.vit.patch_embed.weight.detach().clone()
    for step in range(cfg.steps + 1):
        ratios = [group['lr'] / peak for group, peak in zip(optimizer.param_groups, peaks)]
        assert ratios == pytest.approx([ratios[0]] * len(ratios))
        if step in (0, cfg.steps):
            assert ratios[0] == pytest.approx(0.01)
        if step == cfg.steps:
            break
        optimizer.zero_grad()
        patches = torch.randn(2, 3, 2, 2, 2)
        model.vit.embeddings(patches).square().mean().backward()
        optimizer.step()
        scheduler.step()
    assert not torch.equal(initial_patch, model.vit.patch_embed.weight)


def test_main_config_enables_confirmed_recipe():
    cfg = load_train_config('configs/ucpt/train/main.yaml')
    assert cfg.optim.lr == 1e-4
    assert cfg.optim.layer_decay == 0.95
    assert cfg.model.drop_path_rate == 0.1
