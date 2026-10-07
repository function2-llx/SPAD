from pathlib import Path

from pumit.downstream.cls.config import load_dataset_specs, load_vit_config

CFG = Path('configs/downstream/cls')


def test_load_vit_config_matches_vitl():
    cfg = load_vit_config(CFG / 'vit_l.yaml')
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 16
    assert cfg.intermediate_size == 4096
    assert cfg.num_register_tokens == 4
    assert cfg.patch_size == 16
    assert cfg.grad_ckpt is False


def test_load_dataset_specs():
    specs = load_dataset_specs(CFG / 'datasets.yaml')
    assert specs['pathmnist'].size == 224
    assert specs['pathmnist'].is_3d is False
    assert specs['organmnist3d'].size == 64
    assert specs['organmnist3d'].is_3d is True
    assert specs['pneumoniamnist'].n_classes == 2
