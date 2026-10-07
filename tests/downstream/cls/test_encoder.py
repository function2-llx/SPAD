import pytest
import torch

from pumit.model.vit import ViT, ViTConfig
from pumit.downstream.cls.backbones.encoder import slice_teacher_vit


def test_slice_teacher_vit_strips_prefix():
    cfg = ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=128, num_register_tokens=1)
    vit = ViT(cfg)
    vit_sd = vit.state_dict()
    # Simulate a UCPT model state_dict: teacher_vit.* + vit.* + unrelated seg.* keys
    model_sd = {}
    for k, v in vit_sd.items():
        model_sd[f'teacher_vit.{k}'] = v
        model_sd[f'vit.{k}'] = v
    model_sd['seg.neck.weight'] = torch.zeros(3)
    model_sd['_orig_mod.teacher_vit.norm.bias'] = vit_sd['norm.bias']  # compile-prefixed dup

    sliced = slice_teacher_vit(model_sd)
    # every key present in a bare ViT, no prefix, no seg keys
    fresh = ViT(cfg)
    missing, unexpected = fresh.load_state_dict(sliced, strict=False)
    assert not unexpected, f'unexpected: {unexpected}'
    assert not missing, f'missing: {missing}'


def test_slice_teacher_vit_raises_on_empty():
    # No teacher_vit.* keys (only student + seg) must fail loud, not return {}.
    with pytest.raises(KeyError, match='teacher_vit'):
        slice_teacher_vit({'vit.norm.weight': torch.zeros(4),
                           'seg.neck.weight': torch.zeros(3)})
