"""Check pyramid finetune layer groups and native stochastic-depth schedules without checkpoints."""

import importlib

import pytest
import torch
from torch import nn

from pumit.downstream.cls.backbones import unet3d
from pumit.downstream.cls.optim import _trainable_layer_ids


@pytest.mark.parametrize('name', ['unimiss', 'unimiss_plus'])
@pytest.mark.parametrize('drop_path_rate', [0.0, 0.1])
def test_mit_recipe(monkeypatch, name, drop_path_rate):
    adapter = importlib.import_module(f'pumit.downstream.cls.backbones.{name}')
    plus = name == 'unimiss_plus'
    if plus:
        from pumit.downstream.cls.backbones._unimiss_plus.mitplus import MiT_encoder

        model_type = MiT_encoder
        depths = [1, 2, 4, 2]
        dimensions = [4, 8, 8, 8, 8, 8]
    else:
        from pumit.downstream.cls.backbones._unimiss.MiT import VisionTransformer

        model_type = VisionTransformer
        depths = [2, 3, 4, 3]
        dimensions = [8, 8, 8, 8]
    models = []

    def build_tiny(**kwargs):
        assert kwargs.pop('pretrain') is False
        kwargs['img_size3D'] = [16, 32, 32]
        model = model_type(
            **kwargs, embed_dims=dimensions, depths=depths,
            num_heads=[1, 1, 1, 1], mlp_ratios=[1, 1, 1, 1], sr_ratios=[1, 1, 1, 1],
        )
        models.append(model)
        return model

    monkeypatch.setattr(adapter, 'model_plus' if plus else 'model_small', build_tiny)
    monkeypatch.setattr(adapter, '_load_encoder_state_dict', lambda _: models[-1].state_dict())
    encoder = getattr(adapter, name)(
        dims=3, device='cpu', weights='unused', img_size=32,
        trainable=True, drop_path_rate=drop_path_rate,
    )
    model = encoder.model
    blocks = [block for stage in range(1, 5) for block in getattr(model, f'block{stage}')]
    rates = [getattr(block.drop_path, 'drop_prob', 0.0) for block in blocks]
    assert rates == pytest.approx(torch.linspace(0, drop_path_rate, sum(depths)).tolist())

    layers = _trainable_layer_ids(encoder)
    stems = [model.ConvBlock3D0, model.ConvBlock3D1] if plus else [model.patch_embed3D0]
    assert len(layers) == len(stems) + 4 + sum(depths)
    for index, stem in enumerate(stems):
        assert layers[index] == {id(parameter) for parameter in stem.parameters()}
    index = len(stems)
    for stage in range(1, 5):
        cls_tokens = getattr(model, f'cls_tokens{stage}')
        cls_parameters = (cls_tokens,) if stage == 1 else cls_tokens.parameters()
        embedding_parameters = (
            *getattr(model, f'patch_embed3D{stage}').parameters(),
            getattr(model, f'pos_embed3D{stage}'), *cls_parameters,
        )
        assert layers[index] == {id(parameter) for parameter in embedding_parameters}
        index += 1
        for block in getattr(model, f'block{stage}'):
            expected = {id(parameter) for parameter in block.parameters()}
            if index == len(layers) - 1:
                expected.update(id(parameter) for parameter in model.norm_new.parameters())
                expected.update(id(parameter) for parameter in model.head_new.parameters())
            assert layers[index] == expected
            index += 1
    if plus:
        assert isinstance(model.ConvBlock3D0, adapter._Recompute)


def test_genesis_four_stage_layers(monkeypatch):
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            for channels in (64, 128, 256, 512):
                setattr(self, f'down_tr{channels}', nn.Linear(2, 2))

    monkeypatch.setattr(unet3d, 'UNetEncoder', TinyUNet)
    monkeypatch.setattr(unet3d, 'load_encoder_state_dict', lambda *_: TinyUNet().state_dict())
    encoder = unet3d.models_genesis(
        dims=3, device='cpu', weights='unused', img_size=96,
        trainable=True, drop_path_rate=0.0,
    )
    layers = _trainable_layer_ids(encoder)
    assert len(layers) == 4
    for layer, channels in zip(layers, (64, 128, 256, 512)):
        assert layer == {
            id(parameter) for parameter in getattr(encoder.unet, f'down_tr{channels}').parameters()
        }
    with pytest.raises(ValueError, match='drop_path_rate must be zero'):
        unet3d.models_genesis(
            dims=3, device='cpu', weights='unused', img_size=96, drop_path_rate=0.1,
        )
