"""VoCo finetune recipe propagation without the full-size pretrained model."""

import pytest
import torch
from torch import nn

from pumit.downstream.cls.backbones import voco
from pumit.downstream.cls.optim import build_param_groups
from pumit.downstream.seg.backbones._voco_swin import UnetrBasicBlock


def swin_blocks(swin):
    return [
        block
        for stage in (swin.layers1, swin.layers2, swin.layers3, swin.layers4)
        for layer in stage
        for block in layer.blocks
    ]


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / 'encoder.pt'
    swin = voco.build_swin(feature_size=6)
    state = {f'swinViT.{key}': value for key, value in swin.state_dict().items()}
    for name, in_channels, out_channels in (
        ('encoder1', 1, 6),
        ('encoder2', 6, 6),
        ('encoder3', 12, 12),
        ('encoder4', 24, 24),
        ('encoder10', 96, 96),
    ):
        block = UnetrBasicBlock(in_channels, out_channels)
        state.update({f'{name}.{key}': value for key, value in block.state_dict().items()})
    torch.save(state, path)
    return str(path)


def test_defaults_preserve_frozen_recipe(weights):
    encoder = voco.VoCoEncoder(weights, img_size=64, feature_size=6)
    for block in swin_blocks(encoder.swin):
        assert getattr(block.drop_path, 'drop_prob', 0.0) == 0.0
        assert block.use_checkpoint is False


def test_native_linear_drop_path_and_checkpointing(weights):
    encoder = voco.VoCoEncoder(
        weights, img_size=64, feature_size=6, drop_path_rate=0.1, use_checkpoint=True,
    )
    blocks = swin_blocks(encoder.swin)
    rates = [getattr(block.drop_path, 'drop_prob', 0.0) for block in blocks]
    assert rates == pytest.approx(torch.linspace(0, 0.1, 8).tolist())
    assert all(block.use_checkpoint for block in blocks)


def test_llrd_groups_cover_encoder_once(weights):
    encoder = voco.VoCoEncoder(weights, img_size=64, feature_size=6)
    layers = encoder.parameter_layers()
    assert len(layers) == 6
    flat = [id(parameter) for layer in layers for parameter in layer]
    assert len(flat) == len(set(flat))
    assert set(flat) == {id(parameter) for parameter in encoder.parameters()}
    assert {id(parameter) for parameter in layers[0]} == {
        id(parameter) for parameter in encoder.encoder1.parameters()
    }
    assert {id(parameter) for parameter in layers[1]} == {
        id(parameter) for parameter in encoder.swin.patch_embed.parameters()
    }
    for depth, branch in enumerate(['encoder2', 'encoder3', 'encoder4', 'encoder10'], start=2):
        stage = depth - 1
        expected_parameters = (
            *getattr(encoder.swin, f'layers{stage}c').parameters(),
            *getattr(encoder.swin, f'layers{stage}').parameters(),
            *getattr(encoder, branch).parameters(),
        )
        assert {id(parameter) for parameter in layers[depth]} == {
            id(parameter) for parameter in expected_parameters
        }
    head = nn.Linear(encoder.embed_dim, 2)
    groups = build_param_groups(
        encoder, head, lr_encoder=2e-5, lr_head=1e-3,
        weight_decay=0.01, layer_decay=0.9, weight_decay_policy='vit_standard',
    )
    rates_by_id = {
        id(parameter): group['lr'] for group in groups for parameter in group['params']
    }
    for depth, parameters in enumerate(layers):
        expected = 2e-5 * 0.9 ** (5 - depth)
        assert all(rates_by_id[id(parameter)] == pytest.approx(expected) for parameter in parameters)


def test_checkpointing_preserves_embedding_and_all_parameter_gradients(weights):
    plain = voco.VoCoEncoder(weights, img_size=64, feature_size=6)
    checkpointed = voco.VoCoEncoder(
        weights, img_size=64, feature_size=6, use_checkpoint=True, attention_chunk_size=16,
    )
    plain.train()
    checkpointed.train()
    x_plain = torch.rand(1, 1, 28, 28, 28, requires_grad=True)
    x_checkpointed = x_plain.detach().clone().requires_grad_()
    features_plain, tokens_plain = plain(x_plain)
    features_checkpointed, tokens_checkpointed = checkpointed(x_checkpointed)
    torch.testing.assert_close(features_plain, features_checkpointed)
    torch.testing.assert_close(tokens_plain, tokens_checkpointed)
    coefficient = torch.linspace(0.1, 1, plain.embed_dim)[None]
    (features_plain * coefficient).sum().backward()
    (features_checkpointed * coefficient).sum().backward()
    torch.testing.assert_close(x_plain.grad, x_checkpointed.grad, rtol=1e-4, atol=1e-5)
    actual_parameters = dict(checkpointed.named_parameters())
    for name, parameter in plain.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        actual = actual_parameters[name]
        assert actual.grad is not None, name
        torch.testing.assert_close(parameter.grad, actual.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('trainable', [False, True])
def test_factory_passes_finetune_options(monkeypatch, weights, trainable):
    original = voco.VoCoEncoder

    def tiny_encoder(**kwargs):
        return original(feature_size=6, **kwargs)

    monkeypatch.setattr(voco, 'VoCoEncoder', tiny_encoder)
    encoder = voco.voco_l(
        dims=3, device='cpu', weights=weights, img_size=64,
        trainable=trainable, drop_path_rate=0.1,
    )
    assert encoder.training is trainable
    assert all(parameter.requires_grad is trainable for parameter in encoder.parameters())
    blocks = swin_blocks(encoder.swin)
    assert blocks[-1].drop_path.drop_prob == pytest.approx(0.1)
    assert all(block.use_checkpoint is trainable for block in blocks)
    assert all(
        block.attn.attention_chunk_size == (2048 if trainable else None)
        for block in blocks
    )
