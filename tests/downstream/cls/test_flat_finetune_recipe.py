"""Hub-free flat-backbone tests for the shared finetuning recipe."""

import json
from types import SimpleNamespace

import pytest
import timm
import torch
from torch import nn
from open_clip.timm_model import TimmModel

from pumit.downstream.cls.backbones import biomedclip, eva02, sam_med3d, three_dino
from pumit.downstream.cls.backbones._3dino.models.vision_transformer import DinoVisionTransformer3d
from pumit.downstream.cls.backbones._sam_med3d_encoder import ImageEncoderViT3D
from pumit.downstream.cls.optim import build_param_groups


def _assert_layers(encoder, blocks, embedding, top):
    layers = encoder.parameter_layers()
    assert len(layers) == len(blocks) + 1
    flat = [id(parameter) for layer in layers for parameter in layer]
    assert len(flat) == len(set(flat))
    assert set(flat) == {id(parameter) for parameter in encoder.parameters()}
    assert id(embedding) in {id(parameter) for parameter in layers[0]}
    assert id(top) in {id(parameter) for parameter in layers[-1]}
    groups = build_param_groups(
        encoder, nn.Linear(encoder.embed_dim, 2),
        lr_encoder=2e-5, lr_head=1e-3, weight_decay=.01,
        layer_decay=.9, weight_decay_policy='vit_standard',
    )
    lr_of = {id(parameter): group['lr'] for group in groups for parameter in group['params']}
    assert lr_of[id(embedding)] == pytest.approx(2e-5 * .9 ** len(blocks))
    assert lr_of[id(top)] == 2e-5


def _assert_drop_path(blocks, maximum):
    expected = torch.linspace(0, maximum, len(blocks)).tolist()
    for block, probability in zip(blocks, expected):
        assert getattr(block.drop_path1, 'drop_prob', 0.0) == pytest.approx(probability)
        assert getattr(block.drop_path2, 'drop_prob', 0.0) == pytest.approx(probability)


@pytest.mark.parametrize('maximum', [0.0, 0.1])
def test_eva_native_schedule_and_layer_coverage(monkeypatch, maximum):
    monkeypatch.setattr(eva02, 'EVA_MODELS', {'large': 'eva02_tiny_patch14_224'})
    real_create = timm.create_model
    monkeypatch.setattr(
        eva02.timm, 'create_model',
        lambda name, pretrained, **kwargs: real_create(name, pretrained=False, **kwargs),
    )
    encoder = eva02.eva02_large(
        dims=3, device='cpu', img_size=28, trainable=True, drop_path_rate=maximum,
    )
    model = encoder.model
    _assert_layers(encoder, model.blocks, model.pos_embed, model.fc_norm.weight)
    _assert_drop_path(model.blocks, maximum)


@pytest.mark.parametrize('maximum', [0.0, 0.1])
def test_biomed_native_schedule_projection_and_layer_coverage(monkeypatch, tmp_path, maximum):
    vision_cfg = {'timm_model_name': 'vit_tiny_patch16_224', 'timm_pool': 'token'}
    (tmp_path / 'open_clip_config.json').write_text(
        json.dumps({'model_cfg': {'vision_cfg': vision_cfg}}),
    )
    monkeypatch.setattr(biomedclip, 'snapshot_download', lambda *args, **kwargs: str(tmp_path))

    def create_visual(name, *, vision_cfg):
        assert name == f'local-dir:{tmp_path}'
        assert vision_cfg['timm_pool'] == 'token'
        visual = TimmModel(
            vision_cfg['timm_model_name'], embed_dim=8, image_size=224,
            pool=vision_cfg['timm_pool'], proj='linear', proj_bias=False,
            drop_path=vision_cfg['timm_drop_path'], pretrained=False,
        )
        return SimpleNamespace(visual=visual), None

    monkeypatch.setattr(biomedclip.open_clip, 'create_model_from_pretrained', create_visual)
    encoder = biomedclip.biomedclip(
        dims=3, device='cpu', img_size=32, trainable=True, drop_path_rate=maximum,
    )
    trunk = encoder.visual.trunk
    _assert_layers(encoder, trunk.blocks, trunk.pos_embed, encoder.visual.head[1].weight)
    _assert_drop_path(trunk.blocks, maximum)


def test_biomed_real_open_clip_config_override(monkeypatch, tmp_path):
    model_cfg = {
        'embed_dim': 8,
        'vision_cfg': {
            'timm_model_name': 'vit_tiny_patch16_224',
            'timm_pool': 'token', 'timm_proj': 'linear', 'timm_proj_bias': False,
        },
        'text_cfg': {'context_length': 8, 'vocab_size': 32, 'width': 8, 'heads': 2, 'layers': 1},
    }
    model = biomedclip.open_clip.CLIP(**model_cfg)
    torch.save(model.state_dict(), tmp_path / 'open_clip_pytorch_model.bin')
    (tmp_path / 'open_clip_config.json').write_text(json.dumps({'model_cfg': model_cfg}))
    monkeypatch.setattr(biomedclip, 'snapshot_download', lambda *args, **kwargs: str(tmp_path))
    visual = biomedclip._load_visual(drop_path_rate=.1)
    _assert_drop_path(visual.trunk.blocks, .1)
    for name, value in model.visual.state_dict().items():
        torch.testing.assert_close(visual.state_dict()[name], value)


@pytest.mark.parametrize('maximum', [0.0, 0.1])
@pytest.mark.parametrize('chunks', [0, 2])
def test_three_dino_native_schedule_and_layer_coverage(monkeypatch, maximum, chunks):
    def build(*, drop_path_rate=0.0):
        return DinoVisionTransformer3d(
            img_size=32, patch_size=16, embed_dim=24, depth=4,
            num_heads=3, block_chunks=chunks, drop_path_rate=drop_path_rate,
        )

    monkeypatch.setattr(three_dino, 'build_encoder', build)
    monkeypatch.setattr(three_dino, '_load_state_dict', lambda: build().state_dict())
    encoder = three_dino.three_dino(
        dims=3, device='cpu', img_size=32, trainable=True, drop_path_rate=maximum,
    )
    model = encoder.model
    blocks = (
        [block for chunk in model.blocks for block in chunk if not isinstance(block, nn.Identity)]
        if chunks else list(model.blocks)
    )
    _assert_layers(encoder, blocks, model.mask_token, model.norm.weight)
    _assert_drop_path(blocks, maximum)


def test_sam_neck_layer_coverage_and_no_drop_path(monkeypatch):
    def build(*, img_size=32):
        return ImageEncoderViT3D(
            img_size=img_size, patch_size=16, embed_dim=24, depth=2,
            num_heads=3, out_chans=8, in_chans=1, window_size=0,
        )

    monkeypatch.setattr(sam_med3d, 'build_encoder', build)
    monkeypatch.setattr(sam_med3d, '_load_state_dict', lambda: build().state_dict())
    encoder = sam_med3d.sam_med3d(
        dims=3, device='cpu', img_size=32, trainable=True, drop_path_rate=0.0,
    )
    model = encoder.encoder
    _assert_layers(encoder, model.blocks, model.pos_embed, model.neck[0].weight)
    assert all(not hasattr(block, 'drop_path') for block in model.blocks)
    with pytest.raises(ValueError, match='no native drop path'):
        sam_med3d.sam_med3d(dims=3, device='cpu', img_size=32, drop_path_rate=.1)
