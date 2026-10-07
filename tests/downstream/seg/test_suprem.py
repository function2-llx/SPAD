from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pumit.downstream.seg.suprem import SupremSegmentationNetwork
from tests.downstream.seg.conftest import architecture_kwargs


SUPREM_WEIGHTS = Path('pretrained/suprem/supervised_suprem_unet_2100.pth')


def _network(*, anisotropic=False, deep_supervision=True):
    strides = [(1, 1, 1), (1 if anisotropic else 2, 2, 2), (2, 2, 2), (2, 2, 2)]
    return SupremSegmentationNetwork(
        1, 3, backbone_name='suprem-unet', backbone_config={}, deep_supervision=deep_supervision,
        **architecture_kwargs([64, 128, 256, 512], strides),
    )


@pytest.mark.skipif(not SUPREM_WEIGHTS.is_file(), reason='released SuPreM checkpoint is unavailable')
@pytest.mark.parametrize('anisotropic', [False, True])
def test_full_pretrained_weights_and_scratch_heads(anisotropic):
    model = _network(anisotropic=anisotropic)
    heads_before = {key: value.clone() for key, value in model.decoder.seg_layers.state_dict().items()}
    model.load_pretrained(SUPREM_WEIGHTS)
    released = torch.load(SUPREM_WEIGHTS, map_location='cpu', weights_only=True)['net']
    for key, value in model.encoder.backbone.state_dict().items():
        torch.testing.assert_close(value, released[f'module.backbone.{key}'], rtol=0, atol=0)
    for key, value in model.decoder.state_dict().items():
        if key.startswith('seg_layers.'):
            torch.testing.assert_close(value, heads_before[key.removeprefix('seg_layers.')], rtol=0, atol=0)
            continue
        expected = released[f'module.backbone.{key}']
        if anisotropic and key == 'up_tr64.up_conv.weight':
            expected = expected.sum(dim=2, keepdim=True)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize('anisotropic', [False, True])
def test_deep_supervision_shapes_and_gradients(anisotropic):
    model = _network(anisotropic=anisotropic).train()
    x = torch.randn(2, 1, 8, 8, 8)
    predictions = model(x)
    depth_stride = 1 if anisotropic else 2
    assert [tuple(value.shape) for value in predictions] == [
        (2, 3, 8, 8, 8),
        (2, 3, 8 // depth_stride, 4, 4),
        (2, 3, 4 // depth_stride, 2, 2),
    ]
    sum(value.square().mean() for value in predictions).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    model.eval()
    model.decoder.deep_supervision = False
    with torch.no_grad():
        result = model(x)
    assert tuple(result.shape) == (2, 3, 8, 8, 8)


def test_all_norms_use_standard_sync_batchnorm_and_running_stats():
    model = _network().eval()
    norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    assert len(norms) == 14
    assert all(type(module) is nn.SyncBatchNorm for module in norms)
    for module in (model.encoder.backbone.down_tr64.ops[0].bn1, model.decoder.up_tr64.ops[0].bn1):
        x = torch.randn(2, module.num_features, 2, 2, 2)
        module.running_mean.fill_(2)
        module.running_var.fill_(3)
        before = module.running_mean.clone()
        expected = F.batch_norm(
            x, module.running_mean, module.running_var, module.weight, module.bias,
            training=False, eps=module.eps,
        )
        torch.testing.assert_close(module(x), expected)
        torch.testing.assert_close(module.running_mean, before)
    converted = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    assert all(
        type(module) is nn.SyncBatchNorm
        for module in converted.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )


def test_decoder_checkpoint_requires_all_pretrained_parameters(monkeypatch):
    model = _network()
    state = {
        f'module.backbone.{key}': value
        for key, value in model.decoder.state_dict().items()
        if key.startswith('up_tr')
    }
    del state['module.backbone.up_tr256.ops.0.conv1.weight']
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'net': state})
    with pytest.raises(RuntimeError, match='up_tr256.ops.0.conv1.weight'):
        model.load_pretrained_decoder(SUPREM_WEIGHTS)


@pytest.mark.parametrize('compiled', [False, True])
def test_optimizer_retains_all_output_heads(compiled):
    from types import SimpleNamespace
    from pumit.downstream.seg.trainer import DownstreamSegTrainer

    model = _network()
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = model
    trainer.num_epochs = 200
    trainer.enable_deep_supervision = True
    trainer._do_i_compile = lambda: compiled
    trainer.configuration_manager = SimpleNamespace(
        configuration={'optimization': {
            'optimizer': 'adamw', 'learning_rate': 2e-4,
            'backbone_lr': 8e-5, 'layer_decay': 1.0,
            'warmup_epochs': 20, 'weight_decay': 0.01, 'backbone_wd': 0.05,
            'weight_decay_policy': 'vit_standard', 'amsgrad': False,
            'lr_scheduler': 'poly', 'poly_exponent': 0.9,
        }},
    )
    optimizer, _ = trainer.configure_optimizers()
    members = {id(p) for group in optimizer.param_groups for p in group['params']}
    for head in model.decoder.seg_layers:
        assert all(p.requires_grad and id(p) in members for p in head.parameters())
