"""3D U-Net adapter contract: topology, per-release loader discipline, readout shapes."""
import numpy as np
import pytest
import torch

from pumit.downstream.cls.backbones import unet3d


def write_checkpoint(path, release, unet=None, extra=(), drop=()):
    """Serialize an encoder in one release's layout, with that release's non-encoder keys."""
    unet = unet or unet3d.UNetEncoder()
    sd = {f'{release.prefix}{k}': v for k, v in unet.state_dict().items()}
    sd.update({k: torch.zeros(1) for k in extra})
    for key in drop:
        sd.pop(key)
    torch.save({'epoch': 0, release.container: sd}, path)
    return unet


GENESIS_EXTRA = ('module.up_tr256.up_conv.weight', 'module.out_tr.final_conv.weight')
SUPREM_EXTRA = ('module.backbone.up_tr256.up_conv.weight', 'module.organ_embedding',
                'module.precls_conv.0.weight', 'module.GAP.0.weight',
                'module.controller.weight', 'module.text_to_vision.weight')
RELEASES = [(unet3d.GENESIS, GENESIS_EXTRA), (unet3d.SUPREM, SUPREM_EXTRA)]
IDS = ['genesis', 'suprem']


def test_topology_matches_the_released_checkpoint_layout():
    sd = unet3d.UNetEncoder().state_dict()
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    assert shapes['down_tr64.ops.0.conv1.weight'] == (32, 1, 3, 3, 3)
    assert shapes['down_tr64.ops.1.conv1.weight'] == (64, 32, 3, 3, 3)
    assert shapes['down_tr512.ops.1.conv1.weight'] == (512, 256, 3, 3, 3)
    assert 'down_tr512.ops.1.bn1.running_var' in shapes
    assert sum(v.numel() for k, v in sd.items()
               if 'running' not in k and 'num_batches' not in k) == 7_027_776


@pytest.mark.parametrize('release,extra', RELEASES, ids=IDS)
def test_forward_shapes_and_gap_readout(tmp_path, release, extra):
    write_checkpoint(tmp_path / 'ckpt.pt', release, extra=extra)
    enc = unet3d.UNet3DEncoder(weights=str(tmp_path / 'ckpt.pt'), img_size=96, release=release)
    global_features, patch_tokens = enc(torch.rand(2, 1, 64, 64, 64))
    assert global_features.shape == (2, 512)
    assert patch_tokens.shape == (2, 1728, 512)         # (96/8)^3 = 12^3 bottleneck positions
    torch.testing.assert_close(global_features, patch_tokens.mean(dim=1))
    assert enc.embed_dim == 512


@pytest.mark.parametrize('release,extra', RELEASES, ids=IDS)
def test_loader_strips_prefix_drops_non_encoder_keys(tmp_path, release, extra):
    unet = write_checkpoint(tmp_path / 'ok.pt', release, extra=extra)
    loaded = unet3d.load_encoder_state_dict(str(tmp_path / 'ok.pt'), release)
    assert loaded.keys() == unet.state_dict().keys()


@pytest.mark.parametrize('release,extra', RELEASES, ids=IDS)
def test_loader_rejects_a_key_the_release_does_not_declare(tmp_path, release, extra):
    write_checkpoint(tmp_path / 'bad.pt', release, extra=(*extra, 'module.dense_1.weight'))
    with pytest.raises(RuntimeError, match='unexpected keys'):
        unet3d.load_encoder_state_dict(str(tmp_path / 'bad.pt'), release)


@pytest.mark.parametrize('release,extra', RELEASES, ids=IDS)
def test_loader_requires_every_encoder_key(tmp_path, release, extra):
    write_checkpoint(tmp_path / 'missing.pt', release, extra=extra,
                     drop=[f'{release.prefix}down_tr512.ops.1.bn1.running_mean'])
    with pytest.raises(RuntimeError, match='Missing key'):
        unet3d.UNet3DEncoder(weights=str(tmp_path / 'missing.pt'), img_size=96, release=release)


def test_each_release_rejects_the_other_layout(tmp_path):
    """A SuPreM checkpoint read as Genesis (or vice versa) must fail, not silently miss weights."""
    write_checkpoint(tmp_path / 'genesis.pt', unet3d.GENESIS, extra=GENESIS_EXTRA)
    write_checkpoint(tmp_path / 'suprem.pt', unet3d.SUPREM, extra=SUPREM_EXTRA)
    with pytest.raises((RuntimeError, KeyError)):
        unet3d.load_encoder_state_dict(str(tmp_path / 'suprem.pt'), unet3d.GENESIS)
    with pytest.raises((RuntimeError, KeyError)):
        unet3d.load_encoder_state_dict(str(tmp_path / 'genesis.pt'), unet3d.SUPREM)


@pytest.mark.parametrize('release,extra', RELEASES, ids=IDS)
def test_checkpoint_values_reach_the_encoder(tmp_path, release, extra):
    unet = unet3d.UNetEncoder()
    with torch.no_grad():
        unet.down_tr64.ops[0].conv1.weight.fill_(0.5)
    write_checkpoint(tmp_path / 'ckpt.pt', release, unet, extra=extra)
    enc = unet3d.UNet3DEncoder(weights=str(tmp_path / 'ckpt.pt'), img_size=96, release=release)
    assert torch.all(enc.unet.down_tr64.ops[0].conv1.weight == 0.5)


def test_img_size_must_be_a_multiple_of_the_bottleneck_stride(tmp_path):
    write_checkpoint(tmp_path / 'ckpt.pt', unet3d.GENESIS, extra=GENESIS_EXTRA)
    with pytest.raises(ValueError, match='not divisible by 8'):
        unet3d.UNet3DEncoder(weights=str(tmp_path / 'ckpt.pt'), img_size=60,
                             release=unet3d.GENESIS)


def test_builders_are_3d_only(tmp_path):
    write_checkpoint(tmp_path / 'ckpt.pt', unet3d.GENESIS, extra=GENESIS_EXTRA)
    with pytest.raises(ValueError, match='3D-only'):
        unet3d.models_genesis(dims=2, device='cpu', weights=str(tmp_path / 'ckpt.pt'),
                              img_size=96)


def test_transform_batch_is_unit_range_single_channel():
    x = unet3d.transform_batch(np.full((2, 8, 8, 8), 255, dtype=np.uint8), is_3d=True)['x']
    assert x.shape == (2, 1, 8, 8, 8)
    assert x.max().item() == 1.0
    with pytest.raises(ValueError, match='cannot transform 2D'):
        unet3d.transform_batch(np.zeros((2, 8, 8), dtype=np.uint8), is_3d=False)
