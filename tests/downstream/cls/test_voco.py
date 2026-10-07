"""VoCo adapter contract: topology, loader discipline, readout shapes."""
import pytest
import torch
import torch.nn.functional as F

from pumit.downstream.cls.backbones import voco
from pumit.downstream.seg.backbones._voco_swin import UnetrBasicBlock


def pretrained_state_dict():
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
    return state


def test_forward_matches_official_five_branch_embedding(tmp_path):
    torch.save(pretrained_state_dict(), tmp_path / 'ckpt.pt')
    enc = voco.VoCoEncoder(weights=str(tmp_path / 'ckpt.pt'), img_size=64, feature_size=6)
    enc.eval()
    x = torch.rand(1, 1, 28, 28, 28)
    with torch.no_grad():
        global_features, patch_tokens = enc(x)
        resized = F.interpolate(x, size=(64, 64, 64), mode='trilinear', align_corners=False)
        h0, h1, h2, _, h4 = enc.swin(resized)
        # Spell out the upstream readout independently of the adapter's pooling helpers.
        expected = torch.cat(
            [
                F.adaptive_avg_pool3d(enc.encoder1(resized), 1).flatten(1),
                F.adaptive_avg_pool3d(enc.encoder2(h0), 1).flatten(1),
                F.adaptive_avg_pool3d(enc.encoder3(h1), 1).flatten(1),
                F.adaptive_avg_pool3d(enc.encoder4(h2), 1).flatten(1),
                F.adaptive_avg_pool3d(enc.encoder10(h4), 1).flatten(1),
            ],
            dim=1,
        )
    assert global_features.shape == (1, 144)
    assert patch_tokens.shape == (1, 8, 96)
    torch.testing.assert_close(global_features, expected)
    torch.testing.assert_close(patch_tokens, h4.flatten(2).transpose(1, 2))
    assert enc.embed_dim == 144


@pytest.mark.parametrize('wrapped', [False, True])
def test_loader_preserves_every_pretrained_parameter(tmp_path, wrapped):
    state = pretrained_state_dict()
    torch.save({'state_dict': state} if wrapped else state, tmp_path / 'ckpt.pt')
    enc = voco.VoCoEncoder(weights=str(tmp_path / 'ckpt.pt'), img_size=64, feature_size=6)
    expected = {
        'swin.' + key.removeprefix('swinViT.') if key.startswith('swinViT.') else key: value
        for key, value in state.items()
    }
    actual = enc.state_dict()
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    'prefix', ['swinViT.patch_embed.', 'encoder1.', 'encoder2.', 'encoder3.', 'encoder4.', 'encoder10.'],
)
def test_loader_requires_every_pretrained_branch(tmp_path, prefix):
    state = pretrained_state_dict()
    state.pop(next(key for key in state if key.startswith(prefix)))
    torch.save(state, tmp_path / 'missing.pt')
    with pytest.raises(RuntimeError, match='Missing key'):
        voco.VoCoEncoder(weights=str(tmp_path / 'missing.pt'), img_size=64, feature_size=6)


def test_loader_rejects_unknown_parameters(tmp_path):
    state = pretrained_state_dict()
    state['decoder5.layer.conv1.conv.weight'] = torch.zeros(1)
    torch.save(state, tmp_path / 'unknown.pt')
    with pytest.raises(RuntimeError, match='Unexpected key'):
        voco.VoCoEncoder(weights=str(tmp_path / 'unknown.pt'), img_size=64, feature_size=6)


@pytest.mark.parametrize('branch', ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder10'])
def test_loader_rejects_incompatible_branch_weights(tmp_path, branch):
    state = pretrained_state_dict()
    key = next(key for key in state if key.startswith(f'{branch}.'))
    state[key] = torch.zeros(1)
    torch.save(state, tmp_path / 'incompatible.pt')
    with pytest.raises(RuntimeError, match='size mismatch'):
        voco.VoCoEncoder(weights=str(tmp_path / 'incompatible.pt'), img_size=64, feature_size=6)


def test_transform_batch_is_unit_range_single_channel():
    import numpy as np
    x = voco.transform_batch(np.full((2, 8, 8, 8), 255, dtype=np.uint8), is_3d=True)['x']
    assert x.shape == (2, 1, 8, 8, 8)
    assert x.max().item() == 1.0
    with pytest.raises(ValueError, match='3D-only'):
        voco.transform_batch(np.zeros((2, 8, 8), dtype=np.uint8), is_3d=False)
