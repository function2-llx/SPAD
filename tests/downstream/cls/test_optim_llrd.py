"""LLRD (layer-wise learning-rate decay) parameter-group tests for the cls harness.

Contract mirrors downstream/seg: `scratch` params (the linear head) get the base head LR;
backbone layer `i` of `n` gets `lr_encoder * layer_decay ** (n - 1 - i)`, so the top block
trains at the full encoder LR and shallower layers decay geometrically. `layer_decay == 1`
must reproduce today's two-group behavior exactly.
"""
import pytest
import torch
from torch import nn

from pumit.downstream.cls.optim import build_param_groups


class _FakeBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)


class _FakeViT(nn.Module):
    """Minimal stand-in for the SPAD ViT layer topology (embeddings / layer / norm)."""

    def __init__(self, dim: int = 8, depth: int = 3):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.embeddings.proj = nn.Linear(dim, dim)
        self.layer = nn.ModuleList(_FakeBlock(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)


class _FakeEncoder(nn.Module):
    def __init__(self, dim: int = 8, depth: int = 3):
        super().__init__()
        self.vit = _FakeViT(dim, depth)
        self.embed_dim = dim

    def parameter_layers(self):
        from pumit.downstream.cls.optim import make_parameter_layers
        return make_parameter_layers(
            self.vit.embeddings.parameters(), self.vit.layer, self.vit.norm.parameters()
        )


def _encoder_and_head(dim: int = 8, depth: int = 3):
    return _FakeEncoder(dim, depth), nn.Linear(dim, 2)


def test_layer_decay_one_reproduces_two_groups():
    """layer_decay=1 must be exactly today's behavior: one encoder group + one head group."""
    encoder, head = _encoder_and_head()
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=1.0)
    assert len(groups) == 2
    lrs = sorted(g['lr'] for g in groups)
    assert lrs == [1e-5, 1e-3]
    # every trainable parameter appears exactly once
    seen = [id(p) for g in groups for p in g['params']]
    expected = [id(p) for p in [*encoder.parameters(), *head.parameters()]]
    assert sorted(seen) == sorted(expected)
    assert all(g['weight_decay'] == 1e-2 for g in groups)


def test_llrd_assigns_geometric_lrs_top_layer_at_base():
    """Top backbone layer gets lr_encoder; each shallower layer multiplies by layer_decay.

    Asserts the LR of each *named* layer (not a sorted set), so an inverted decay direction
    -- shallowest at base LR, deepest most decayed -- fails here.
    """
    encoder, head = _encoder_and_head(depth=3)
    decay = 0.5
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=decay)
    by_name = {g['name']: g['lr'] for g in groups}
    # 4 backbone layers (embeddings + 3 blocks; norm folded into the top block) + 1 head
    n = 4
    for index in range(n):
        expected = 1e-5 * decay ** (n - 1 - index)
        assert by_name[f'backbone_layer_{index:02d}'] == pytest.approx(expected), (
            f'layer {index} must be at lr_encoder * {decay}^{n - 1 - index}'
        )
    # direction: deepest is undecayed base, shallowest is the most decayed
    assert by_name['backbone_layer_03'] == pytest.approx(1e-5)
    assert by_name['backbone_layer_00'] == pytest.approx(1e-5 * decay ** 3)
    assert by_name['backbone_layer_00'] < by_name['backbone_layer_03']
    assert by_name['scratch'] == 1e-3


def test_deepest_layer_holds_the_final_norm():
    """The final norm folds into the top block, so it trains at the undecayed encoder LR."""
    encoder, head = _encoder_and_head(depth=3)
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=0.5)
    top = next(g for g in groups if g['name'] == 'backbone_layer_03')
    top_ids = {id(p) for p in top['params']}
    assert {id(p) for p in encoder.vit.norm.parameters()} <= top_ids
    assert {id(p) for p in encoder.vit.layer[-1].parameters()} <= top_ids
    assert top['lr'] == pytest.approx(1e-5)


def test_embeddings_are_the_shallowest_layer():
    """Layer 0 is the patch/position embedding, i.e. the most decayed group."""
    encoder, head = _encoder_and_head(depth=3)
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=0.5)
    bottom = next(g for g in groups if g['name'] == 'backbone_layer_00')
    assert {id(p) for p in encoder.vit.embeddings.parameters()} == {id(p) for p in bottom['params']}
    assert bottom['lr'] == pytest.approx(1e-5 * 0.5 ** 3)


def test_partition_covers_every_trainable_parameter_exactly_once():
    encoder, head = _encoder_and_head(depth=4)
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=0.75)
    seen = [id(p) for g in groups for p in g['params']]
    assert len(seen) == len(set(seen)), 'a parameter landed in two groups'
    expected = {id(p) for p in [*encoder.parameters(), *head.parameters()]}
    assert set(seen) == expected


def test_frozen_parameters_are_excluded():
    encoder, head = _encoder_and_head()
    encoder.vit.layer[0].fc.weight.requires_grad_(False)
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=0.75)
    seen = {id(p) for g in groups for p in g['params']}
    assert id(encoder.vit.layer[0].fc.weight) not in seen
    assert all(p.requires_grad for g in groups for p in g['params'])


def test_rejects_invalid_layer_decay():
    encoder, head = _encoder_and_head()
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match='layer_decay'):
            build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                               weight_decay=1e-2, layer_decay=bad)


def test_requires_parameter_layers_when_decaying():
    """An encoder without parameter_layers() must fail loudly rather than silently flatten."""
    class _NoLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

    with pytest.raises(TypeError, match='parameter_layers'):
        build_param_groups(_NoLayers(), nn.Linear(8, 2), lr_encoder=1e-5, lr_head=1e-3,
                           weight_decay=1e-2, layer_decay=0.75)


def test_real_spad_vit_partition_covers_every_parameter():
    """A real (randomly initialized) SPAD ViT-L: 25 layers, full trainable coverage.

    Guards the contract that every trainable ViT parameter lives under embeddings / layer / norm.
    If a future ViT gains a trainable parameter outside those (e.g. a projection head or an
    un-popped mask_token), _trainable_layer_ids raises and this test fails in CI rather than at
    finetune runtime.
    """
    from pumit.downstream.cls.backbones.vit_spad import SpadEncoder
    from pumit.downstream.cls.config import load_vit_config
    from pumit.model.vit import ViT

    config = load_vit_config('configs/downstream/cls/vit_l.yaml')
    encoder = SpadEncoder(ViT(config), dims=3)
    layers = encoder.parameter_layers()
    assert len(layers) == 1 + config.num_hidden_layers == 25   # embeddings + 24 blocks

    head = nn.Linear(encoder.embed_dim, 11)
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=0.9)
    covered = [id(p) for g in groups for p in g['params']]
    assert len(covered) == len(set(covered))
    expected = {id(p) for p in [*encoder.parameters(), *head.parameters()] if p.requires_grad}
    assert set(covered) == expected
    by_name = {g['name']: g['lr'] for g in groups}
    assert by_name['backbone_layer_24'] == pytest.approx(1e-5)              # top block undecayed
    assert by_name['backbone_layer_00'] == pytest.approx(1e-5 * 0.9 ** 24)  # embeddings



def test_weight_decay_policy_all_is_the_default_and_decays_everything():
    encoder, head = _encoder_and_head()
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=1.0)
    assert all(g['weight_decay'] == 1e-2 for g in groups)
    assert not any('no_decay' in g['name'] for g in groups)


def test_vit_standard_exempts_norms_biases_and_tokens():
    """norms/biases/cls_token land in a wd=0 group; only >=2-D weights keep weight decay."""
    encoder, head = _encoder_and_head()
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=1.0,
                                weight_decay_policy='vit_standard')
    no_decay = [g for g in groups if g['name'].endswith('no_decay')]
    decay = [g for g in groups if not g['name'].endswith('no_decay')]
    assert no_decay and decay
    assert all(g['weight_decay'] == 0.0 for g in no_decay)
    assert all(g['weight_decay'] == 1e-2 for g in decay)
    # the cls_token (1x1xD, and a NoWeightDecayParameter in the real ViT) must be exempt
    exempt = {id(p) for g in no_decay for p in g['params']}
    assert id(encoder.vit.embeddings.cls_token) in exempt
    assert id(encoder.vit.layer[0].norm1.weight) in exempt   # LayerNorm weight is 1-D
    assert id(encoder.vit.layer[0].fc.bias) in exempt        # bias is 1-D
    # a 2-D linear weight still decays
    decayed = {id(p) for g in decay for p in g['params']}
    assert id(encoder.vit.layer[0].fc.weight) in decayed
    # partition is still exact
    covered = [id(p) for g in groups for p in g['params']]
    assert len(covered) == len(set(covered))
    assert set(covered) == {id(p) for p in [*encoder.parameters(), *head.parameters()]}


def test_vit_standard_composes_with_llrd():
    """Each backbone layer splits into decay/no_decay groups that share that layer's LR."""
    encoder, head = _encoder_and_head(depth=3)
    groups = build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                                weight_decay=1e-2, layer_decay=0.5,
                                weight_decay_policy='vit_standard')
    by_name = {g['name']: g for g in groups}
    for index in range(4):
        expected_lr = 1e-5 * 0.5 ** (4 - 1 - index)
        pair = [g for name, g in by_name.items()
                if name.startswith(f'backbone_layer_{index:02d}')]
        assert pair, f'layer {index} produced no groups'
        assert all(g['lr'] == pytest.approx(expected_lr) for g in pair)
    covered = [id(p) for g in groups for p in g['params']]
    assert set(covered) == {id(p) for p in [*encoder.parameters(), *head.parameters()]}


def test_rejects_unknown_weight_decay_policy():
    encoder, head = _encoder_and_head()
    with pytest.raises(ValueError, match='weight_decay_policy'):
        build_param_groups(encoder, head, lr_encoder=1e-5, lr_head=1e-3,
                           weight_decay=1e-2, weight_decay_policy='bogus')


def test_spad_encoder_honors_img_size_and_token_grid():
    """--img-size controls the resize target; 192/16 = 12 -> 1728 tokens (vs 256 -> 4096).

    Checks the patch-embedding grid rather than a full forward: xformers' attention kernels
    reject fp32/CPU, so a CPU forward is not available in this env.
    """
    from pumit.downstream.cls.backbones.vit_spad import IMG_SIZE, SpadEncoder
    from pumit.downstream.cls.config import load_vit_config
    from pumit.downstream.cls.data import resize_volume
    from pumit.model.vit import ViT

    config = load_vit_config('configs/downstream/cls/vit_l.yaml')
    assert IMG_SIZE == 256, 'the default must stay 256 so existing results reproduce'
    vit = ViT(config)
    for edge, expected_tokens in ((256, 16 ** 3), (192, 12 ** 3)):
        assert edge % config.patch_size == 0, f'{edge} must be divisible by the patch size'
        encoder = SpadEncoder(vit, dims=3, img_size=edge)
        assert encoder.img_size == edge
        resized = resize_volume(torch.randn(1, 3, 8, 8, 8), encoder.img_size, is_3d=True)
        assert resized.shape[-3:] == (edge, edge, edge)
        with torch.no_grad():
            tokens = vit.embeddings.patch_embeddings(resized, da=0)
        n_tokens = tokens.shape[1] if tokens.ndim == 3 else tokens[0].numel() // config.hidden_size
        assert n_tokens == expected_tokens, (
            f'img_size={edge} must yield ({edge}/{config.patch_size})^3 = {expected_tokens} tokens'
        )


def test_real_vit_no_decay_group_exempts_marked_tokens():
    """On the real ViT-L, vit_standard must exempt the NoWeightDecayParameter-marked tokens.

    cls_token/register_tokens are typed NoWeightDecayParameter, a marker plain AdamW ignores --
    so without this policy the cls readout's own token is weight-decayed.
    """
    from pumit.downstream.cls.backbones.vit_spad import SpadEncoder
    from pumit.downstream.cls.config import load_vit_config
    from pumit.model.vit import ViT
    from pumit.types import NoWeightDecayParameter

    config = load_vit_config('configs/downstream/cls/vit_l.yaml')
    encoder = SpadEncoder(ViT(config), dims=3)
    head = nn.Linear(encoder.embed_dim, 11)
    groups = build_param_groups(encoder, head, lr_encoder=1e-4, lr_head=1e-3,
                                weight_decay=1e-2, weight_decay_policy='vit_standard')
    exempt = {id(p) for g in groups if g['weight_decay'] == 0.0 for p in g['params']}
    marked = [(n, p) for n, p in encoder.named_parameters()
              if isinstance(p, NoWeightDecayParameter)]
    assert marked, 'the real ViT must carry NoWeightDecayParameter markers'
    for name, parameter in marked:
        assert id(parameter) in exempt, f'{name} is marked no-decay but was assigned weight decay'
    # and every >=2-D weight still decays
    decayed = {id(p) for g in groups if g['weight_decay'] > 0 for p in g['params']}
    assert id(encoder.vit.layer[0].attention.q_proj.weight) in decayed
    assert id(encoder.vit.layer[0].mlp.up_proj.weight) in decayed
