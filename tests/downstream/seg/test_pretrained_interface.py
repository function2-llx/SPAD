from types import SimpleNamespace

import pytest
import torch

from pumit.downstream.seg import registry
from pumit.downstream.seg.adapters.pyramid import PlanAlignedPyramidEncoder3D
from pumit.downstream.seg.mask2former import PlanAlignedMask2FormerSegmentationNetwork
from pumit.downstream.seg.network import (
    PlanAlignedSegmentationNetwork,
    PlanAlignedUNetWithoutRefinerSegmentationNetwork,
)
from tests.downstream.seg.test_network import (
    _CoarseFlatEncoder,
    _FlatBackbone,
    _architecture_kwargs,
)


@pytest.mark.parametrize(
    'network_class',
    [
        PlanAlignedSegmentationNetwork,
        PlanAlignedUNetWithoutRefinerSegmentationNetwork,
        PlanAlignedMask2FormerSegmentationNetwork,
    ],
)
@pytest.mark.parametrize('use_weights', [False, True])
def test_encoder_pretrained_interface_preserves_readout(
    network_class, use_weights, tmp_path, monkeypatch,
):
    def build_encoder(name, plan, input_channels, config):
        if network_class is PlanAlignedSegmentationNetwork:
            return PlanAlignedPyramidEncoder3D(
                _FlatBackbone(), plan, input_channels=input_channels,
            )
        return _CoarseFlatEncoder(plan)

    monkeypatch.setattr(registry, 'build_encoder', build_encoder)
    network = network_class(
        input_channels=1,
        num_classes=3,
        backbone_name='test',
        backbone_config={},
        deep_supervision=True,
        **_architecture_kwargs(),
    )
    encoder_parameter_ids = {id(parameter) for parameter in network.encoder.parameters()}
    readout_before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
        if id(parameter) not in encoder_parameter_ids
    }
    calls = []

    def load_encoder(encoder, weights):
        calls.append((encoder, weights))
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.fill_(0.125)

    monkeypatch.setattr(
        registry, 'BACKBONES', {'test': SimpleNamespace(load_pretrained=load_encoder)},
    )
    weights = tmp_path / 'weights.pth' if use_weights else None
    network.load_pretrained(weights)

    assert calls == [(network.encoder, weights)]
    assert all(torch.all(parameter == 0.125) for parameter in network.encoder.parameters())
    assert readout_before
    for name, parameter in network.named_parameters():
        if name in readout_before:
            assert torch.equal(parameter, readout_before[name])
