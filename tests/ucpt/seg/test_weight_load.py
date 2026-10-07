import pathlib

import pytest
import torch
from safetensors import safe_open

from pumit.ucpt.seg.decoder import FusionEncoder, SemanticHead
from pumit.ucpt.seg.neck import SPADNeck
from pumit.ucpt.seg.weight_load import load_sam3_weights

CKPT = 'pretrained/facebook/sam3/model.safetensors'


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_load_reports_matched():
    neck, fusion, head = SPADNeck(), FusionEncoder(), SemanticHead()
    report = load_sam3_weights(CKPT, neck=neck, fusion=fusion, head=head)
    assert report['loaded'] > 0
    assert report['loaded'] >= 6 * 4  # fusion encoder: >=4 matched tensors per layer
    assert isinstance(report['unmatched_target'], list)


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_fusion_encoder_all_layers_loaded():
    fusion = FusionEncoder()
    report = load_sam3_weights(CKPT, neck=SPADNeck(), fusion=fusion, head=SemanticHead())
    fusion_detail = report['detail']['fusion']
    # 6 layers, 26 tensors each = 156 total
    assert fusion_detail['matched'] >= 6 * 26, \
        f'fusion matched {fusion_detail["matched"]}, expected >= 156, missing: {fusion_detail["missing"][:10]}'


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_head_core_keys_loaded():
    head = SemanticHead()
    report = load_sam3_weights(CKPT, neck=SPADNeck(), fusion=FusionEncoder(), head=head)
    head_detail = report['detail']['head']
    # prompt_cross_attn (8) + prompt_cross_attn_norm (2) + pixel_decoder convs (6) + norms (6) + semantic_projection (2) = 24
    assert head_detail['matched'] >= 24, \
        f'head matched {head_detail["matched"]}, expected >= 24, missing: {head_detail["missing"][:10]}'


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_neck_keys_loaded():
    neck = SPADNeck()
    report = load_sam3_weights(CKPT, neck=neck, fusion=FusionEncoder(), head=SemanticHead())
    neck_detail = report['detail']['neck']
    # 16 mapped keys: 8 weights + 8 biases for the 8 conv layers
    assert neck_detail['matched'] >= 8, \
        f'neck matched {neck_detail["matched"]}, expected >= 8, missing: {neck_detail["missing"][:10]}'


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_weights_actually_change():
    """Verify that loading actually changes weights from random init."""
    head = SemanticHead()
    original_weight = head.semantic_projection.weight.clone()
    load_sam3_weights(CKPT, neck=SPADNeck(), fusion=FusionEncoder(), head=head)
    assert not torch.allclose(original_weight, head.semantic_projection.weight), \
        'semantic_projection.weight unchanged after load'


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_return_structure():
    neck, fusion, head = SPADNeck(), FusionEncoder(), SemanticHead()
    report = load_sam3_weights(CKPT, neck=neck, fusion=fusion, head=head)
    assert set(report.keys()) == {'loaded', 'unmatched_target', 'detail'}
    assert set(report['detail'].keys()) == {'fusion', 'head', 'neck'}
    for key in ('fusion', 'head', 'neck'):
        info = report['detail'][key]
        assert 'matched' in info
        assert 'missing' in info
        assert 'unexpected' in info


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason='SAM 3 ckpt not present')
def test_load_text_projection_when_requested():
    projection = torch.nn.Linear(1024, 256)
    report = load_sam3_weights(
        CKPT,
        neck=SPADNeck(),
        fusion=FusionEncoder(),
        head=SemanticHead(),
        text_projection=projection,
    )

    with safe_open(CKPT, framework='pt') as checkpoint:
        torch.testing.assert_close(
            projection.weight,
            checkpoint.get_tensor('detector_model.text_projection.weight'),
        )
        torch.testing.assert_close(
            projection.bias,
            checkpoint.get_tensor('detector_model.text_projection.bias'),
        )
    assert report['detail']['text_projection']['matched'] == 2
