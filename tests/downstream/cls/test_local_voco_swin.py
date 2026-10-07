import copy

import pytest
import torch

from pumit.downstream.seg.backbones import _voco_swin as voco


def test_disabled_options_preserve_state_dict_and_outputs():
    kwargs = dict(in_chans=1, embed_dim=3, window_size=(2, 2, 2), num_heads=(1, 1, 1, 1))
    model = voco.SwinTransformer(**kwargs).eval()
    explicit = voco.SwinTransformer(
        **kwargs, drop_path_rate=0.0, use_checkpoint=False, attention_chunk_size=None,
    ).eval()
    explicit.load_state_dict(model.state_dict(), strict=True)
    assert model.state_dict().keys() == explicit.state_dict().keys()
    assert not any('drop_path' in name for name in model.state_dict())
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        expected = model(x)
        actual = explicit(x)
    for expected_stage, actual_stage in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_stage, expected_stage, rtol=0, atol=0)


@pytest.mark.parametrize('shifted', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_chunked_attention_preserves_outputs_and_gradients(shifted, training):
    reference = voco.WindowAttention(12, 3, (2, 2, 2)).train(training)
    chunked = copy.deepcopy(reference)
    chunked.attention_chunk_size = 5
    mask = voco.compute_mask((4, 4, 4), (2, 2, 2), (1, 1, 1), torch.device('cpu')) if shifted else None
    x = torch.randn(24, 8, 12, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = reference(x, mask)
    actual = chunked(y, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-5)
    for expected_parameter, actual_parameter in zip(reference.parameters(), chunked.parameters(), strict=True):
        assert actual_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-5)


def test_drop_path_schedule_and_checkpoint_options():
    model = voco.SwinTransformer(
        in_chans=1, embed_dim=3, num_heads=(1, 1, 1, 1),
        drop_path_rate=0.1, use_checkpoint=True, attention_chunk_size=5,
    )
    blocks = [block for stage in (model.layers1, model.layers2, model.layers3, model.layers4) for block in stage[0].blocks]
    assert len(blocks) == 8
    rates = [getattr(block.drop_path, 'drop_prob', 0.0) for block in blocks]
    assert rates == pytest.approx(torch.linspace(0, 0.1, 8).tolist())
    assert all(block.use_checkpoint for block in blocks)
    assert all(block.attn.attention_chunk_size == 5 for block in blocks)


def test_block_checkpoint_covers_attention_and_mlp(monkeypatch):
    reference = voco.SwinTransformerBlock(12, 3, (2, 2, 2), (1, 1, 1))
    chunked = copy.deepcopy(reference)
    chunked.use_checkpoint = True
    chunked.attn.attention_chunk_size = 5
    checkpoint_calls = []
    original_checkpoint = voco.checkpoint

    def record_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append((function.__self__, function.__name__))
        assert kwargs['use_reentrant'] is False
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(voco, 'checkpoint', record_checkpoint)
    mask = voco.compute_mask((4, 4, 4), (2, 2, 2), (1, 1, 1), torch.device('cpu'))
    x = torch.randn(2, 4, 4, 4, 12, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = reference(x, mask)
    actual = chunked(y, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-5)
    for expected_parameter, actual_parameter in zip(reference.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-4, atol=1e-4)
    assert (chunked, '_attention') in checkpoint_calls
    assert (chunked, '_mlp') in checkpoint_calls
    assert (chunked.attn, '_attention') in checkpoint_calls
