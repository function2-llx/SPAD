"""Tests for the Universal U-Net Task-Aware Bottleneck."""

import math

import pytest
import torch

from pumit.spad_unet.task_aware_bottleneck import (
    _Attention,
    MultiScaleTaskAwareBottleneck,
    RandomFourierPositionEncoding3D,
    TaskAwareBottleneck,
)


def test_random_fourier_position_encoding_matches_declared_formula():
    encoding = RandomFourierPositionEncoding3D(embedding_dim=4, seed=17)

    actual = encoding((2, 1, 1), dtype=torch.float32)
    coordinates = torch.tensor([
        [0.25, 0.5, 0.5],
        [0.75, 0.5, 0.5],
    ])
    phases = 2 * math.pi * ((2 * coordinates - 1) @ encoding.gaussian_matrix)
    expected = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)[None]

    torch.testing.assert_close(actual, expected)


def test_random_fourier_position_encoding_is_seeded_checkpoint_state():
    first = RandomFourierPositionEncoding3D(embedding_dim=8, seed=23)
    second = RandomFourierPositionEncoding3D(embedding_dim=8, seed=23)

    assert torch.equal(first.gaussian_matrix, second.gaussian_matrix)
    assert 'gaussian_matrix' in first.state_dict()


def test_tab_preserves_grid_and_backpropagates_to_selected_bank_rows():
    adapter = TaskAwareBottleneck(
        bottleneck_channels=8,
        num_datasets=3,
        tokens_per_dataset=16,
        embedding_dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=128,
        attention_downsample_rate=2,
    )
    shared_bottleneck = torch.randn(
        1,
        8,
        2,
        3,
        4,
    )
    bottleneck = shared_bottleneck.expand(2, -1, -1, -1, -1).clone().requires_grad_()

    output = adapter(bottleneck, torch.tensor([0, 2]))
    output.mean().backward()

    assert output.shape == bottleneck.shape
    assert not torch.equal(output[0], output[1])
    assert bottleneck.grad is not None
    assert adapter.task_tokens.grad is not None
    assert adapter.task_tokens.grad[0].abs().sum() > 0
    assert adapter.task_tokens.grad[1].abs().sum() == 0
    assert adapter.task_tokens.grad[2].abs().sum() > 0


def test_multiscale_tab_preserves_independent_feature_grids_and_channels():
    adapter = MultiScaleTaskAwareBottleneck(
        feature_channels=(8, 12, 16),
        num_datasets=3,
        tokens_per_dataset=4,
        embedding_dim=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        attention_downsample_rate=2,
    )
    features = [
        torch.randn(2, 8, 4, 6, 8, requires_grad=True),
        torch.randn(2, 12, 2, 3, 4, requires_grad=True),
        torch.randn(2, 16, 1, 2, 2, requires_grad=True),
    ]

    outputs = adapter(features, torch.tensor([0, 2]))
    sum(output.mean() for output in outputs).backward()

    assert [output.shape for output in outputs] == [
        feature.shape for feature in features
    ]
    assert len(adapter.input_projections) == 3
    assert [projection.in_channels for projection in adapter.input_projections] == [
        8,
        12,
        16,
    ]
    assert adapter.level_embeddings.grad is not None
    assert adapter.task_tokens.grad is not None
    assert adapter.task_tokens.grad[0].abs().sum() > 0
    assert adapter.task_tokens.grad[1].abs().sum() == 0
    assert adapter.task_tokens.grad[2].abs().sum() > 0


def test_tab_output_projections_use_channels_last_weight_layout():
    adapter = TaskAwareBottleneck(
        bottleneck_channels=24,
        num_datasets=2,
        embedding_dim=16,
        num_heads=4,
        mlp_dim=32,
    )
    assert adapter.output_projection.weight.stride() == (16, 1, 16, 16, 16)

    multi_scale_adapter = MultiScaleTaskAwareBottleneck(
        feature_channels=(24, 32, 40),
        num_datasets=2,
        embedding_dim=16,
        num_heads=4,
        mlp_dim=32,
    )
    assert [projection.weight.stride() for projection in multi_scale_adapter.output_projections] == [
        (16, 1, 16, 16, 16),
        (16, 1, 16, 16, 16),
        (16, 1, 16, 16, 16),
    ]


def test_sdpa_attention_matches_declared_scaled_dot_product():
    torch.manual_seed(19)
    attention = _Attention(
        embedding_dim=64,
        num_heads=4,
        downsample_rate=2,
    )
    query = torch.randn(2, 7, 64)
    key = torch.randn(2, 11, 64)
    value = torch.randn(2, 11, 64)

    actual = attention(query, key, value)
    q = attention._separate_heads(attention.q_proj(query)).transpose(1, 2)
    k = attention._separate_heads(attention.k_proj(key)).transpose(1, 2)
    v = attention._separate_heads(attention.v_proj(value)).transpose(1, 2)
    weights = torch.softmax(
        q @ k.transpose(-2, -1) / math.sqrt(attention.head_dim),
        dim=-1,
    )
    expected = attention.out_proj(
        attention._recombine_heads((weights @ v).transpose(1, 2).contiguous())
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for the production BF16 SDPA preflight',
)
def test_tab_sdpa_bfloat16_compiles_fullgraph_on_cuda():
    torch.manual_seed(29)
    device = torch.device('cuda')
    adapter = TaskAwareBottleneck(
        bottleneck_channels=320,
        num_datasets=12,
    ).to(device=device, dtype=torch.bfloat16)
    bottleneck = torch.randn(
        3,
        320,
        4,
        6,
        6,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    dataset_indices = torch.tensor([0, 5, 11], device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        eager_output = adapter(bottleneck, dataset_indices)
        torch.cuda.synchronize()
    sdpa_operators = sorted({
        event.key
        for event in profiler.key_averages()
        if 'scaled_dot_product' in event.key
    })
    print({
        'device': torch.cuda.get_device_name(),
        'capability': torch.cuda.get_device_capability(),
        'sdpa_operators': sdpa_operators,
    })
    assert sdpa_operators

    compiled = torch.compile(adapter, fullgraph=True, dynamic=False)
    compiled_output = compiled(bottleneck, dataset_indices)
    compiled_output.square().mean().backward()

    print({
        'compiled_eager_max_abs_diff': (
            (compiled_output - eager_output).abs().max().detach().item()
        ),
    })
    torch.testing.assert_close(
        compiled_output,
        eager_output,
        rtol=2e-2,
        atol=2e-2,
    )
    assert torch.isfinite(compiled_output).all()
    assert bottleneck.grad is not None
    assert torch.isfinite(bottleneck.grad).all()
    assert adapter.task_tokens.grad is not None
    assert torch.isfinite(adapter.task_tokens.grad).all()
