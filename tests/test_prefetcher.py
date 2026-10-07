from dataclasses import dataclass

import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from pumit.prefetcher import _iter_tensors


@dataclass
class _NestedBatch:
    tensor: torch.Tensor
    tensors: list[torch.Tensor]
    mapping: dict[str, torch.Tensor]
    attn_bias: BlockDiagonalMask


def test_iter_tensors_recurses_through_batch_and_attention_bias():
    shared = torch.randn(2)
    bias = BlockDiagonalMask.from_seqlens([2, 3], device=torch.device('cpu'))
    batch = _NestedBatch(
        tensor=shared,
        tensors=[torch.randn(3), shared],
        mapping={'value': torch.randn(4)},
        attn_bias=bias,
    )

    tensors = list(_iter_tensors(batch))
    actual_ids = {id(tensor) for tensor in tensors}
    expected_ids = {
        id(shared),
        id(batch.tensors[0]),
        id(batch.mapping['value']),
        id(bias.q_seqinfo.seqstart),
        id(bias.k_seqinfo.seqstart),
    }
    assert len(tensors) == len(actual_ids)  # repeated references are recorded once
    assert actual_ids == expected_ids
