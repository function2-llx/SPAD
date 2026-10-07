"""Shared helpers for UCPT tests.

This conftest is loaded by pytest for every test module under ``tests/ucpt/``.
Helpers here are plain functions (not fixtures) so they can be imported
explicitly by the tests that need them.
"""
import dataclasses

import torch


def use_cpu_attention_reference(monkeypatch):
    """Use dense SDPA for xformers calls in CPU-only integration tests."""
    if torch.cuda.is_available():
        return
    from torch.nn import functional as F

    def reference_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
        mask = attn_bias
        if attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
            mask = attn_bias.materialize(
                (q.shape[0], q.shape[2], q.shape[1], k.shape[1]), device=q.device, dtype=q.dtype,
            )
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, dropout_p=p, scale=scale,
        ).transpose(1, 2)

    monkeypatch.setattr('pumit.model.vit.xops.memory_efficient_attention', reference_attention)


def cast_ucpt_batch_dtype(batch, dt):
    """Cast float-tensor fields in UCPTBatch to dt.

    ``UCPTBatch.to`` only moves device; this casts floating-point fields in
    place. Walks dataclass fields, skipping ``device='cpu'`` metadata fields
    (Python ints fenced from the compiled graph). Handles ``Tensor`` and
    ``list[Tensor]`` field types.
    """
    for f in dataclasses.fields(batch):
        if f.metadata.get('device') == 'cpu':
            continue
        val = getattr(batch, f.name)
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            setattr(batch, f.name, val.to(dt))
        elif isinstance(val, list):
            setattr(batch, f.name, [
                t.to(dt) if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                for t in val
            ])
    return batch
