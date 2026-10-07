import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def _load_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        'cache_sam3_text_embeddings',
        root / 'scripts/ucpt/cache_sam3_text_embeddings.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def __call__(self, prompts, **kwargs):
        seq_len = kwargs['max_length']
        input_ids = torch.zeros(len(prompts), seq_len, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, prompt in enumerate(prompts):
            real = len(prompt.split()) + 2
            input_ids[index, :real] = torch.arange(1, real + 1)
            attention_mask[index, :real] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class _TextEncoder(nn.Module):
    def forward(self, input_ids, attention_mask, return_dict):
        hidden = input_ids.unsqueeze(-1).expand(-1, -1, 1024).float()
        return SimpleNamespace(last_hidden_state=hidden)


def test_encode_prompts_returns_preprojection_sam3_width_and_masks():
    module = _load_module()
    prompts = ['liver', 'left kidney']
    embeddings, masks = module.encode_prompts(
        prompts,
        _Tokenizer(),
        _TextEncoder(),
        batch_size=1,
        device=torch.device('cpu'),
    )

    assert [embedding.shape for embedding in embeddings] == [(32, 1024), (32, 1024)]
    assert [int(mask.sum()) for mask in masks] == [3, 4]
    for embedding, mask in zip(embeddings, masks, strict=True):
        assert embedding[~mask].count_nonzero() == 0
