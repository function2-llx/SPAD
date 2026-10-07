"""Load and serve pre-computed text embeddings for class prompts."""

from pathlib import Path

import torch
from torch import Tensor, nn


class TextEncoder(nn.Module):
    """Adapts cached frozen text-encoder features to the seg decoder space.

    ``resizer`` mirrors SAM 3's detector text projection. Its input width is configured independently from the cache
    path and checkpoint initialization.
    """

    def __init__(self, *, text_embed_dim: int = 1152, hidden_size: int = 256):
        super().__init__()
        self.resizer = nn.Linear(text_embed_dim, hidden_size)

    def forward(self, text_raw: Tensor) -> Tensor:
        """text_raw: (K, L, text_embed_dim) -> (K, L, hidden_size)."""
        return self.resizer(text_raw)


class TextEmbeddingCache:
    """Pre-computed token embeddings indexed by exact prompt text."""

    def __init__(self, cache_path: Path | str):
        """
        Args:
            cache_path: Serialized token-embedding cache.
        """
        data = torch.load(cache_path, weights_only=True)
        if data.get('key_type') != 'prompt_text':
            raise ValueError(
                f'{cache_path} is not a prompt-text-keyed embedding cache; regenerate it with '
                'scripts/ucpt/cache_text_embeddings.py'
            )
        self.embeddings: dict[str, Tensor] = data['embeddings']
        self.valid_masks: dict[str, Tensor] = data['valid_masks']
        self.dim: int = data['dim']
        self.seq_len: int = data['seq_len']
        if self.embeddings.keys() != self.valid_masks.keys():
            raise ValueError(f'{cache_path}: embedding and valid-mask prompt keys differ')

    def require_prompts(self, prompts: list[str]) -> None:
        """Fail if the cache does not cover every requested prompt."""
        missing = sorted(set(prompts) - self.embeddings.keys())
        if missing:
            preview = missing[:5]
            raise KeyError(f'text cache is missing {len(missing)} required prompts: {preview!r}')

    def get(self, prompt: str) -> Tensor:
        return self.embeddings[prompt]

    def get_batch(self, prompts: list[str]) -> Tensor:
        return torch.stack([self.embeddings[prompt] for prompt in prompts])

    def get_mask_batch(self, prompts: list[str]) -> Tensor:
        return torch.stack([self.valid_masks[prompt] for prompt in prompts])
