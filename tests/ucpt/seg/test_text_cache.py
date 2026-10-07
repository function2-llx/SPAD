import orjson
import numpy as np
import pytest
import torch

from pumit.text_prompt import (
    SegmentationPromptResolver,
    build_segmentation_prompt,
    build_segmentation_prompts,
    normalize_modality,
)
from pumit.ucpt.seg.text_encoding import TextEmbeddingCache

L = 24


def _mask(real: int) -> torch.Tensor:
    return torch.arange(L) < real


def test_normalize_modality_maps_mri_subtypes():
    assert normalize_modality('MRI/T2-FLAIR') == 'MRI'
    assert normalize_modality('CT') == 'CT'


def test_normalize_modality_raises_on_unknown():
    with pytest.raises(KeyError):
        normalize_modality('not-a-modality')


def test_prompt_resolver_maps_raw_label_identity_to_exact_text(tmp_path):
    path = tmp_path / 'captions.json'
    path.write_bytes(orjson.dumps({
        'KiTS23': {'tumor': ['kidney tumor']},
        'BUSI': {'tumor': ['breast tumor']},
    }))
    resolver = SegmentationPromptResolver(path)

    assert resolver.get('KiTS23', 'tumor', 'CT') == \
        'segmentation mask of kidney tumor in a CT scan'
    assert resolver.get('BUSI', 'tumor', 'US') == \
        'segmentation mask of breast tumor in an ultrasound image'


def test_prompt_resolver_samples_text_variants(tmp_path):
    path = tmp_path / 'captions.json'
    path.write_bytes(orjson.dumps({
        'KiTS23': {'tumor': ['kidney tumor', 'renal tumor']},
    }))
    resolver = SegmentationPromptResolver(path)

    assert resolver.get('KiTS23', 'tumor', 'CT') == \
        'segmentation mask of kidney tumor in a CT scan'
    prompts = {
        resolver.get(
            'KiTS23',
            'tumor',
            'CT',
            rng=np.random.default_rng(seed),
        )
        for seed in range(16)
    }
    assert prompts == {
        'segmentation mask of kidney tumor in a CT scan',
        'segmentation mask of renal tumor in a CT scan',
    }


def test_prompt_inventory_includes_every_variant():
    prompts = build_segmentation_prompts({
        'source-a': {'raw-a': ['liver', 'hepatic parenchyma']},
        'source-b': {'raw-b': ['liver']},
    })

    assert build_segmentation_prompt('liver', 'CT') in prompts
    assert build_segmentation_prompt('hepatic parenchyma', 'CT') in prompts


def test_cache_serves_token_sequences_and_masks(tmp_path):
    prompt = 'segmentation mask of liver in a CT scan'
    emb = torch.zeros(L, 1152)
    emb[:6] = torch.randn(6, 1152)
    mask = _mask(6)
    path = tmp_path / 'cache.pt'
    torch.save({'key_type': 'prompt_text',
                'embeddings': {prompt: emb}, 'valid_masks': {prompt: mask},
                'dim': 1152, 'seq_len': L}, path)

    cache = TextEmbeddingCache(path)
    assert cache.dim == 1152
    assert cache.seq_len == L
    assert cache.get_batch([prompt]).shape == (1, L, 1152)
    m = cache.get_mask_batch([prompt])
    assert m.shape == (1, L)
    assert m.dtype == torch.bool
    assert int(m.sum()) == 6


def test_distinct_prompt_texts_resolve_to_distinct_embeddings(tmp_path):
    kidney_prompt = 'segmentation mask of kidney tumor in a CT scan'
    breast_prompt = 'segmentation mask of breast tumor in an ultrasound image'
    kits_tumor = torch.randn(L, 1152)
    busi_tumor = torch.randn(L, 1152)
    emb = {
        kidney_prompt: kits_tumor,
        breast_prompt: busi_tumor,
    }
    masks = {kidney_prompt: _mask(5), breast_prompt: _mask(5)}
    path = tmp_path / 'cache.pt'
    torch.save(
        {'key_type': 'prompt_text', 'embeddings': emb, 'valid_masks': masks, 'dim': 1152, 'seq_len': L},
        path,
    )
    cache = TextEmbeddingCache(path)
    assert torch.equal(cache.get(kidney_prompt), kits_tumor)
    assert torch.equal(cache.get(breast_prompt), busi_tumor)
    assert not torch.equal(cache.get(kidney_prompt), cache.get(breast_prompt))


def test_get_mask_batch_reflects_per_key_lengths(tmp_path):
    gallbladder_prompt = 'segmentation mask of gallbladder in a CT scan'
    femur_prompt = 'segmentation mask of left femur in a CT scan'
    emb = {
        gallbladder_prompt: torch.zeros(L, 4),
        femur_prompt: torch.zeros(L, 4),
    }
    masks = {
        gallbladder_prompt: _mask(9),
        femur_prompt: _mask(11),
    }
    path = tmp_path / 'te.pt'
    torch.save(
        {'key_type': 'prompt_text', 'embeddings': emb, 'valid_masks': masks, 'dim': 4, 'seq_len': L},
        path,
    )
    cache = TextEmbeddingCache(path)
    m = cache.get_mask_batch([gallbladder_prompt, femur_prompt])
    assert m.shape == (2, L)
    assert m[0].sum().item() == 9
    assert m[1].sum().item() == 11


def test_cache_rejects_legacy_key_format(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save(
        {
            'embeddings': {('source', 'class', 'CT'): torch.zeros(1, 2)},
            'valid_masks': {('source', 'class', 'CT'): torch.ones(1, dtype=torch.bool)},
            'dim': 2,
            'seq_len': 1,
        },
        path,
    )

    with pytest.raises(ValueError, match='prompt-text-keyed'):
        TextEmbeddingCache(path)


def test_cache_requires_complete_prompt_coverage(tmp_path):
    prompt = build_segmentation_prompt('liver', 'CT')
    path = tmp_path / 'cache.pt'
    torch.save(
        {
            'key_type': 'prompt_text',
            'embeddings': {prompt: torch.zeros(2, 4)},
            'valid_masks': {prompt: torch.ones(2, dtype=torch.bool)},
            'dim': 4,
            'seq_len': 2,
        },
        path,
    )
    cache = TextEmbeddingCache(path)
    cache.require_prompts([prompt])
    with pytest.raises(KeyError, match='missing 1 required'):
        cache.require_prompts([prompt, build_segmentation_prompt('kidney', 'CT')])
