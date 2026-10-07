import importlib.util
import pathlib

import torch


def _load_module():
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        'cache_text_embeddings', root / 'scripts/ucpt/cache_text_embeddings.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_encode_pads_to_L_and_builds_mask():
    """encode_prompts returns (L,1152) zero-padded tensors + (L,) bool masks
    with real-length True."""
    mod = _load_module()

    embs, masks = mod.encode_prompts(['segmentation mask of liver in a CT scan'])
    assert embs[0].shape == (mod.SEQ_LEN, 1152)
    assert masks[0].shape == (mod.SEQ_LEN,)
    assert masks[0].dtype == torch.bool
    real = int(masks[0].sum())
    assert real == 9  # 'segmentation mask of liver in a CT scan' -> 9 SigLIP2 tokens (verified)
    # padded positions are exactly zero
    assert embs[0][real:].abs().max().item() == 0.0


def test_build_prompts_uses_text_as_key_and_deduplicates_equal_variants():
    mod = _load_module()
    captions = {
        'source-a': {'raw-a': ['liver', 'hepatic parenchyma']},
        'source-b': {'raw-b': ['liver']},
    }

    prompts = mod.build_prompts(captions)

    assert len(prompts) == 2 * len(mod.MODALITY_CONTEXTS)
    assert 'segmentation mask of liver in a CT scan' in prompts
    assert 'segmentation mask of hepatic parenchyma in a CT scan' in prompts
