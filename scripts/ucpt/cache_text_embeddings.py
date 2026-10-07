"""Pre-compute token-level SigLIP2 text embeddings for UCPT segmentation prompts.

Reads the nested caption map (source -> {raw_name -> [captions]}) and encodes
one token sequence per distinct rendered prompt across every caption variant.
Embeddings are keyed by the exact prompt text, so source-local label identities
never enter the text cache.

Each prompt is encoded at its natural length. The per-token `last_hidden_state`
is then zero-padded to `SEQ_LEN` with a bool valid mask marking the real
positions.

Usage:
    pixi run -e default python scripts/ucpt/cache_text_embeddings.py \
        --captions-dir src/pumit/ucpt/seg/class_captions \
        --output precompute/ucpt/text_embeddings.pt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from pumit.text_prompt import (
    MODALITY_CONTEXTS,
    build_segmentation_prompts,
    class_captions_sha256,
    load_class_captions,
)

MODEL_NAME = 'pretrained/siglip2-so400m-patch14-384'
SEQ_LEN = 32


def build_prompts(
    captions: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Build the distinct prompts required by the caption taxonomy.

    Takes the Cartesian product of every caption variant and supported coarse
    modality. Equal rendered text from different raw label identities appears
    once.

    Returns:
        Sorted unique prompt strings.
    """
    return build_segmentation_prompts(captions)


def encode_prompts(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Encode each prompt at natural length, then pad to SEQ_LEN."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text_model = AutoModel.from_pretrained(MODEL_NAME).text_model.cuda().eval()
    embeddings: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for p in prompts:
        enc = tokenizer([p], return_tensors='pt')
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            h = text_model(**enc).last_hidden_state[0].float().cpu()  # (real_len, 1152)
        real = h.shape[0]
        assert real <= SEQ_LEN, f'prompt {p!r} has {real} tokens > SEQ_LEN={SEQ_LEN}'
        padded = torch.zeros(SEQ_LEN, h.shape[1])
        padded[:real] = h
        embeddings.append(padded)
        masks.append(torch.arange(SEQ_LEN) < real)
    return embeddings, masks


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute token-level SigLIP2 text embeddings for the UCPT caption taxonomy.'
    )
    parser.add_argument(
        '--captions-dir',
        type=Path,
        required=True,
        help='Directory containing one <source>.json caption document per label source',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output .pt file for cached token embeddings + masks',
    )
    args = parser.parse_args()

    captions = load_class_captions(args.captions_dir)
    n_pairs = sum(len(names) for names in captions.values())
    print(f'Loaded {len(captions)} sources, {n_pairs} (source, name) pairs')
    prompts = build_prompts(captions)
    print(
        f'Will encode {len(prompts)} distinct prompts from '
        f'{n_pairs} source/name pairs x {len(MODALITY_CONTEXTS)} modalities'
    )

    embeddings, masks = encode_prompts(prompts)
    print(f'Embedding dim: {embeddings[0].shape[1]}, seq_len: {SEQ_LEN}')

    embedding_dict: dict[str, torch.Tensor] = dict(zip(prompts, embeddings))
    mask_dict: dict[str, torch.Tensor] = dict(zip(prompts, masks))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'key_type': 'prompt_text',
            'embeddings': embedding_dict,
            'valid_masks': mask_dict,
            'dim': embeddings[0].shape[1],
            'seq_len': SEQ_LEN,
            'captions_sha256': class_captions_sha256(captions),
        },
        args.output,
    )
    print(f'Saved {len(embedding_dict)} token sequences (L={SEQ_LEN}) to {args.output}')


if __name__ == '__main__':
    main()
