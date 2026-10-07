"""Pre-compute SAM 3 token embeddings for UCPT segmentation prompts.

The cache stays keyed by exact rendered prompt text. It stores the 1024-d token states before SAM 3's detector
``text_projection`` so projection initialization remains an independent model configuration.

Usage:
    pixi run -e default python scripts/ucpt/cache_sam3_text_embeddings.py \
        --captions-dir src/pumit/ucpt/seg/class_captions \
        --model pretrained/facebook/sam3 \
        --output precompute/ucpt/sam3_text_embeddings.pt
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, Sam3Model

from pumit.text_prompt import (
    build_segmentation_prompts,
    class_captions_sha256,
    load_class_captions,
)

SEQ_LEN = 32


def load_text_encoder(model_path: Path, device: torch.device) -> tuple[object, nn.Module]:
    """Load the native SAM 3 tokenizer and text encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Sam3Model.from_pretrained(model_path, low_cpu_mem_usage=True)
    text_encoder = model.text_encoder.to(device).eval()
    del model
    return tokenizer, text_encoder


def encode_prompts(
    prompts: list[str],
    tokenizer,
    text_encoder: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Encode prompts at SAM 3's fixed context length."""
    embeddings: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        tokens = tokenizer(
            batch,
            padding='max_length',
            max_length=SEQ_LEN,
            truncation=False,
            return_tensors='pt',
        )
        if tokens.input_ids.shape[1] != SEQ_LEN:
            raise ValueError(f'a prompt exceeds SAM 3 context length {SEQ_LEN}: {batch!r}')
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        with torch.inference_mode():
            hidden = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state.float()
        valid = attention_mask.bool()
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0)
        embeddings.extend(hidden.cpu().unbind())
        masks.extend(valid.cpu().unbind())
    return embeddings, masks


def main() -> None:
    parser = argparse.ArgumentParser(description='Pre-compute SAM 3 token embeddings for UCPT prompts.')
    parser.add_argument('--captions-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))
    args = parser.parse_args()

    captions = load_class_captions(args.captions_dir)
    prompts = build_segmentation_prompts(captions)
    print(f'Encoding {len(prompts)} distinct prompts with SAM 3')

    tokenizer, text_encoder = load_text_encoder(args.model, args.device)
    embeddings, masks = encode_prompts(
        prompts,
        tokenizer,
        text_encoder,
        batch_size=args.batch_size,
        device=args.device,
    )

    embedding_dict = dict(zip(prompts, embeddings, strict=True))
    mask_dict = dict(zip(prompts, masks, strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'key_type': 'prompt_text',
            'embeddings': embedding_dict,
            'valid_masks': mask_dict,
            'dim': embeddings[0].shape[1],
            'seq_len': SEQ_LEN,
            'encoder': 'sam3',
            'model': str(args.model),
            'captions_sha256': class_captions_sha256(captions),
        },
        args.output,
    )
    print(f'Saved {len(embedding_dict)} token sequences shaped ({SEQ_LEN}, {embeddings[0].shape[1]}) to {args.output}')


if __name__ == '__main__':
    main()
