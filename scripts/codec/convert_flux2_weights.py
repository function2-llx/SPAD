"""Convert BFL ae.safetensors (FLUX.2 dev repo) to diffusers-format .pt for SPADFlux2AE.

BFL key layout (from refs/repos/flux2/src/flux2/autoencoder.py):
  encoder.down.{i}.block.{j}.*          -> resnets in down block i
  encoder.down.{i}.downsample.conv.*    -> downsampler in down block i
  encoder.mid.block_1/block_2/attn_1.*  -> mid block
  encoder.norm_out.*                    -> conv_norm_out
  encoder.quant_conv.*                  -> quant_conv
  decoder.up.{i}.block.{j}.*            -> resnets (BFL prepends: index 0 = highest res)
  decoder.up.{i}.upsample.conv.*        -> upsampler (same reversal)
  decoder.mid.block_1/block_2/attn_1.*  -> mid block
  decoder.norm_out.*                    -> conv_norm_out
  decoder.post_quant_conv.*             -> post_quant_conv
  bn.*                                  -> excluded

BFL Decoder reversal: `self.up.insert(0, up)` while iterating reversed(range(4)).
So BFL up.0 = highest-resolution block, but our diffusers up_blocks.0 = lowest-resolution.
BFL index i maps to diffusers index 3-i.

Attention weights are Conv2d [C, C, 1, 1] and must be squeezed to [C, C] for Linear.
"""

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


_NUM_BLOCKS = 4


def _remap_key(key: str) -> str | None:
    """Map a single BFL key to its diffusers equivalent. Returns None to drop the key."""
    if key.startswith('bn.'):
        return None

    # Attention sub-keys: must run BEFORE mid-block rules so attn_1.* sub-parts are
    # still present in the suffix.
    # encoder.mid.attn_1.q.* -> encoder.mid_block.attentions.0.to_q.*
    # decoder.mid.attn_1.q.* -> decoder.mid_block.attentions.0.to_q.*
    key = re.sub(r'(encoder|decoder)\.mid\.attn_1\.q\.', r'\1.mid_block.attentions.0.to_q.', key)
    key = re.sub(r'(encoder|decoder)\.mid\.attn_1\.k\.', r'\1.mid_block.attentions.0.to_k.', key)
    key = re.sub(r'(encoder|decoder)\.mid\.attn_1\.v\.', r'\1.mid_block.attentions.0.to_v.', key)
    key = re.sub(r'(encoder|decoder)\.mid\.attn_1\.proj_out\.', r'\1.mid_block.attentions.0.to_out.0.', key)
    key = re.sub(r'(encoder|decoder)\.mid\.attn_1\.norm\.', r'\1.mid_block.attentions.0.group_norm.', key)

    # Attention sub-keys in down/up blocks (none in default FLUX.2 config, but handle for safety).
    # encoder.down.{i}.attn.{j}.q.* etc. -- not present in default config, skip.

    # Mid block resnets (after attn sub-keys are already remapped above).
    key = re.sub(r'(encoder|decoder)\.mid\.block_1\.', r'\1.mid_block.resnets.0.', key)
    key = re.sub(r'(encoder|decoder)\.mid\.block_2\.', r'\1.mid_block.resnets.1.', key)

    # nin_shortcut -> conv_shortcut (applies to both encoder and decoder resnets).
    key = key.replace('.nin_shortcut.', '.conv_shortcut.')

    # norm_out -> conv_norm_out.
    key = re.sub(r'(encoder|decoder)\.norm_out\.', r'\1.conv_norm_out.', key)

    # Encoder down blocks: down.{i}.block.{j} -> down_blocks.{i}.resnets.{j}.
    key = re.sub(
        r'encoder\.down\.(\d+)\.block\.(\d+)\.',
        lambda m: f'encoder.down_blocks.{m.group(1)}.resnets.{m.group(2)}.',
        key,
    )
    # Encoder downsamplers: down.{i}.downsample.conv -> down_blocks.{i}.downsamplers.0.conv.
    key = re.sub(
        r'encoder\.down\.(\d+)\.downsample\.conv\.',
        lambda m: f'encoder.down_blocks.{m.group(1)}.downsamplers.0.conv.',
        key,
    )

    # Decoder up blocks (reversed): up.{i}.block.{j} -> up_blocks.{3-i}.resnets.{j}.
    key = re.sub(
        r'decoder\.up\.(\d+)\.block\.(\d+)\.',
        lambda m: f'decoder.up_blocks.{_NUM_BLOCKS - 1 - int(m.group(1))}.resnets.{m.group(2)}.',
        key,
    )
    # Decoder upsamplers (reversed): up.{i}.upsample.conv -> up_blocks.{3-i}.upsamplers.0.conv.
    key = re.sub(
        r'decoder\.up\.(\d+)\.upsample\.conv\.',
        lambda m: f'decoder.up_blocks.{_NUM_BLOCKS - 1 - int(m.group(1))}.upsamplers.0.conv.',
        key,
    )

    return key


def convert(src: Path) -> dict[str, torch.Tensor]:
    raw = load_file(src)
    out: dict[str, torch.Tensor] = {}

    attn_weight_suffixes = ('.to_q.weight', '.to_k.weight', '.to_v.weight', '.to_out.0.weight')

    for bfl_key, tensor in raw.items():
        new_key = _remap_key(bfl_key)
        if new_key is None:
            continue

        # Squeeze Conv2d [C, C, 1, 1] attention weights to [C, C] for Linear.
        if tensor.ndim == 4 and tensor.shape[2] == 1 and tensor.shape[3] == 1:
            if any(new_key.endswith(s) for s in attn_weight_suffixes):
                tensor = tensor.squeeze(-1).squeeze(-1)

        out[new_key] = tensor

    return out


def validate(sd: dict[str, torch.Tensor]) -> None:
    from pumit.codec.flux2 import SPADFlux2AE

    model = SPADFlux2AE()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys after conversion: {unexpected}")
    if missing:
        print(f"[warn] Missing keys (will use random init): {len(missing)} keys")


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert BFL ae.safetensors to diffusers-format .pt')
    parser.add_argument('--input', required=True, type=Path, help='Path to BFL ae.safetensors')
    parser.add_argument('--output', required=True, type=Path, help='Output .pt path')
    args = parser.parse_args()

    print(f'Loading {args.input}')
    sd = convert(args.input)
    print(f'Converted {len(sd)} keys')

    print('Validating against SPADFlux2AE ...')
    validate(sd)
    print('Validation passed.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.output)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
