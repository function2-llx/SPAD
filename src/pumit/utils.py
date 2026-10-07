import hashlib
import os

import msgpack
import torch
from torch import nn

__all__ = [
    'deterministic_seed',
    'is_natural_modality',
    'load_ckpt',
]


def deterministic_seed(*args) -> int:
    """Derive a deterministic int seed from msgpack-serializable arguments.

    Uses sha256 for uniform distribution, truncated to 8 bytes (64-bit seed).
    Avoids Python's hash() which is randomized per-process via PYTHONHASHSEED.
    """
    return int.from_bytes(hashlib.sha256(msgpack.packb(args)).digest()[:8])

def is_natural_modality(modality: str) -> bool:
    return modality.startswith('RGB') or modality.startswith('gray')


def load_ckpt(
    model: nn.Module,
    ckpt_or_path: dict | str | bytes | os.PathLike | None,
    state_dict_key: str | None = None,
    key_prefix: str = '',
):
    if ckpt_or_path is None:
        return
    if isinstance(ckpt_or_path, dict):
        ckpt = ckpt_or_path
    elif isinstance(ckpt_or_path, (str, os.PathLike)) and str(ckpt_or_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(str(ckpt_or_path))
        state_dict_key = None
    else:
        ckpt: dict = torch.load(ckpt_or_path, map_location='cpu')
    if state_dict_key is None:
        if 'state_dict' in ckpt:
            state_dict_key = 'state_dict'
        elif 'model' in ckpt:
            state_dict_key = 'model'
    from timm.models import clean_state_dict
    state_dict = clean_state_dict(ckpt if state_dict_key is None else ckpt[state_dict_key])
    if key_prefix:
        state_dict = {
            k[len(key_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(key_prefix)
        }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if not isinstance(ckpt_or_path, dict):
        print(f'Loaded {state_dict_key} from checkpoint {ckpt_or_path}')
    print('missing keys:', missing_keys)
    print('unexpected keys:', unexpected_keys)
