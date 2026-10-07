from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    flag: str
    size: int          # native MedMNIST download size (must be a supported size)
    is_3d: bool
    task: str
    n_classes: int


def load_dataset_specs(path: str | Path) -> dict[str, DatasetSpec]:
    raw = yaml.safe_load(Path(path).read_text())
    return {
        flag: DatasetSpec(flag=flag, **entry)
        for flag, entry in raw.items()
    }


def load_vit_config(path: str | Path):
    from pumit.model.vit import ViTConfig  # local: only SPAD backbones need it (py3.13 chain)
    raw = yaml.safe_load(Path(path).read_text())
    return ViTConfig(**raw)
