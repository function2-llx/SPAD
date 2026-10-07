"""VAE training configuration dataclasses."""

from dataclasses import dataclass
from typing import Literal

from pumit.data.config import DepthTierConfig, MAX_INPLANE_RATIO
from pumit.data.config import validate_depth_tiers as _validate_depth_tiers

MAX_DA = 4

# Re-export for backwards compatibility
__all__ = ['MAX_DA', 'MAX_INPLANE_RATIO', 'DepthTierConfig', 'validate_depth_tiers', 'TrainConfig']


def validate_depth_tiers(depth_tiers: dict[int | None, DepthTierConfig], *, max_da: int = MAX_DA) -> None:
    """Validate depth tier configuration (defaults max_da to codec's MAX_DA=4)."""
    _validate_depth_tiers(depth_tiers, max_da=max_da)


@dataclass
class TrainConfig:
    pretrained: str
    depth_tiers: dict[int | None, DepthTierConfig]
    steps: int
    lr: float
    warmup_steps: int
    save_dir: str | None = None
    accum_steps: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 10.0
    grad_ckpt: bool = True
    l1_weight: float = 1.0
    smooth_l1_beta: float = 0.0
    kl_weight: float = 1e-6
    ema_decay: float = 0.999
    save_every: int = 500
    log_every: int = 20
    viz_every: int = 100
    num_workers: int = 16
    wandb_project: str = 'pumit-codec'
    wandb_name: str | None = None
    resume: str | None = None
    smooth_spad: bool = True
    model: Literal['klvae', 'flux2'] = 'flux2'
    inflator: str = 'center'
    stream_dir: str | None = None

    def __post_init__(self):
        validate_depth_tiers(self.depth_tiers, max_da=MAX_DA)

    def validate(self):
        if self.save_dir is None:
            raise ValueError("save_dir must be specified (in config yaml or via --save-dir CLI arg)")
