"""UCPT training configuration and jsonargparse integration."""

from dataclasses import dataclass, field
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace


@dataclass
class UCPTDataConfig:
    """Training inputs and masked-view sampling."""

    stream_dir: str = ''
    latent_dir: str | None = None
    virtual_lanes: int = 32
    data_root: str = ''
    text_cache_path: str = ''
    class_captions_dir: str = ''
    random_mask_ratio_2d: tuple[float, float] = (0.70, 0.80)
    random_mask_ratio_3d: tuple[float, float] = (0.75, 0.85)
    block_mask_ratio_2d: tuple[float, float] = (0.70, 0.80)
    block_mask_ratio_3d: tuple[float, float] = (0.75, 0.85)


@dataclass
class UCPTModelConfig:
    """UCPT model architecture, objectives, and initialization."""

    # ViT
    pretrained: str = ''
    n_register_tokens: int = 4
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    grad_ckpt: bool = True
    # None checkpoints every ViT layer.
    grad_ckpt_first_n_layers: int | None = None

    # SSL
    recon_decoder_depth: int = 4
    recon_decoder_dim: int = 384
    recon_decoder_heads: int = 6
    latent_channels: int = 32
    recon_weight: float = 1.0
    image_distill_weight: float = 1.0
    patch_distill_weight: float = 1.0
    cls_predictor_hidden: int = 4096
    patch_distill_decoder_depth: int = 8
    patch_distill_decoder_dim: int = 384
    patch_distill_decoder_heads: int = 8
    teacher_momentum: float = 0.996

    # Segmentation
    text_embed_dim: int = 1152
    seg_hidden_size: int = 256
    fusion_grad_ckpt: bool = False
    seg_weight: float = 1.0
    artifact_momentum: float = 0.996
    sam3_checkpoint: str = ''
    load_sam3_text_projection: bool = False


@dataclass
class UCPTOptimConfig:
    """Optimization and learning-rate schedule."""

    steps: int = 100000
    lr: float = 1.5e-4
    layer_decay: float = 1.0
    weight_decay: float = 0.05
    warmup_steps: int = 10000
    cooldown_steps: int = 20000
    min_lr: float = 1.0e-6
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    ssl_lr_fraction: float = 1.0
    seg_lr_fraction: float = 0.3


@dataclass
class UCPTRunConfig:
    """Run identity, observability, and execution controls."""

    seed: int = 42
    save_dir: str = 'outputs/ucpt/main'
    wandb_entity: str | None = None
    wandb_project: str = 'pumit-ucpt'
    wandb_name: str | None = None
    log_every: int = 20
    save_every: int = 5000
    save_rotating_every: int = 200
    compile_prewarm_batches: int = 2
    num_workers: int = 4
    augment_threads: int = 6
    numa_shared: bool = True
    nccl_high_priority: bool = True
    ddp_bucket_cap_mb: float = 50.0
    tcmalloc_release_every: int = 0
    gc_interval: int = 100


@dataclass
class UCPTTrainConfig:
    """Complete UCPT training configuration."""

    data: UCPTDataConfig = field(default_factory=UCPTDataConfig)
    model: UCPTModelConfig = field(default_factory=UCPTModelConfig)
    optim: UCPTOptimConfig = field(default_factory=UCPTOptimConfig)
    run: UCPTRunConfig = field(default_factory=UCPTRunConfig)


def parse_config() -> tuple[UCPTTrainConfig, bool, str | None]:
    """Parse the training config and execution-only CLI flags."""
    parser = ArgumentParser(description='Train UCPT')
    parser.add_argument('--config', action='config', required=True, help='YAML config file')
    parser.add_class_arguments(UCPTTrainConfig)
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument(
        '--cache-archive',
        type=str,
        default=None,
        help='Persistent tar.zst of the inductor/triton cache. Extracted to /dev/shm before compile and archived after prewarming.',
    )
    args = parser.parse_args()
    return config_from_namespace(parser, args), args.no_compile, args.cache_archive


def config_from_namespace(parser: ArgumentParser, args: Namespace) -> UCPTTrainConfig:
    """Build the root dataclass from a parsed jsonargparse namespace."""
    args = parser.instantiate(args)
    return UCPTTrainConfig(
        data=args.data,
        model=args.model,
        optim=args.optim,
        run=args.run,
    )


def load_train_config(path: str | Path) -> UCPTTrainConfig:
    """Load and validate a UCPT YAML configuration."""
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(UCPTTrainConfig)
    return config_from_namespace(parser, parser.parse_path(path))
