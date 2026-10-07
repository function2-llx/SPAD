"""Stage 1: Train SPAD KL-VAE.

Usage:
    torchrun --nproc_per_node=N scripts/vae/train.py --config configs/codec/vae.yaml [overrides]
    python scripts/vae/train.py --config configs/codec/vae.yaml [overrides]
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from pumit.data import DABatch
from pumit.data.config import DepthTierConfig
from pumit.codec.datamodule import TransformConf
from pumit.codec import SPADKLVAE, SPADFlux2AE, CodecLoss
from pumit.compile_cache import parallel_precompile
from pumit.codec.config import MAX_DA, TrainConfig
from pumit.codec.replay_dataset import CodecReplayDataset
from pumit.codec.transforms import build_codec_pipeline


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _collect_warmup_shapes(
    cfg: TrainConfig,
) -> list[tuple[int | None, int | None, int, int]]:
    shapes = []
    for da_key, tier_cfg in cfg.depth_tiers.items():
        for depth, batch in zip(tier_cfg.tiers, tier_cfg.batch_sizes):
            if da_key is None:
                shapes.append((None, None, depth, batch))
                continue
            da_min = max(da_key - 1, 0)
            da_max = min(da_key + 1, MAX_DA)
            for da_dec in range(da_min, da_max + 1):
                shapes.append((da_key, da_dec, depth, batch))
    return shapes


def parse_config() -> tuple[TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description='Train SPAD codec (Stage 1)')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--spike-monitor-after', type=int, default=20)
    parser.add_argument('--spike-abs-threshold', type=float, default=0.10)
    parser.add_argument('--spike-ema-factor', type=float, default=3.0)
    parser.add_argument('--spike-cooldown', type=int, default=100)
    parser.add_argument('--stream-dir', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--skip-warmup', action='store_true')
    parser.add_argument('--cache-archive', type=str, default=None,
                        help='Path to torchinductor.tar.zst; extract to /dev/shm for faster warmup')
    parser.add_argument('--cache-dir', type=str, default='/dev/shm/torchinductor_cache/codec',
                        help='Fast local directory for compile cache (extracted from archive)')
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    # Convert depth_tiers from raw dicts to DepthTierConfig objects
    depth_tiers = {}
    for key, val in raw.pop('depth_tiers').items():
        depth_tiers[key if key is None else int(key)] = DepthTierConfig(
            tiers=tuple(val['tiers']),
            batch_sizes=tuple(val['batch_sizes']),
        )
    raw['depth_tiers'] = depth_tiers

    cfg = TrainConfig(**raw)

    # CLI overrides
    if args.lr is not None:
        cfg.lr = args.lr
    if args.steps is not None:
        cfg.steps = args.steps
    if args.resume is not None:
        cfg.resume = args.resume
    if args.wandb_name is not None:
        cfg.wandb_name = args.wandb_name
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.stream_dir is not None:
        cfg.stream_dir = args.stream_dir
    if args.save_every is not None:
        cfg.save_every = args.save_every

    cfg.validate()
    return cfg, args



def load_pretrained_weights(model: torch.nn.Module, pretrained: str):
    """Load 2D pretrained weights from a saved state_dict (.pt file)."""
    sd = torch.load(pretrained, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)


def save_reconstruction(
    save_dir: Path,
    step: int,
    rank: int,
    batch: DABatch,
    recon: torch.Tensor,
    da_enc: int | None,
    da_dec: int | None,
):
    from PIL import Image

    step_dir = save_dir / 'reconstruction' / f'step{step:06d}'
    step_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(min(2, batch.img.shape[0])):
        img = batch.img[sample_idx].detach().float().cpu()
        recon_img = recon[sample_idx].detach().float().cpu()
        d = img.shape[1]
        if d <= 1:
            indices = [0]
        else:
            indices = np.linspace(0, d - 1, min(4, d), dtype=int).tolist()

        panels = []
        for i in indices:
            inp_slice = img[:, i].clamp(0, 1)
            rec_slice = recon_img[:, i].clamp(0, 1)
            if batch.not_rgb[sample_idx]:
                inp_slice = inp_slice[0:1]
                rec_slice = rec_slice[0:1]
            panel = torch.cat([inp_slice, rec_slice], dim=-1)
            panels.append(panel)
        grid = torch.cat(panels, dim=-2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        if grid_np.shape[2] == 1:
            grid_np = grid_np[:, :, 0]

        fname = f'rank{rank}_s{sample_idx}_enc{da_enc}_dec{da_dec}.webp'
        Image.fromarray(grid_np).save(step_dir / fname, quality=90)



class _TeeStream:
    """Write to both a file and the original stream."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')
    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
    def flush(self):
        self.file.flush()
        self.stream.flush()
    def isatty(self):
        return self.stream.isatty()


def _setup_logging(save_dir: Path):
    """Tee stdout/stderr to save_dir/train.log while keeping console output."""
    log_file = open(save_dir / 'train.log', 'a')
    sys.stdout = _TeeStream(log_file, sys.__stdout__)
    sys.stderr = _TeeStream(log_file, sys.__stderr__)


def _save_checkpoint(
    save_dir: Path,
    optim_step: int,
    orig_model: torch.nn.Module,
    ema_shadow: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    scaler: torch.amp.GradScaler,
    wandb_id: str | None,
    config: dict,
    update_latest: bool = True,
):
    ckpt_dir = save_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'checkpoint-{optim_step}.pt'
    torch.save({
        'model': orig_model.state_dict(),
        'ema': ema_shadow,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'wandb_id': wandb_id,
        'step': optim_step,
        'config': config,
    }, ckpt_path)
    if update_latest:
        latest = save_dir / 'checkpoint-latest.pt'
        latest.unlink(missing_ok=True)
        latest.symlink_to(Path('checkpoints') / ckpt_path.name)
    print(f'Saved checkpoint to {ckpt_path}')
    return ckpt_path


def main():
    cfg, cli_args = parse_config()

    # Init distributed (torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, etc.)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    is_main = rank == 0
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True
    warmup_shapes = _collect_warmup_shapes(cfg)
    torch._dynamo.config.recompile_limit = len(warmup_shapes) + 8
    if is_main:
        print(f'[{_ts()}] {len(warmup_shapes)} warmup shapes, recompile_limit={len(warmup_shapes) + 8}')

    # Auto-resume: check for latest checkpoint in save_dir
    latest_ckpt = Path(cfg.save_dir) / 'latest-run' / 'checkpoint-latest.pt'
    if cfg.resume is None and latest_ckpt.exists():
        # Resolve through the latest-run symlink now: wandb.init re-points latest-run
        # to a fresh (empty) run dir below, which would otherwise invalidate this path.
        cfg.resume = str(latest_ckpt.resolve())
        if is_main:
            print(f'[{_ts()}] Auto-resuming from {latest_ckpt}')

    from pumit.spadop.conv import INFLATORS, SPADConv3d_K3S1, SPADConv3d_K3S2

    # Build model + load weights + compile + DDP
    if cfg.model == 'flux2':
        model = SPADFlux2AE(grad_ckpt=cfg.grad_ckpt)
    else:
        model = SPADKLVAE(grad_ckpt=cfg.grad_ckpt)
    if cfg.inflator != 'center':
        inflator_fn = INFLATORS[cfg.inflator]
        for m in model.modules():
            if isinstance(m, (SPADConv3d_K3S1, SPADConv3d_K3S2)):
                m.inflator = inflator_fn

    if is_main:
        print(f'[{_ts()}] Model architecture:\n{model}')

    save_dir = Path(cfg.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = None
    wandb_run_id = None
    if cfg.resume and not cfg.wandb_name:
        ckpt_meta = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        wandb_run_id = ckpt_meta.get('wandb_id')
        del ckpt_meta
    if is_main:
        import wandb
        wandb_kwargs = dict(project=cfg.wandb_project, name=cfg.wandb_name, config=vars(cfg))
        if wandb_run_id:
            wandb_kwargs.update(id=wandb_run_id, resume='allow')
        wandb.init(**wandb_kwargs)
        logger = wandb
        save_dir = save_dir / Path(wandb.run.dir).parent.name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{_ts()}] Output dir: {save_dir}')
        # Update latest-run symlink
        latest_run = Path(cfg.save_dir) / 'latest-run'
        latest_run.unlink(missing_ok=True)
        latest_run.symlink_to(save_dir.name)

    # Broadcast save_dir to all ranks (needed for multi-rank visualization)
    if world_size > 1:
        save_dir_list = [str(save_dir)]
        dist.broadcast_object_list(save_dir_list, src=0)
        save_dir = Path(save_dir_list[0])
        save_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging and save config
    if is_main:
        _setup_logging(save_dir)
        cfg_dict = {k: v if not isinstance(v, dict) else {str(kk): vars(vv) if hasattr(vv, '__dict__') else vv for kk, vv in v.items()} for k, v in vars(cfg).items()}
        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)
        print(f'[{_ts()}] Config saved to {save_dir / "config.yaml"}')

    # Load weights (safe now, no fork)
    if cfg.resume is None and cfg.pretrained and Path(cfg.pretrained).exists():
        load_pretrained_weights(model, cfg.pretrained)
        if is_main:
            print(f'[{_ts()}] Loaded pretrained from {cfg.pretrained}')
    model = model.to(device)

    # Loss
    loss_fn = CodecLoss(l1_weight=cfg.l1_weight, kl_weight=cfg.kl_weight, smooth_l1_beta=cfg.smooth_l1_beta).to(device)

    # Compile + DDP
    if is_main:
        print(f'[{_ts()}] Compiling full model...')
    model = torch.compile(model)
    model = DDP(model, device_ids=[device], gradient_as_bucket_view=True)
    raw_model = model.module
    orig_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model

    # Precompile: all ranks participate via dist store
    compile_baseline = None
    if not cli_args.skip_warmup:
        size_xy = TransformConf().size_xy
        precompile_inputs = [
            ((torch.randn(batch, 3, depth, size_xy, size_xy),),
             {"da": da_enc, "da_dec": da_dec})
            for da_enc, da_dec, depth, batch in warmup_shapes
        ]
        parallel_precompile(
            model, precompile_inputs,
            loss_fn=lambda out: out.recon.mean(),
            cache_archive=cli_args.cache_archive,
            cache_dir=cli_args.cache_dir,
            amp_dtype=torch.float16,
        )

        compile_baseline = torch._dynamo.utils.counters.get('stats', {}).get('unique_graphs')
    if compile_baseline is None:
        if is_main:
            print('WARNING: Dynamo unique_graphs counter unavailable; compile monitoring disabled')
    elif is_main:
        print(f'Compile warmup: {compile_baseline} unique graphs compiled')

    # Optimizer & scheduler (linear warmup + cosine decay)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': cfg.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=cfg.lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=cfg.warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps - cfg.warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[cfg.warmup_steps],
    )

    # EMA
    ema_shadow = {k: v.clone() for k, v in orig_model.state_dict().items()}

    # GradScaler for fp16 amp (prevents gradient underflow; default init_scale/factors).
    # Skip detection below relies on the default growth_factor=2.0 / backoff_factor=0.5.
    scaler = torch.amp.GradScaler('cuda')

    # Resume
    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        sd = ckpt['model']
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        orig_model.load_state_dict(sd)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        if 'ema' in ckpt:
            ema_shadow = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['ema'].items()}
        start_step = ckpt['step']
        print(f'Resumed from step {start_step}')

    # Spike monitor (rank 0 only)
    from pumit.codec.spike_monitor import SpikeMonitor
    spike_monitor = SpikeMonitor(
        monitor_after=cli_args.spike_monitor_after,
        abs_threshold=cli_args.spike_abs_threshold,
        ema_factor=cli_args.spike_ema_factor,
        cooldown=cli_args.spike_cooldown,
    ) if is_main else None

    # Data
    if cfg.stream_dir is None:
        raise ValueError("stream_dir is required (set in config YAML or via --stream-dir)")
    trans_conf = TransformConf()
    pipeline = build_codec_pipeline(trans_conf, depth_tiers=cfg.depth_tiers, max_da=MAX_DA, smooth_spad=cfg.smooth_spad)
    start_offset = start_step * cfg.accum_steps
    replay_ds = CodecReplayDataset(
        cfg.stream_dir,
        rank=rank,
        world_size=world_size,
        pipeline=pipeline,
        start_offset=start_offset,
    )
    loader = torch.utils.data.DataLoader(
        replay_ds, batch_size=None,
        num_workers=cfg.num_workers, prefetch_factor=2, persistent_workers=True,
    )
    data_iter = iter(loader)

    # Effective steps check
    if is_main:
        total_micro = len(replay_ds)
        effective_steps = total_micro // cfg.accum_steps
        target_steps = cfg.steps - start_step
        if effective_steps < target_steps:
            print(f'WARNING: stream only has {effective_steps} effective steps, but config requests {target_steps}')
            print(f'  Training will run for min({effective_steps}, {target_steps}) = {min(effective_steps, target_steps)} steps')

    # Training loop
    model.train()
    # This FP16 autocast is left over from a training experiment; the stable codec training recipe uses BF16.
    # FP16's observed accuracy advantage applies to offline latent precomputation, not optimization.
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
    logged_da_mem: set[int | None] = set()
    da_step_counts: dict[int | None, int] = {}
    no_sync = model.no_sync

    pbar = range(start_step, cfg.steps)
    if is_main:
        from tqdm import tqdm
        pbar = tqdm(pbar, initial=start_step, total=cfg.steps, desc='Training')

    t_start = torch.cuda.Event(enable_timing=True)
    t_data = torch.cuda.Event(enable_timing=True)
    t_compute = torch.cuda.Event(enable_timing=True)

    for step in pbar:
        t_start.record()
        optimizer.zero_grad()
        accum_losses: dict[str, float] = {}
        optim_step = step + 1
        is_log_step = is_main and optim_step % cfg.log_every == 0
        step_loss_sum = torch.tensor(0.0, device=device) if is_main else None

        for micro in range(cfg.accum_steps):
            sync_ctx = nullcontext() if micro == cfg.accum_steps - 1 else no_sync()
            batch = next(data_iter)
            x = batch.img.to(device)
            if micro == 0:
                t_data.record()
            da_enc = batch.da_enc
            da_dec = batch.da_dec if cfg.smooth_spad else batch.da_enc
            da_step_counts[da_enc] = da_step_counts.get(da_enc, 0) + 1

            with sync_ctx:
                with autocast_ctx:
                    output = model(x, da=da_enc, da_dec=da_dec)
                    assert x.shape == output.recon.shape, f'shape mismatch: x={tuple(x.shape)}, recon={tuple(output.recon.shape)}, da_enc={da_enc}, da_dec={da_dec}'
                    losses = loss_fn(x, output)
                loss = losses['loss'] / cfg.accum_steps
                scaler.scale(loss).backward()
                if is_main:
                    step_loss_sum += losses['loss'].detach()

            if is_log_step:
                for k, v in losses.items():
                    accum_losses[k] = accum_losses.get(k, 0.0) + v.item() / cfg.accum_steps

            da_group = batch.da_enc
            if is_main and da_group not in logged_da_mem:
                logged_da_mem.add(da_group)
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f'DA={da_group}: peak_mem={peak_gb:.1f} GB, img_shape={tuple(batch.img.shape)}')

        # Spike detection (before optimizer step so window captures the model that produced the spike)
        if is_main:
            step_loss = (step_loss_sum / cfg.accum_steps).item()
            if spike_monitor.check(step_loss, optim_step):
                print(f'[{_ts()}] SPIKE DETECTED at step {optim_step}: step_loss={step_loss:.4f} ema={spike_monitor.ema:.4f}')
                spike_paths = spike_monitor.dump_window(save_dir)
                print(f'[{_ts()}] Dumped {len(spike_paths)} spike checkpoints')

        # Optimizer step (fp16: unscale before clip so grad_norm is true-magnitude)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = scaler.get_scale() >= scale_before
        if optimizer_stepped:
            scheduler.step()
        if is_main and not optimizer_stepped and optim_step > cli_args.spike_monitor_after:
            print(f'[{_ts()}] WARNING: GradScaler skipped optimizer step at step {optim_step} '
                  f'(scale {scale_before} -> {scaler.get_scale()}, outside calibration window)')

        # EMA update (skip on a scaler-skipped step: weights did not change)
        if optimizer_stepped:
            with torch.no_grad():
                for k, v in orig_model.state_dict().items():
                    ema_shadow[k].lerp_(v, 1 - cfg.ema_decay)

        # Push state into spike monitor window (after optimizer + EMA, captures post-step state)
        if is_main:
            spike_monitor.push_state({
                'model': orig_model.state_dict(),
                'ema': ema_shadow,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'wandb_id': wandb.run.id if logger else None,
                'step': optim_step,
                'config': vars(cfg),
            })

        t_compute.record()

        # Logging
        if is_log_step:
            log_dict = {f'train/{k}': v for k, v in accum_losses.items()}
            log_dict['train/grad_norm'] = grad_norm.item()
            log_dict['train/grad_scale'] = scaler.get_scale()
            log_dict['train/lr'] = scheduler.get_last_lr()[0]
            log_dict['train/step'] = optim_step
            t_compute.synchronize()
            ms_data = t_start.elapsed_time(t_data)
            ms_compute = t_data.elapsed_time(t_compute)
            ms_total = t_start.elapsed_time(t_compute)
            log_dict['train/time_data'] = ms_data / 1000
            log_dict['train/time_compute'] = ms_compute / 1000
            print(f'[{_ts()}] step={optim_step} data={ms_data / 1000:.1f}s compute={ms_compute / 1000:.1f}s total={ms_total / 1000:.1f}s')
            if compile_baseline is not None:
                current_graphs = torch._dynamo.utils.counters['stats']['unique_graphs']
                if current_graphs > compile_baseline:
                    print(f'WARNING: {current_graphs - compile_baseline} new compilations since warmup')
                    compile_baseline = current_graphs
            if logger:
                logger.log(log_dict, step=optim_step)
                for da_key, count in sorted(da_step_counts.items(), key=lambda x: (x[0] is None, x[0])):
                    logger.log({f'train/da_steps/{da_key}': count}, step=optim_step)
            else:
                print(f'step={optim_step}', {k: f'{v:.4f}' for k, v in log_dict.items()})

        # Reconstruction visualization (all ranks save 1-2 samples)
        if optim_step % cfg.viz_every == 0:
            with torch.no_grad():
                save_reconstruction(save_dir, optim_step, rank, batch, output.recon.detach(), da_enc, da_dec)

        # Free memory
        del x, output, losses, loss

        # Save periodic checkpoint
        if is_main and optim_step % cfg.save_every == 0:
            _save_checkpoint(
                save_dir, optim_step, orig_model, ema_shadow,
                optimizer, scheduler,
                scaler=scaler,
                wandb_id=wandb.run.id if logger else None,
                config=vars(cfg),
            )

    # Final save
    if is_main:
        final_ckpt = save_dir / 'checkpoints' / f'checkpoint-{cfg.steps}.pt'
        if final_ckpt.exists():
            print(f'Final checkpoint already saved by periodic save: {final_ckpt}')
        else:
            _save_checkpoint(
                save_dir, cfg.steps, orig_model, ema_shadow,
                optimizer, scheduler,
                scaler=scaler,
                wandb_id=wandb.run.id if logger else None,
                config=vars(cfg),
            )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
