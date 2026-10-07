"""UCPT (Unified Continued Pretraining) training: JEPA SSL + segmentation.

Usage (always launch via torchrun, even for single-GPU debugging):
    torchrun --nproc_per_node=N -m pumit.ucpt.train --config configs/ucpt/train.yaml [overrides]
"""

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pumit.compile_cache import (
    BackgroundCompileCachePublisher,
    extract_compile_cache,
)
from pumit.train_utils import (
    BackgroundCheckpointer,
    setup_distributed,
    setup_logging,
)
from pumit.text_prompt import (
    build_segmentation_prompts,
    class_captions_sha256,
    load_class_captions,
)
from pumit.ucpt.model import UCPTModel
from pumit.ucpt.seg.text_encoding import TextEmbeddingCache
from .config import parse_config
from .data import build_replay_dataset
from .model_setup import build_model, prewarm_seg_decoder
from .optim import build_param_groups, make_optimizer_scheduler
from .state import (
    RunLock,
    make_checkpoint_state,
    prepare_run,
    restore_checkpoint,
)

_COMPILE_CACHE_ARCHIVE_MIN_STEP = 16
_COMPILE_CACHE_ARCHIVE_ZSTD_THREADS = 8


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _inductor_compile_threads(local_world_size: int) -> int:
    if local_world_size < 1:
        raise ValueError(f'LOCAL_WORLD_SIZE must be positive, got {local_world_size}')
    return max(1, 32 // local_world_size)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compile_cache_archive_due(completed_step: int) -> bool:
    return (
        completed_step >= _COMPILE_CACHE_ARCHIVE_MIN_STEP
        and completed_step & (completed_step - 1) == 0
        and (completed_step.bit_length() - 1) % 2 == 0
    )


def _compile_cache_live_dir(cache_archive: str | Path) -> Path:
    """Return an archive-specific shared tmpfs directory."""
    archive_key = hashlib.sha256(str(Path(cache_archive).resolve()).encode()).hexdigest()[:16]
    root = Path(os.environ.get('UCPT_COMPILE_CACHE_ROOT', '/dev/shm/ucpt_compile_cache'))
    return root / archive_key


def _node_leader_ranks(world_size: int, local_world_size: int) -> list[int]:
    if world_size < 1 or local_world_size < 1:
        raise ValueError('world_size and local_world_size must be positive')
    if world_size % local_world_size:
        raise ValueError(
            f'world_size ({world_size}) must be divisible by '
            f'LOCAL_WORLD_SIZE ({local_world_size})'
        )
    return list(range(0, world_size, local_world_size))


def _log_input_stalls(iterable, *, rank: int, start_step: int, threshold: float):
    """Yield items while reporting unusually slow iterator fetches on every rank."""
    iterator = iter(iterable)
    step = start_step
    while True:
        started = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            return
        wait = time.perf_counter() - started
        if wait >= threshold:
            print(f'[{_ts()}] [input-stall] rank={rank} step={step} wait={wait:.2f}s', flush=True)
        step += 1
        yield item


def main():
    cfg, no_compile, cache_archive = parse_config()
    resolved_config = asdict(cfg)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if cfg.run.nccl_high_priority:
        os.environ.setdefault('TORCH_NCCL_HIGH_PRIORITY', '1')
    rank, world_size, device = setup_distributed()
    is_main = rank == 0
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', world_size))
    if cfg.data.virtual_lanes < 1:
        raise ValueError(f'virtual_lanes must be positive, got {cfg.data.virtual_lanes}')
    assert cfg.data.virtual_lanes % world_size == 0, (
        f'virtual_lanes ({cfg.data.virtual_lanes}) must be divisible by '
        f'DDP world_size ({world_size})'
    )
    os.environ.setdefault(
        'TORCHINDUCTOR_COMPILE_THREADS',
        str(_inductor_compile_threads(local_world_size)),
    )

    def log_main(*args, **kwargs):
        if is_main:
            print(f'[{_ts()}]', *args, **kwargs)

    # Seed BEFORE build_model (rank-0 new-head init reproducible + broadcast via DDP).
    torch.manual_seed(cfg.run.seed)
    log_main(f'Seeded torch RNG with seed={cfg.run.seed}')
    log_main(f'NCCL high-priority streams: {os.environ.get("TORCH_NCCL_HIGH_PRIORITY", "0")}')
    log_main(f'CUDA allocator config: {os.environ["PYTORCH_CUDA_ALLOC_CONF"]}')
    log_main(f'Inductor compile workers per rank: {os.environ["TORCHINDUCTOR_COMPILE_THREADS"]}')

    # NUMA pin.
    from pumit.numa import pin_to_gpu_numa
    shared_numa = os.environ.get('UCPT_SHARED_NUMA', str(int(cfg.run.numa_shared))) == '1'
    affinity = pin_to_gpu_numa(shared=shared_numa)
    if affinity:
        mode = 'shared' if shared_numa else 'exclusive'
        print(
            f'[{_ts()}] [rank={rank} local_rank={local_rank}] NUMA: pinned to '
            f'node {affinity.numa_node}, {len(affinity.physical_cores)} '
            f'physical cores ({mode})',
            flush=True,
        )
    else:
        print(
            f'[{_ts()}] [rank={rank} local_rank={local_rank}] NUMA: pinning skipped',
            flush=True,
        )

    # Variable-length packing: different shapes every step; benchmark=False re-runs cuDNN autotuning every step.
    torch.backends.cudnn.benchmark = False

    save_dir = Path(cfg.run.save_dir).resolve()
    run_lock = None
    startup_list = [None]
    if is_main:
        import wandb

        captions = load_class_captions(cfg.data.class_captions_dir)
        TextEmbeddingCache(cfg.data.text_cache_path).require_prompts(
            build_segmentation_prompts(captions)
        )
        resolved_config['data_artifacts'] = {
            'class_captions_sha256': class_captions_sha256(captions),
            'text_cache_sha256': _sha256_file(cfg.data.text_cache_path),
        }
        run_lock = RunLock.acquire(save_dir)
        startup_list[0] = prepare_run(
            save_dir,
            resolved_config,
            world_size,
            wandb_id=wandb.util.generate_id(),
        )
    if world_size > 1:
        dist.broadcast_object_list(startup_list, src=0)
    startup = startup_list[0]
    assert startup is not None

    if is_main:
        setup_logging(save_dir)
        log_main(f'Output dir: {save_dir}')
    if startup.checkpoint_step > cfg.optim.steps:
        raise ValueError(
            f'latest checkpoint step {startup.checkpoint_step} exceeds target steps {cfg.optim.steps}',
        )
    if startup.checkpoint_step == cfg.optim.steps:
        log_main(f'Run already complete at step {cfg.optim.steps}')
        if world_size > 1:
            dist.destroy_process_group()
        if run_lock is not None:
            run_lock.close()
        return

    # Checkpoints are authoritative. W&B may retain a short orphan tail after rollback.
    logger = None
    if is_main:
        import wandb

        logger = wandb.init(
            entity=cfg.run.wandb_entity,
            project=cfg.run.wandb_project,
            dir=str(save_dir),
            id=startup.manifest.wandb_id,
            name=cfg.run.wandb_name,
            config=resolved_config,
            resume=startup.wandb_resume,
        )
        if logger.id != startup.manifest.wandb_id:
            raise RuntimeError(
                f'W&B run id mismatch: expected {startup.manifest.wandb_id}, got {logger.id}',
            )
        if logger.offline:
            raise RuntimeError('UCPT automatic resume requires W&B online mode')
        if startup.checkpoint is not None:
            log_main(
                f'Auto-resuming checkpoint step {startup.checkpoint_step}; '
                'W&B history after that step is non-authoritative',
            )

    # Model.
    model = build_model(cfg.model, device)
    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log_main(f'Model: {n_train / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total params')

    # Param groups (6-way) + optimizer + scheduler.
    param_groups = build_param_groups(model, cfg.optim)
    if is_main:
        counts = {g['group_name']: len(g['params']) for g in param_groups}
        log_main(f'Param groups: {counts}')
    optimizer, scheduler = make_optimizer_scheduler(cfg.optim, param_groups)

    # Resume (strict, before compile).
    start_step = startup.checkpoint_step
    if startup.checkpoint is not None:
        restore_checkpoint(
            startup.checkpoint,
            checkpoint_step=start_step,
            run_id=startup.manifest.run_id,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        log_main(f'Resumed from step {start_step}')

    # Establish the exact training/eval modes before compile prewarming. UCPTModel.train() keeps EMA twins in eval mode.
    model.train()

    # Data.
    dataset = build_replay_dataset(
        cfg.data,
        rank=rank,
        world_size=world_size,
        start_offset=start_step,
        n_prefix=model.vit.n_prefix,
        augment_threads=cfg.run.augment_threads,
        tcmalloc_release_every=cfg.run.tcmalloc_release_every,
    )
    log_main(
        f'UCPTReplayDataset: {len(dataset)} steps, {dataset.virtual_lanes} virtual lanes, '
        f'{len(dataset.my_lane_ids)} lanes/rank'
    )
    min_dataset_len = torch.tensor(len(dataset), device=device, dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(min_dataset_len, op=dist.ReduceOp.MIN)
    remaining_steps = cfg.optim.steps - start_step
    if remaining_steps > min_dataset_len.item():
        raise ValueError(
            f'replay stream is too short: need {remaining_steps} steps after resume, '
            f'shortest rank has {min_dataset_len.item()}',
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.run.num_workers,
        persistent_workers=cfg.run.num_workers > 0,
        pin_memory=True,
        prefetch_factor=2,
    )
    # Start workers before compile prewarming so data loading does not delay the first measured training step.
    loader_iter = iter(loader)
    primed_batches = []
    cuda_prefetcher = None

    # Compile stable modules after resume loading and before DDP. Segmentation orchestration stays eager; only its PixelDecoder is
    # compiled.
    cache_dir = None
    cache_publisher = None
    if not no_compile:
        # DDP graph splitting fails on float outputs during drop-path recompilation; keep the standard DDP reducer.
        torch._dynamo.config.optimize_ddp = False
        if cache_archive is not None:
            # Extract the shared tmpfs cache before torch.compile reads its environment variables.
            # /dev/shm is node-local, so every node's local rank 0 extracts before the global barrier.
            archive_status = [
                Path(cache_archive).exists() if is_main else None
            ]
            if dist.is_initialized():
                dist.broadcast_object_list(archive_status, src=0)
            archive_exists = archive_status[0]
            assert archive_exists is not None
            cache_dir, _ = extract_compile_cache(
                cache_archive,
                _compile_cache_live_dir(cache_archive),
                rank=local_rank,
                archive_exists=archive_exists,
            )
        log_main('Compiling ViT, teacher ViT, fusion encoder, PixelDecoder, SSL post-processing, and segmentation loss...')
        model.vit.compile(dynamic=True)
        model.teacher_vit.compile(dynamic=True)
        torch._dynamo.config.recompile_limit = 128
        model.seg.fusion.compile(dynamic=True, fullgraph=True)
        model.seg.head.pixel_decoder.compile(dynamic=True, fullgraph=True)
        model.ssl_post = torch.compile(model.ssl_post, dynamic=True)
        model.seg_loss_sample = torch.compile(model.seg_loss_sample, dynamic=True, fullgraph=True)
        # Prewarm the segmentation decoder's DA, size, and singleton-K guards before DDP synchronization.
        t0 = time.perf_counter()
        prewarm_seg_decoder(
            model,
            cfg.data.text_cache_path,
            cfg.model.text_embed_dim,
            device,
        )
        log_main(f'Seg decoder compiled + pre-warmed in {time.perf_counter() - t0:.0f}s')

        # Compile the complete forward/backward contract on real rank-local shapes before DDP installs gradient hooks.
        # The batches are retained and replayed as the first training batches, so prewarming does not alter data order.
        n_prewarm = min(cfg.run.compile_prewarm_batches, cfg.optim.steps - start_step)
        if n_prewarm < 0:
            raise ValueError('compile_prewarm_batches must be non-negative')
        if n_prewarm > 0 and torch.cuda.is_available():
            from pumit.prefetcher import CUDAPrefetcher

            cuda_prefetcher = CUDAPrefetcher(loader_iter, device)
            primed_batches = [next(cuda_prefetcher) for _ in range(n_prewarm)]
            t0 = time.perf_counter()
            fork_devices = [device]
            with torch.random.fork_rng(devices=fork_devices):
                for batch in primed_batches:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out = model(batch, reduce_metrics=False)
                    out.loss.backward()
                    model.zero_grad(set_to_none=True)
                    del out
            torch.cuda.synchronize(device)
            log_main(
                f'Full model pre-warmed on {n_prewarm} replayed batches in '
                f'{time.perf_counter() - t0:.0f}s',
            )

        # Each node leader snapshots its prewarm cache in the background. Global rank 0 builds the canonical union on the same
        # background worker; this run continues from each node's already-warm live cache.
        if cache_dir is not None:
            if dist.is_initialized():
                dist.barrier()
            session_list = [uuid.uuid4().hex if is_main else None]
            if dist.is_initialized():
                dist.broadcast_object_list(session_list, src=0)
            session = session_list[0]
            assert session is not None
            if local_rank == 0:
                print(
                    f'[{_ts()}] [compile_cache] global rank {rank} publishing '
                    f'node-local prewarm part for session {session}',
                    flush=True,
                )
                cache_publisher = BackgroundCompileCachePublisher(
                    cache_archive,
                    cache_dir,
                    session=session,
                    node_leader_ranks=_node_leader_ranks(
                        world_size,
                        local_world_size,
                    ),
                    rank=rank,
                    zstd_threads=_COMPILE_CACHE_ARCHIVE_ZSTD_THREADS,
                )
                cache_publisher.start()

    if world_size > 1:
        ddp_bucket_cap_mb = float(os.environ.get('UCPT_DDP_BUCKET_CAP_MB', cfg.run.ddp_bucket_cap_mb))
        if ddp_bucket_cap_mb <= 0:
            raise ValueError('UCPT_DDP_BUCKET_CAP_MB must be positive')
        log_main(f'DDP bucket cap: {ddp_bucket_cap_mb:g} MiB')
        model = DDP(
            model, device_ids=[device], gradient_as_bucket_view=True,
            batched_grad_copy=True,
            find_unused_parameters=False, static_graph=False,
            bucket_cap_mb=ddp_bucket_cap_mb,
        )

    raw_model: UCPTModel = model
    if isinstance(raw_model, DDP):
        raw_model = raw_model.module
    # UCPT compiles child modules rather than the top-level model, so this is normally a no-op.
    if hasattr(raw_model, '_orig_mod'):
        raw_model = raw_model._orig_mod

    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() \
        else torch.amp.autocast('cpu', dtype=torch.float32)

    if torch.cuda.is_available():
        if cuda_prefetcher is None:
            from pumit.prefetcher import CUDAPrefetcher
            cuda_prefetcher = CUDAPrefetcher(loader_iter, device)
        prefetcher = chain(primed_batches, cuda_prefetcher)
    else:
        prefetcher = loader_iter
    if input_stall_threshold := os.environ.get('UCPT_INPUT_STALL_THRESHOLD'):
        prefetcher = _log_input_stalls(
            prefetcher,
            rank=rank,
            start_step=start_step,
            threshold=float(input_stall_threshold),
        )
    # loader_iter forked DataLoader workers before compile, so they do not inherit the callback / open file /
    # disabled-GC / frozen generation installed below.
    from pumit.ucpt.train.gc_control import GCController, maybe_install
    gc_probe = maybe_install(save_dir, rank)
    gc_controller = GCController(cfg.run.gc_interval) if cfg.run.gc_interval > 0 else None
    if gc_controller is not None:
        gc_controller.setup()
        log_main(f'GCController: freeze+disable, synchronized collect every {cfg.run.gc_interval} steps')
    if is_main:
        from tqdm import tqdm
        prefetcher = tqdm(
            prefetcher,
            initial=start_step,
            total=cfg.optim.steps,
            desc='Rollout progress',
            dynamic_ncols=True,
        )

    t_step = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t_mid = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t_step_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    checkpointer = (
        BackgroundCheckpointer(save_dir, permanent_every=cfg.run.save_every)
        if is_main else None
    )
    # Minimal rank-0 runtime trace after an unrecorded settling window. The skip is not proof of steady state;
    # inspect the trace for recompilation before treating it as such. Shape, stack, and memory collection remain
    # disabled because they add substantial overhead and are unnecessary for the first-pass kernel/NCCL timeline.
    prof = None
    if os.environ.get('UCPT_STEP_PROFILE') and is_main and torch.cuda.is_available():
        profile_skip = int(os.environ.get('UCPT_STEP_PROFILE_SKIP', '30'))
        profile_active = int(os.environ.get('UCPT_STEP_PROFILE_ACTIVE', '3'))
        trace_path = save_dir / 'torch-profile-rank0.json'
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                skip_first=profile_skip, wait=0, warmup=1, active=profile_active, repeat=1,
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()
        log_main(
            f'UCPT_STEP_PROFILE: skip={profile_skip}, warmup=1, active={profile_active}, '
            f'trace={trace_path}',
        )

    # Optional low-overhead acceptance trace. Consecutive CUDA event timestamps include device work and any device-idle
    # gap while the host waits for input; unlike per-step synchronize(), buffering events does not serialize the loop.
    timing_events = None
    timing_steps = []
    if os.environ.get('UCPT_STEP_TIMINGS') and torch.cuda.is_available():
        timing_origin = torch.cuda.Event(enable_timing=True)
        timing_origin.record()
        timing_events = [timing_origin]
        log_main('UCPT_STEP_TIMINGS: recording per-rank CUDA timeline intervals')

    for step, batch in enumerate(prefetcher, start=start_step):
        if checkpointer is not None:
            checkpointer.raise_if_failed()
        if cache_publisher is not None:
            cache_publisher.raise_if_failed()
        if gc_probe is not None:
            gc_probe.set_step(step)
        if t_step is not None:
            t_step.record()
        optimizer.zero_grad()

        # Training requires both sample kinds for DDP grad coverage; the model still permits single-kind
        # evaluation batches.
        assert batch.total_seg_len > 0, (
            f'zero-labeled batch at step {step} (stream={cfg.data.stream_dir}); '
            f'packer invariant violated'
        )
        assert batch.n_ssl_patches > 0, (
            f'zero-unlabeled batch at step {step} (stream={cfg.data.stream_dir}); '
            f'packer invariant violated'
        )

        log_due = step % cfg.run.log_every == 0
        with autocast_ctx:
            out = model(batch, reduce_metrics=log_due)
        # t_mid splits the step into forward and backward+opt halves for coarse attribution.
        # Forward waits on the async count all-reduce and, on logging steps, runs the detached-metric all-reduce, so its cross-rank spread cannot distinguish globally slow work from a straggler.
        if t_mid is not None:
            t_mid.record()

        out.loss.backward()

        # One-time DDP grad-sync sanity check (warn, not raise — bf16+cudagraph near-misses).
        if world_size > 1 and step == start_step:
            sync_targets = [('vit_qkv', raw_model.vit.layer[0].attention.q_proj.weight)]
            for tag, param in sync_targets:
                if param.grad is None or param.grad.numel() == 0:
                    log_main(f'[grad-sync] {tag}: NO GRAD (skipped)')
                    continue
                grad_buf = param.grad.detach().contiguous()
                others = [torch.zeros_like(grad_buf) for _ in range(world_size)]
                dist.all_gather(others, grad_buf)
                maxdiff = max(
                    (others[i] - grad_buf).abs().max().item()
                    for i in range(world_size) if i != rank
                )
                log_main(
                    f'[grad-sync] {tag}: rank0->rank_k max|diff| = {maxdiff:.2e} '
                    f'({"OK" if maxdiff < 1e-5 else "WARNING: not reduced"})',
                )

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.grad_clip)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            raw_model.ema_update()

        if t_step_end is not None:
            t_step_end.record()

        completed = step + 1

        # Manual GC: on the completed-step interval, barrier so all ranks start the collect together, then synchronized full
        # collect. This aligns collection with rotating checkpoints when their intervals match.
        if gc_controller is not None and gc_controller.due(completed):
            if world_size > 1:
                dist.barrier()
            gc_controller.collect()

        if prof is not None:
            prof.step()

        # Logging (per-component losses + per-group LRs + labeled count). All ranks compute
        # their step time and all_gather, so DDP straggler vs global-slow is distinguishable:
        # train/time_step = rank0, train/time_step_max = slowest rank (the real DDP step cost),
        # train/time_step_mean = cross-rank average.
        if log_due:
            if t_step_end is not None:
                t_step_end.synchronize()
                ms_step = t_step.elapsed_time(t_step_end)
                ms_fwd = t_step.elapsed_time(t_mid)
            else:
                ms_step = ms_fwd = 0.0
            if world_size > 1:
                # Pack timing and labeled length into the existing all_gather.
                # Both timing halves contain collectives (forward waits on the
                # count reduce and, on logging steps, the metric reduce), so
                # their spreads are coarse balance telemetry.
                gathered = [torch.zeros(3, device=device) for _ in range(world_size)]
                dist.all_gather(
                    gathered,
                    torch.tensor(
                        [ms_step, ms_fwd, batch.total_seg_len],
                        device=device,
                    ),
                )
                step_vals = [g[0].item() for g in gathered]
                fwd_vals = [g[1].item() for g in gathered]
                total_seg_len = sum(g[2].item() for g in gathered)
                ms_max, ms_mean = max(step_vals), sum(step_vals) / len(step_vals)
                fwd_max, fwd_mean = max(fwd_vals), sum(fwd_vals) / len(fwd_vals)
            else:
                ms_max = ms_mean = ms_step
                fwd_max = fwd_mean = ms_fwd
                total_seg_len = float(batch.total_seg_len)
            if is_main:
                last_lrs = scheduler.get_last_lr()
                lr_by_group = {g['group_name']: lr for g, lr in zip(param_groups, last_lrs)}
                lr_vit = max(lr for name, lr in lr_by_group.items() if name.startswith('vit'))
                log_dict = {
                    'train/recon_loss': out.recon_loss.item(),
                    'train/image_distill_loss': out.image_distill_loss.item(),
                    'train/patch_distill_loss': out.patch_distill_loss.item(),
                    'train/seg_loss': out.seg_loss.item(),
                    'train/seg_focal_loss': out.seg_focal_loss.item(),
                    'train/seg_dice_loss': out.seg_dice_loss.item(),
                    'train/loss': out.loss.item(),
                    'train/grad_norm': grad_norm.item(),
                    'train/lr_vit': lr_vit,
                    'train/lr_ssl': lr_by_group.get('ssl', 0.0),
                    'train/lr_seg': lr_by_group.get('seg', 0.0),
                    'train/total_seg_len': total_seg_len,
                    'train/step': step,
                    'train/teacher_momentum': cfg.model.teacher_momentum,
                    'train/artifact_momentum': cfg.model.artifact_momentum,
                    'train/time_step': ms_step / 1000,
                    'train/time_step_max': ms_max / 1000,
                    'train/time_step_mean': ms_mean / 1000,
                    'train/fwd_max': fwd_max / 1000,
                    'train/fwd_mean': fwd_mean / 1000,
                }
                if logger:
                    logger.log(log_dict, step=step)
                # tmax/tmean = full-step spread, tfmax/tfmean = forward-half spread. Both
                # are collective-masked (see comment at t_mid.record) — balance telemetry only.
                log_main(
                    f'step={step} loss={out.loss.item():.3f} seg={out.seg_loss.item():.3f} '
                    f'grad_norm={grad_norm.item():.2f} lr_vit={lr_vit:.2e} '
                    f'tmax={ms_max / 1000:.2f} tmean={ms_mean / 1000:.2f} '
                    f'tfmax={fwd_max / 1000:.2f} tfmean={fwd_mean / 1000:.2f}',
                )

        # Checkpoint (resume-only; export_artifact removed — downstream slices the checkpoint).
        # Two decoupled tiers: permanent (save_every, kept forever) and rotating
        # (save_rotating_every, keep-2, background-written). A step on both cadences
        # saves once, as permanent.
        permanent_due = completed % cfg.run.save_every == 0
        rotating_due = (cfg.run.save_rotating_every > 0
                        and completed % cfg.run.save_rotating_every == 0)
        if is_main and (permanent_due or rotating_due):
            state = make_checkpoint_state(
                run_id=startup.manifest.run_id,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed,
                config=resolved_config,
            )
            checkpointer.save(state, rotating=not permanent_due)
            log_main(
                f'Enqueued {"permanent" if permanent_due else "rotating"} '
                f'checkpoint at step {completed}',
            )

        if is_main and cache_publisher is not None and _compile_cache_archive_due(completed):
            if cache_publisher.submit():
                print(
                    f'[{_ts()}] Enqueued compile-cache snapshot at step '
                    f'{completed} on global rank {rank}',
                    flush=True,
                )

        if timing_events is not None:
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_end.record()
            timing_events.append(timing_end)
            timing_steps.append(step)

        # --optim.steps below the dataset length must terminate here: the loader
        # iterates the full replay stream and would otherwise keep going.
        if completed >= cfg.optim.steps:
            break

    if prof is not None:
        prof.stop()
        log_main('UCPT_STEP_PROFILE: trace exported to run dir')
    if timing_events is not None:
        timing_events[-1].synchronize()
        intervals = [
            start.elapsed_time(end) / 1000
            for start, end in zip(timing_events, timing_events[1:])
        ]
        timing_path = save_dir / f'step-times-rank{rank}.json'
        timing_path.write_text(json.dumps({'steps': timing_steps, 'seconds': intervals}))
        log_main(f'UCPT_STEP_TIMINGS: wrote {len(intervals)} intervals per rank')

    # Final save, unless the same completed step was already enqueued by either checkpoint cadence.
    if is_main:
        final_completed = step + 1
        final_already_enqueued = (
            final_completed % cfg.run.save_every == 0
            or (cfg.run.save_rotating_every > 0
                and final_completed % cfg.run.save_rotating_every == 0)
        )
        if not final_already_enqueued:
            state = make_checkpoint_state(
                run_id=startup.manifest.run_id,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=final_completed,
                config=resolved_config,
            )
            checkpointer.save(state, rotating=False)
            log_main(f'Enqueued final checkpoint at step {final_completed}')
        checkpointer.close()  # flush pending writes before teardown
    if cache_publisher is not None:
        cache_publisher.close()

    if logger is not None:
        logger.finish()

    if world_size > 1:
        dist.destroy_process_group()

    if gc_probe is not None:
        gc_probe.close()

    if run_lock is not None:
        run_lock.close()

if __name__ == '__main__':
    main()
