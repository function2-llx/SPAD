"""End-to-end finetune of a backbone encoder + linear head on MedMNIST.

Usage:
    pixi run -e cls python -m pumit.downstream.cls.finetune \
        --backbone dinov3 --weights pretrained/dinov3-vitl16/model.safetensors \
        --vit-config configs/downstream/cls/vit_l.yaml --img-size 192 \
        --datasets configs/downstream/cls/datasets.yaml \
        --flag nodulemnist3d --num-updates 500 --eval-every 50 \
        --lr 1e-3 --lr-backbone 1e-5 \
        --out outputs/downstream/cls/dinov3/finetune/nodulemnist3d.json

The head input is the backbone's native global readout: each adapter returns its model's own
official global representation as the first element of forward(), so the comparison is between
the representations the models were built with, not a pooling scheme imposed here.
"""
from __future__ import annotations

import argparse
import dataclasses
import random
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import os

import orjson
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from pumit.downstream.cached_probe import build_probe_head

from .config import load_dataset_specs, load_vit_config
from .data import DEFAULT_DATA_ROOT, build_arrays
from .augment import FLIP_VIEWS, flip_view, flips_enabled_for, random_flip_3d
from .optim import build_param_groups, build_warmup_cosine, warmup_steps_for
from .registry import BACKBONES
from .metrics import evaluate
from .task import CLASSIFICATION_SELECTION, classification_objective
from .checkpointing import capture_rng_state, restore_rng_state, save_checkpoint, load_checkpoint, cpu_snapshot


def _rank_world() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _evaluate_on_rank_zero(logits, labels, spec, split, data_root, evaluate_fn=evaluate):
    rank, world = _rank_world()
    metrics = evaluate_fn(
        logits, labels.numpy(), spec.flag, spec.size, split, root=data_root,
    ) if rank == 0 else None
    if world > 1:
        payload = [metrics]
        dist.broadcast_object_list(payload, src=0)
        metrics = payload[0]
    return metrics


def seed_process(seed: int) -> None:
    """Seed model/head initialization while the training order uses its own generator."""
    if seed < 0:
        raise ValueError(f'seed must be non-negative, got {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class FinetuneRun:
    """Validation-selected state and history from one finetune run."""

    best_epoch: int
    best_metrics: dict[str, float]
    history: list[dict]


@torch.no_grad()
def observed_token_count(encoder, spec, transform_batch, device, data_root) -> int:
    """Sequence length the encoder actually produces, measured from one real forward pass.

    The requested --img-size cannot stand in for this: several backbones resize to a fixed shape
    of their own, so only an observed count is evidence that two arms ran at the same token
    budget. Uses a single sample and restores the caller's train/eval mode.
    """
    images, _ = build_arrays(spec.flag, spec.size, 'val', root=data_root)
    was_training = encoder.training
    encoder.eval()
    xb = {k: v.to(device) for k, v in transform_batch(images[:1], is_3d=spec.is_3d).items()}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, patch_tokens = encoder(**xb)
    encoder.train(was_training)
    return int(patch_tokens.shape[1])


def load_probe_head_state(path: str, head_type: str, flag: str, seed: int) -> dict[str, Tensor]:
    """Read one seed's val-selected probe head from a train_probe --save-heads bundle.

    Fails loudly on a head-type, dataset or seed mismatch: silently finetuning a randomly
    initialized head would make an LP-FT arm indistinguishable from the plain protocol.
    """
    bundle = torch.load(path, weights_only=True)
    if bundle['head'] != head_type:
        raise ValueError(f'{path} holds {bundle["head"]!r} heads, --head is {head_type!r}')
    if bundle['flag'] != flag:
        raise ValueError(f'{path} was fit on {bundle["flag"]!r}, --flag is {flag!r}')
    states = bundle['state_dicts']
    if seed >= len(states):
        raise ValueError(f'{path} holds {len(states)} seeds, --seed is {seed}')
    return states[seed]


def build_trainable_encoder(args, device: str):
    """Dispatch to the chosen backbone's create_model (RoPE aug off for repro eval).

    Each optional CLI flag is forwarded only when given, so a backbone that does not declare it
    raises TypeError instead of silently discarding it: passing --img-size to a backbone whose
    input size is fixed used to be a no-op that still recorded the requested value.
    """
    bb = BACKBONES[args.backbone]
    extra = {}
    if args.vit_config is not None:
        cfg = load_vit_config(args.vit_config)
        if args.drop_path is not None:
            # stochastic depth is a train-time regularizer with no parameters, so overriding the
            # config's 0.0 here does not conflict with weights pretrained without it
            cfg = dataclasses.replace(cfg, drop_path_rate=args.drop_path)
        extra['vit_config'] = cfg
    elif args.drop_path is not None:
        extra['drop_path_rate'] = args.drop_path
    if args.img_size is not None:
        extra['img_size'] = args.img_size
    if args.weights is not None:
        extra['weights'] = args.weights
    return bb.create_model(dims=args.dims, device=device, trainable=True, **extra)


def iter_batch_indices(n: int, batch_size: int, *, steps: int, generator: torch.Generator):
    """Yield `steps` full batches of indices from an endless stream of fresh permutations.

    The sampler cycles permutations rather than iterating epochs: a "pass over the data" has no
    special meaning here, so the training budget is a step count and every step is a full batch.
    The ragged tail of a permutation is carried into the next one instead of emitting a short
    batch, which keeps the shape static -- required for CUDA graphs under compile reduce-overhead.
    """
    if steps <= 0:
        raise ValueError(f'steps must be positive, got {steps}')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    pool = torch.empty(0, dtype=torch.long)
    for _ in range(steps):
        while pool.numel() < batch_size:
            pool = torch.cat([pool, torch.randperm(n, generator=generator)])
        yield pool[:batch_size]
        pool = pool[batch_size:]


def iter_eval_batches(n: int, batch_size: int):
    """Yield (indices, valid_count) covering 0..n-1 in fixed-size batches.

    Every batch has exactly `batch_size` entries: a short tail is padded by repeating the last
    index, and `valid_count` says how many leading rows are real. This keeps evaluation to ONE
    input shape, which matters under compile reduce-overhead -- each distinct shape captures its
    own CUDA graph with a permanently retained memory pool, and ragged tails added a pool per
    split until the GPU ran out (observed 56.5 GiB in private pools).
    """
    import numpy as np
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        idx = np.arange(start, stop)
        valid = len(idx)
        if valid < batch_size:                       # pad by repeating the final index
            idx = np.concatenate([idx, np.full(batch_size - valid, n - 1)])
        yield idx, valid


@torch.no_grad()
def eval_split(encoder, head, spec, split, device, batch_size, transform_batch,
               data_root=DEFAULT_DATA_ROOT, tta=False):
    """One ordered pass over `split`, returning (logits, labels) for metric computation.

    Evaluation only: a single deterministic pass in index order, no loss and no optimizer. The
    training path is `train_steps`, which draws from the cycling-permutation sampler instead.

    With `tta`, each batch is additionally evaluated under all 8 flip orientations and the softmax
    probabilities are averaged (nnU-Net's mirroring TTA). The returned tensor is then the log of
    those averaged probabilities, which the softmax in `logits_to_scores` inverts exactly, so the
    metric path is unchanged. The caller gates `tta` on `flips_enabled_for`: on organmnist3d a
    mirrored view contradicts the side-specific labels.
    """
    images, labels = build_arrays(spec.flag, spec.size, split, root=data_root)
    y = torch.from_numpy(labels)
    encoder.eval()
    head.eval()
    rank, world = _rank_world()
    # Unequal numbers of evaluation batches must not enter DDP forward collectives.
    for module in (encoder, head):
        if isinstance(module, DistributedDataParallel):
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)
    encoder = encoder.module if isinstance(encoder, DistributedDataParallel) else encoder
    head = head.module if isinstance(head, DistributedDataParallel) else head
    views = FLIP_VIEWS if tta else ((),)
    logits_all = []
    indices_all = []
    for batch_index, (idx, valid) in enumerate(iter_eval_batches(images.shape[0], batch_size)):
        if batch_index % world != rank:
            continue
        raw = images[idx]
        outputs = []
        for axes in views:
            xb = transform_batch(flip_view(raw, axes), is_3d=spec.is_3d)
            xb = {k: v.to(device) for k, v in xb.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                global_features, _ = encoder(**xb)
                logits = head(global_features.float())
            outputs.append(logits.detach().float())
        if len(outputs) == 1:
            batch_logits = outputs[0]
        else:
            probs = torch.stack([torch.softmax(o, dim=1) for o in outputs]).mean(dim=0)
            batch_logits = probs.log()
        logits_all.append(batch_logits.cpu()[:valid])   # drop the padding rows
        indices_all.append(torch.from_numpy(idx[:valid]))
    if world == 1:
        return torch.cat(logits_all), y
    indices = torch.cat(indices_all) if indices_all else torch.empty(0, dtype=torch.long)
    logits = torch.cat(logits_all) if logits_all else torch.empty((0, spec.n_classes))
    shards = [None] * world
    dist.all_gather_object(shards, (indices, logits, y[indices]))
    indices = torch.cat([shard[0] for shard in shards])
    order = indices.argsort()
    return (
        torch.cat([shard[1] for shard in shards])[order],
        torch.cat([shard[2] for shard in shards])[order],
    )


def eval_boundaries(total_steps: int, eval_every: int) -> list[int]:
    """Step numbers at which to validate: every `eval_every` steps, plus always the final step.

    Handles a budget shorter than the interval (e.g. --num-updates 20 with --eval-every 50), which
    would otherwise produce no boundaries at all and skip validation entirely.
    """
    boundaries = list(range(eval_every, total_steps + 1, eval_every))
    if not boundaries or boundaries[-1] != total_steps:
        boundaries.append(total_steps)
    return boundaries


def train_steps(encoder, head, spec, device, batch_size, steps, transform_batch,
                opt, sched, shuffle_generator, data_root=DEFAULT_DATA_ROOT, augment=False,
                accum_steps=1, progress_bar=None):
    """Run `steps` optimizer updates with `batch_size` samples globally across ranks and micro-batches.

    DDP averages equal-size rank gradients; each micro-batch loss is divided only by `accum_steps`.
    Optimizer and scheduler advance once per global batch, independently of GPU count.
    """
    rank, world = _rank_world()
    if accum_steps <= 0 or batch_size % (world * accum_steps):
        raise ValueError(
            f'global batch_size {batch_size} must be divisible by world_size {world} '
            f'times accum_steps {accum_steps}',
        )
    if accum_steps > 1:
        # Stable .grad buffers, allocated before any compiled backward: under reduce-overhead the
        # CUDA-graph replay owns its gradient outputs and overwrites them on the next micro-batch,
        # so accumulating into a graph-owned .grad reads clobbered memory. Preallocated grads (and
        # zero_grad(set_to_none=False) below) keep accumulation in ordinary tensors.
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
    images, labels = build_arrays(spec.flag, spec.size, 'train', root=data_root)
    y = torch.from_numpy(labels)
    encoder.train(True)
    head.train(True)
    for batch in iter_batch_indices(images.shape[0], batch_size, steps=steps,
                                    generator=shuffle_generator):
        opt.zero_grad(set_to_none=accum_steps == 1)
        if world > 1:
            # Every rank consumes the same sampler/augmentation RNG before taking its own slice.
            global_raw = images[batch.numpy()]
            if augment:
                global_raw = random_flip_3d(global_raw, generator=shuffle_generator)
            local_size = batch_size // world
            start = rank * local_size
            raw_rank = global_raw[start:start + local_size]
            batch = batch[start:start + local_size]
        for micro_index, micro in enumerate(batch.chunk(accum_steps)):
            idx = micro.numpy()
            if world > 1:
                start = micro_index * len(micro)
                raw = raw_rank[start:start + len(micro)]
            else:
                raw = images[idx]
            if augment and world == 1:
                # flip the raw volume before the backbone's own normalization/resize
                raw = random_flip_3d(raw, generator=shuffle_generator)
            xb = transform_batch(raw, is_3d=spec.is_3d)
            xb = {k: v.to(device) for k, v in xb.items()}
            target = y[idx].to(device)
            with ExitStack() as sync:
                if micro_index + 1 < accum_steps:
                    for module in (encoder, head):
                        if isinstance(module, DistributedDataParallel):
                            sync.enter_context(module.no_sync())
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    global_features, _ = encoder(**xb)
                    logits = head(global_features.float())
                    loss = classification_objective(logits, target) / accum_steps
                loss.backward()
        opt.step()
        sched.step()
        if progress_bar is not None:
            progress_bar.update(1)


def _state_module(module):
    """Checkpoint model weights independently of DDP and compile wrappers."""
    while isinstance(module, (DistributedDataParallel, OptimizedModule)):
        module = module.module if isinstance(module, DistributedDataParallel) else module._orig_mod
    return module


def snapshot(encoder, head) -> dict:
    """CPU clone of encoder and head weights without execution-wrapper prefixes."""
    return {
        'encoder': cpu_snapshot(_state_module(encoder).state_dict()),
        'head': cpu_snapshot(_state_module(head).state_dict()),
    }


def fit(
    encoder,
    head,
    spec,
    *,
    total_steps: int,
    eval_every: int,
    eval_batch_size: int | None = None,
    augment: bool = False,
    tta: bool = False,
    accum_steps: int = 1,
    device: str,
    batch_size: int,
    transform_batch,
    opt,
    sched,
    data_root: str,
    evaluate_fn: Callable = evaluate,
    progress: bool = True,
    seed: int = 0,
    checkpoint_path: Path | None = None,
) -> FinetuneRun:
    """Train `total_steps` optimizer steps, validating every `eval_every` steps.

    A latest checkpoint is saved after each validation interval and automatically resumed when present.
    Saving at these boundaries also preserves the existing sampler's per-interval permutation stream.
    """
    if total_steps <= 0:
        raise ValueError(f'total_steps must be positive, got {total_steps}')
    if eval_every <= 0:
        raise ValueError(f'eval_every must be positive, got {eval_every}')
    if seed < 0:
        raise ValueError(f'seed must be non-negative, got {seed}')
    eval_batch_size = eval_batch_size or batch_size
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    best_epoch = -1
    best_metrics: dict[str, float] | None = None
    best_state: dict | None = None
    history: list[dict] = []
    boundaries = eval_boundaries(total_steps, eval_every)
    done = 0
    rank, world = _rank_world()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            state = load_checkpoint(checkpoint_path)
            _state_module(encoder).load_state_dict(state['model']['encoder'])
            _state_module(head).load_state_dict(state['model']['head'])
            opt.load_state_dict(state['optimizer'])
            sched.load_state_dict(state['scheduler'])
            done = state['step']
            best_epoch, best_metrics = state['best_epoch'], state['best_metrics']
            best_state, history = state['best_state'], state['history']
            if rank < len(state['rng_states']):
                restore_rng_state(state['rng_states'][rank], shuffle_generator, device)
            else:
                # Newly added ranks keep their seeded model RNG and join the restored data stream.
                shuffle_generator.set_state(state['rng_states'][0]['shuffle_generator'])
            if rank == 0:
                print(f'[resume] update {done}/{total_steps} from {checkpoint_path}', flush=True)
            del state
    remaining = [step for step in boundaries if step > done]
    with tqdm(
        total=total_steps, initial=done, desc='finetune', unit='update',
        ncols=80, disable=not progress or rank != 0,
    ) as progress_bar:
        for boundary in remaining:
            epoch = len(history)
            train_steps(
                encoder,
                head,
                spec,
                device,
                batch_size,
                boundary - done,
                transform_batch,
                opt,
                sched,
                shuffle_generator,
                data_root=data_root,
                augment=augment,
                accum_steps=accum_steps,
                progress_bar=progress_bar,
            )
            done = boundary
            val_logits, val_y = eval_split(
                encoder,
                head,
                spec,
                'val',
                device,
                eval_batch_size,
                transform_batch,
                data_root=data_root,
                tta=tta,
            )
            val_metrics = _evaluate_on_rank_zero(
                val_logits, val_y, spec, 'val', data_root, evaluate_fn,
            )
            test_logits, test_y = eval_split(
                encoder,
                head,
                spec,
                'test',
                device,
                eval_batch_size,
                transform_batch,
                data_root=data_root,
                tta=tta,
            )
            test_metrics = _evaluate_on_rank_zero(
                test_logits, test_y, spec, 'test', data_root, evaluate_fn,
            )
            history.append({
                'epoch': epoch,
                'step': boundary,
                **{f'val_{name}': value for name, value in val_metrics.items()},
                **{f'test_{name}': value for name, value in test_metrics.items()},
            })
            if CLASSIFICATION_SELECTION.is_better(val_metrics, best_metrics):
                best_epoch = epoch
                best_metrics = val_metrics
                best_state = snapshot(encoder, head)
            progress_bar.set_postfix(
                val_auc=f'{val_metrics["auc"]:.4f}',
                test_auc_diag=f'{test_metrics["auc"]:.4f}',
            )
            if checkpoint_path is not None:
                rng_state = capture_rng_state(shuffle_generator, device)
                rng_states = [None] * world if rank == 0 else None
                if world > 1:
                    dist.gather_object(rng_state, rng_states, dst=0)
                else:
                    rng_states = [rng_state]
                if rank == 0:
                    save_checkpoint(
                        checkpoint_path,
                        {
                            'step': done,
                            'model': best_state if best_epoch == epoch else snapshot(encoder, head),
                            'optimizer': opt.state_dict(),
                            'scheduler': sched.state_dict(),
                            'best_state': best_state,
                            'best_epoch': best_epoch,
                            'best_metrics': best_metrics,
                            'history': history,
                            'rng_states': rng_states,
                        },
                    )
                    print(f'[checkpoint] update {done}/{total_steps} -> {checkpoint_path}', flush=True)
                if world > 1:
                    dist.barrier()

    if best_metrics is None or best_state is None:
        raise RuntimeError('finetune completed without a validation-selected checkpoint')
    _state_module(encoder).load_state_dict(best_state['encoder'])
    _state_module(head).load_state_dict(best_state['head'])
    return FinetuneRun(best_epoch=best_epoch, best_metrics=best_metrics, history=history)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True, choices=sorted(BACKBONES),
                   help='backbone name (eva02-b, eva02-l, dinov3, ucpt, biomedclip, m3d)')
    p.add_argument('--weights', default=None,
                   help='weights path, for the backbones that load from a file (dinov3/ucpt '
                        'safetensors or ckpt, where omitting it means random init; unimiss and '
                        'unimiss-plus, where it is mandatory). The rest load a pinned hub '
                        'checkpoint internally and reject this flag')
    p.add_argument('--vit-config', default=None,
                   help='ViT config yaml (required for spad backbones dinov3/ucpt)')
    p.add_argument('--img-size', type=int, default=None,
                   help='resize edge, required by the backbones that expose one: dinov3/ucpt '
                        '(patch 16), eva02-b/l (patch 14), biomedclip (patch 16). It sets the '
                        'token grid -- 192 -> 12^3=1728 at patch 16, 168 -> 12^3 at patch 14 -- '
                        'and must be divisible by the patch size. 3dino/sam-med3d/unimiss/'
                        'unimiss-plus/m3d pin their own input and reject this flag')
    p.add_argument('--datasets', required=True)
    p.add_argument('--flag', required=True)
    p.add_argument('--epochs', type=int, default=10,
                   help='legacy budget: converted to steps as epochs * ceil(n_train/batch_size). '
                        'Prefer --num-updates, which is exact and identical across datasets')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='base LR: the from-scratch head, and the undecayed top of the backbone '
                        'when --lr-backbone is not given. Mirrors the seg side, where `slr` is the '
                        'base and `bblr` overrides the backbone.')
    p.add_argument('--lr-backbone', type=float, default=None,
                   help='override the backbone LR; defaults to --lr, which ties the head to the '
                        'top block as the MAE/BEiT recipes do')
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--num-updates', type=int, default=None,
                   help='fixed optimizer-step budget; overrides --epochs. The sampler cycles '
                        'fresh permutations, so this is an exact step count with no rounding '
                        'to dataset-sized boundaries -- identical schedule for every dataset')
    p.add_argument('--eval-every', type=int, default=50,
                   help='validate every N optimizer steps (uniform cadence across datasets, '
                        'unlike per-epoch validation which scales with dataset size)')
    p.add_argument('--warmup-fraction', type=float, default=0.05,
                   help='linear warmup as a fraction of total steps before cosine; 0 disables')
    p.add_argument('--tta', action='store_true',
                   help='average softmax probabilities over all 8 flip orientations at every '
                        'evaluation (nnU-Net mirroring TTA). Applies to validation as well as '
                        'test, so selection and reporting use the same inference procedure. '
                        'Auto-disabled where flips are (organmnist3d). Costs 8x eval time.')
    p.add_argument('--drop-path', type=float, default=None,
                   help='maximum stochastic-depth probability, linearly scaled across blocks; '
                        'passed through the ViT config or the backbone factory')
    p.add_argument('--layer-decay', type=float, default=1.0,
                   help='layer-wise LR decay in (0, 1]; 1.0 = flat encoder LR (default)')
    p.add_argument('--weight-decay-policy', choices=('all', 'vit_standard'),
                   default='vit_standard',
                   help="'vit_standard' (default) exempts norms, biases, layer-scale gammas "
                        'and cls/register/position tokens from weight decay, matching UCPT '
                        "pretraining and downstream/seg; 'all' decays everything (the "
                        'original cls behavior, kept for reproducing pre-2026-08 results)')
    p.add_argument(
        '--batch-size', type=int, default=32,
        help='global samples per optimizer update, across all DDP ranks and accumulation steps',
    )
    p.add_argument(
        '--accum-steps', type=int, default=1,
        help='micro-batches per rank per update; batch-size must be divisible by world-size * accum-steps',
    )
    p.add_argument('--augment', action='store_true',
                   help='random flips (p=0.5 per axis) on the training stream. Exact on the '
                        'cubic volumes; skipped for organmnist3d, whose labels encode '
                        'laterality (kidney/femur/lung right vs left)')
    p.add_argument(
        '--eval-batch-size', type=int, default=64,
        help='batch size for validation and test passes; also determines evaluation CUDA-graph pool size',
    )
    p.add_argument('--seed', type=int, default=0,
                   help='experiment seed; also defines the independent per-epoch shuffle stream')
    p.add_argument('--head', default='linear', choices=('linear', 'mlp'),
                   help='classification head on the native readout; mlp adds one GELU hidden '
                        'layer of width embed_dim (same builder as the frozen probe)')
    p.add_argument('--head-init', default=None,
                   help='LP-FT: initialize the head from a frozen-probe .heads.pt (written by '
                        'train_probe --save-heads), selecting the entry for --seed. The head type '
                        'and dataset must match, so the head starts fit to this backbone instead '
                        'of random')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                   help='persistent MedMNIST npz dir (offline load)')
    p.add_argument(
        '--compile-mode', default='reduce-overhead',
        choices=('reduce-overhead', 'default', 'max-autotune', 'off'),
        help='torch.compile mode for the encoder with static input shapes; off uses eager execution.',
    )
    return p


def resolve_lrs(args) -> tuple[float, float]:
    """(backbone LR, base LR). --lr-backbone defaults to --lr, tying the head to the top block."""
    return (args.lr if args.lr_backbone is None else args.lr_backbone), args.lr


def main() -> None:
    args = build_arg_parser().parse_args()
    launched = 'LOCAL_RANK' in os.environ
    if launched:
        if int(os.environ['WORLD_SIZE']) != int(os.environ['LOCAL_WORLD_SIZE']):
            raise ValueError('classification DDP supports a single node per experiment')
        if torch.device(args.device).type != 'cuda':
            raise ValueError('torchrun classification training requires CUDA')
        local_rank = int(os.environ['LOCAL_RANK'])
        args.device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', device_id=torch.device(args.device))
    try:
        if launched:
            from pumit.numa import pin_to_gpu_numa

            pin_to_gpu_numa()
        _run(args)
    finally:
        if launched and dist.is_initialized():
            dist.destroy_process_group()


def _run(args) -> None:
    rank, world = _rank_world()
    if args.accum_steps <= 0 or args.batch_size % (world * args.accum_steps):
        raise ValueError(
            f'global batch_size {args.batch_size} must be divisible by world_size {world} '
            f'times accum_steps {args.accum_steps}',
        )
    seed_process(args.seed)
    spec = load_dataset_specs(args.datasets)[args.flag]
    # a mirrored view contradicts organmnist3d's side-specific labels, exactly as for train flips
    tta = args.tta and flips_enabled_for(spec.flag)
    lr_backbone, lr_base = resolve_lrs(args)
    args.dims = 3 if spec.is_3d else 2
    transform_batch = BACKBONES[args.backbone].transform_batch
    encoder = build_trainable_encoder(args, args.device)
    # `encoder` stays the ORIGINAL module for optimizer/LLRD/state-dict purposes (LLRD introspects
    # .parameter_layers(), which torch.compile would hide behind _orig_mod); `forward_encoder` is
    # what the train/eval loops call. Parameters are shared, so grads flow to the same tensors.
    forward_encoder = encoder
    if args.compile_mode != 'off':
        forward_encoder = torch.compile(encoder, mode=args.compile_mode, dynamic=False)
    head = build_probe_head(int(encoder.embed_dim), spec.n_classes, args.head).to(args.device)
    if args.head_init is not None:
        head.load_state_dict(load_probe_head_state(args.head_init, args.head, spec.flag, args.seed))
    # measured before compile: --img-size records the request, this records what the model ran at
    tokens = observed_token_count(encoder, spec, transform_batch, args.device, args.data_root)

    n_train = build_arrays(spec.flag, spec.size, 'train', root=args.data_root)[0].shape[0]
    # An exact step budget when given; otherwise derive one from --epochs for back-compat.
    steps = (args.num_updates if args.num_updates is not None
             else args.epochs * ((n_train + args.batch_size - 1) // args.batch_size))
    opt = AdamW(
        build_param_groups(
            encoder, head,
            lr_encoder=lr_backbone,
            lr_head=lr_base,
            weight_decay=args.weight_decay,
            layer_decay=args.layer_decay,
            weight_decay_policy=args.weight_decay_policy,
        )
    )
    warmup = warmup_steps_for(steps, args.warmup_fraction)
    sched = build_warmup_cosine(opt, total_steps=steps, warmup_steps=warmup)

    if world > 1:
        device_ids = [torch.device(args.device).index]
        forward_encoder = DistributedDataParallel(
            forward_encoder, device_ids=device_ids, gradient_as_bucket_view=True,
        )
        head = DistributedDataParallel(head, device_ids=device_ids, gradient_as_bucket_view=True)
        # Share initialization, but use independent per-rank stochastic-depth draws.
        torch.manual_seed(args.seed + rank)

    run = fit(
        forward_encoder,
        head,
        spec,
        total_steps=steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        augment=args.augment and flips_enabled_for(spec.flag),
        tta=tta,
        accum_steps=args.accum_steps,
        device=args.device,
        batch_size=args.batch_size,
        transform_batch=transform_batch,
        opt=opt,
        sched=sched,
        data_root=args.data_root,
        seed=args.seed,
        checkpoint_path=Path(args.out).with_suffix('.checkpoint.pt'),
    )
    test_logits, test_y = eval_split(forward_encoder, head, spec, 'test', args.device,
                                     args.eval_batch_size,
                                     transform_batch, data_root=args.data_root, tta=tta)
    m = _evaluate_on_rank_zero(test_logits, test_y, spec, 'test', args.data_root)
    if rank != 0:
        return
    summary = {'flag': spec.flag, 'mode': 'finetune', 'readout': 'native',
               'head': args.head, 'head_init': args.head_init, 'protocol': 'finetune',
               'backbone': args.backbone, 'weights': args.weights, 'vit_config': args.vit_config,
               'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
               'world_size': world, 'local_batch_size': args.batch_size // world,
               'accum_steps': args.accum_steps,
               # resolved values, not the flags: the existing tables key on these two names
               'lr_encoder': lr_backbone, 'lr_head': lr_base,
               'weight_decay': args.weight_decay,
               'layer_decay': args.layer_decay, 'drop_path': args.drop_path,
               'weight_decay_policy': args.weight_decay_policy,
               'img_size': args.img_size,
               'tokens': tokens,
               'embed_dim': int(encoder.embed_dim),
               'compile_mode': args.compile_mode,
               'warmup_fraction': args.warmup_fraction,
               'warmup_steps': warmup,
               'total_steps': steps,
               'num_updates': args.num_updates,
               'eval_every': args.eval_every,
               'eval_batch_size': args.eval_batch_size,
               'augment': args.augment and flips_enabled_for(spec.flag), 'tta': tta,
               'best_val_auc': run.best_metrics['auc'], 'best_val_acc': run.best_metrics['acc'],
               'best_test_auc_diag': max(record['test_auc'] for record in run.history),
               'best_test_acc_diag': max(record['test_acc'] for record in run.history),
               'selection': CLASSIFICATION_SELECTION.selected_record(run.best_epoch, run.best_metrics),
               'history': run.history, **m}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents partial output and replaces symlinks without overwriting their targets.
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    os.replace(tmp, out)
    print(f'finetune {spec.flag}: AUC={m["auc"]:.4f} ACC={m["acc"]:.4f} '
          f'(best val_auc={run.best_metrics["auc"]:.4f}, epoch={run.best_epoch}) -> {out}')


if __name__ == '__main__':
    main()
