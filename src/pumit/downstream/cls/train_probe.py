"""Train linear/MLP probe heads on cached native MedMNIST features.

Usage:
    pixi run -e cls python -m pumit.downstream.cls.train_probe \
        --features-dir precompute/downstream/cls/mock/pneumoniamnist \
        --flag pneumoniamnist --size 224 --n-classes 2 \
        --head linear --epochs 50 --seeds 3 \
        --out outputs/downstream/cls/mock/pneumoniamnist_probe.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import torch
import torch.nn as nn

from pumit.downstream.cached_probe import (
    CachedProbeRun,
    aggregate_metric_records,
    build_probe_head,
    train_cached_probe,
    train_cached_probe_seeds,
)

from .task import CLASSIFICATION_SELECTION, classification_objective
from .data import DEFAULT_DATA_ROOT
from .metrics import evaluate, fallback_metrics


def build_cls_head(embed_dim: int, n_classes: int, head_type: str) -> nn.Module:
    return build_probe_head(embed_dim, n_classes, head_type)


def train_one_seed(
    train_x, train_y, val_x, val_y, *, n_classes, head_type,
    epochs, lr, weight_decay, batch_size, device, seed,
    progress=True, updates=None, eval_every=None,
) -> CachedProbeRun:
    """Train one classification probe seed and restore the best-val-AUC head."""

    def evaluator(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return fallback_metrics(logits, target.numpy(), n_classes)

    return train_cached_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        head_factory=lambda input_dim: build_cls_head(input_dim, n_classes, head_type),
        objective=classification_objective,
        evaluator=evaluator,
        selection=CLASSIFICATION_SELECTION,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        seed=seed,
        progress=progress,
        updates=updates,
        eval_every=eval_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--flag', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--n-classes', type=int, required=True)
    parser.add_argument('--head', default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--standardize', action='store_true',
                        help='z-score features per dimension using train-split statistics before '
                             'fitting the head, making the probe invariant to each backbone\'s '
                             'feature scale')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--epochs', type=int, default=50)
    budget.add_argument('--updates', type=int, help='train for this many full-batch optimizer updates')
    parser.add_argument('--eval-every', type=int, help='validation interval in updates; required with --updates')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    parser.add_argument('--save-heads', action='store_true',
                        help='also write the val-selected head weights per seed to '
                             '<out>.heads.pt, for LP-FT (finetune --head-init). Off by default: '
                             'an mlp head on a 1024-d readout is ~4 MB per seed')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help='persistent MedMNIST npz dir used by the official Evaluator')
    args = parser.parse_args()
    if args.updates is not None:
        if args.updates <= 0 or args.eval_every is None or args.eval_every <= 0:
            parser.error('--updates and --eval-every must both be positive in fixed-update mode')
    elif args.eval_every is not None:
        parser.error('--eval-every requires --updates')

    features_dir = Path(args.features_dir)
    train = torch.load(features_dir / 'train.pt', weights_only=True)
    val = torch.load(features_dir / 'val.pt', weights_only=True)
    test = torch.load(features_dir / 'test.pt', weights_only=True)
    train_x, val_x, test_x = train['native'], val['native'], test['native']
    train_y, val_y, test_y = train['label'], val['label'], test['label']
    if args.standardize:
        # Per-dimension z-score from the TRAIN split only, applied to every split. Without it the
        # probe reads raw features whose scale varies ~24x across backbones, and since Adam's step
        # size is invariant to gradient scale, a head on small features needs proportionally more
        # steps to reach the same logit magnitude -- a fixed epoch budget then handicaps those arms.
        mu = train_x.mean(0, keepdim=True)
        sigma = train_x.std(0, keepdim=True).clamp_min(1e-6)   # dead dimensions stay at zero
        train_x, val_x, test_x = ((v - mu) / sigma for v in (train_x, val_x, test_x))

    def val_evaluator(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return fallback_metrics(logits, target.numpy(), args.n_classes)

    runs = train_cached_probe_seeds(
        seeds=args.seeds,
        train_features=train_x,
        train_targets=train_y,
        val_features=val_x,
        val_targets=val_y,
        head_factory=lambda input_dim: build_cls_head(input_dim, args.n_classes, args.head),
        objective=classification_objective,
        evaluator=val_evaluator,
        selection=CLASSIFICATION_SELECTION,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        updates=args.updates,
        eval_every=args.eval_every,
    )

    test_results, val_results = [], []
    for seed, run in enumerate(runs):
        with torch.no_grad():
            test_logits = run.model(test_x.to(args.device)).cpu()
        metrics = evaluate(
            test_logits,
            test_y.numpy(),
            args.flag,
            args.size,
            'test',
            root=args.data_root,
        )
        print(
            f'seed {seed}: val_AUC={run.best_metrics["auc"]:.4f} '
            f'AUC={metrics["auc"]:.4f} ACC={metrics["acc"]:.4f}'
        )
        val_results.append(run.best_metrics)
        test_results.append(metrics)

    val_summary = aggregate_metric_records(val_results)
    test_summary = aggregate_metric_records(test_results)
    summary = {
        'flag': args.flag,
        'head': args.head,
        'readout': 'native',
        'standardize': args.standardize,
        'seeds': args.seeds,
        'data_root': args.data_root,
        # Which encoder and geometry produced the cached features. The probe never sees the
        # encoder, so without carrying this through, a result cannot state what it measured.
        'features': train.get('meta', {}),
        'per_seed': test_results,
        'per_seed_val': val_results,
        'protocol': 'frozen_probe',
        'selection_policy': CLASSIFICATION_SELECTION.policy_record(),
        'per_seed_selection': [],
        'metrics': {'val': val_summary, 'test': test_summary},
        'val_auc_mean': val_summary['auc']['mean'],
        'val_auc_std': val_summary['auc']['std'],
        'val_acc_mean': val_summary['acc']['mean'],
        'val_acc_std': val_summary['acc']['std'],
        'auc_mean': test_summary['auc']['mean'],
        'auc_std': test_summary['auc']['std'],
        'acc_mean': test_summary['acc']['mean'],
        'acc_std': test_summary['acc']['std'],
    }
    for seed, run in enumerate(runs):
        if args.updates is None:
            selected = CLASSIFICATION_SELECTION.selected_record(run.best_epoch, run.best_metrics)
        else:
            selected = {
                **CLASSIFICATION_SELECTION.policy_record(),
                'selected_update': run.best_update,
                'value': CLASSIFICATION_SELECTION.value(run.best_metrics),
                'metrics': run.best_metrics,
            }
        summary['per_seed_selection'].append({'seed': seed, **selected})
    if args.updates is not None:
        summary.update({
            'updates': args.updates,
            'eval_every': args.eval_every,
            'batch_size': args.batch_size,
            'per_seed_history': [run.history for run in runs],
        })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    if args.save_heads:
        heads = output.with_suffix('.heads.pt')
        torch.save({'head': args.head, 'flag': args.flag, 'n_classes': args.n_classes,
                    'features': train.get('meta', {}),
                    # index = seed, matching per_seed/per_seed_val above
                    'state_dicts': [{k: v.cpu() for k, v in run.model.state_dict().items()}
                                    for run in runs]}, heads)
        print(f'heads -> {heads}')
    print(
        f'val_AUC {summary["val_auc_mean"]:.4f} '
        f'AUC {summary["auc_mean"]:.4f}+/-{summary["auc_std"]:.4f} '
        f'ACC {summary["acc_mean"]:.4f}+/-{summary["acc_std"]:.4f} -> {output}'
    )


if __name__ == '__main__':
    main()
