"""Train a lightweight probe head on pre-extracted CLS features to predict voxel spacing.

Usage:
    pixi run -e default python -m pumit.downstream.spatial.train_spacing \
        --features-dir /path/to/features \
        --head mlp --epochs 50 --seeds 5

Expected data format:
    features_dir/train.pt: {'cls': Tensor[N, D], 'spacing': Tensor[N, 3]}
    features_dir/val.pt:   {'cls': Tensor[N, D], 'spacing': Tensor[N, 3]}

Targets are log(spacing). NaN values are masked per-element during loss and evaluation.
"""

import argparse
from pathlib import Path

import orjson
import torch

from ..cached_probe import (
    aggregate_metric_records,
    train_cached_probe_seeds,
)
from .spacing import (
    SPACING_SELECTION,
    build_spacing_head,
    compute_spacing_metrics,
    masked_l1_loss,
    train_spacing_probe,
)

# Preserve the script-level names used by earlier ad-hoc imports.
build_probe_head = build_spacing_head
compute_metrics = compute_spacing_metrics
train_one_seed = train_spacing_probe


def main():
    parser = argparse.ArgumentParser(description='Train spacing prediction probe on CLS features')
    parser.add_argument('--features-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True, help='e.g., outputs/probe/dinov3-vitb16-uniform/')
    parser.add_argument('--head', type=str, default='mlp', choices=['linear', 'mlp', 'mlp2', 'mlp3'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    train_data = torch.load(features_dir / 'train.pt', map_location='cpu', weights_only=True)
    val_data = torch.load(features_dir / 'val.pt', map_location='cpu', weights_only=True)

    train_cls = train_data['cls'].float()
    train_spacing = torch.log(train_data['spacing'].float())
    val_cls = val_data['cls'].float()
    val_spacing = torch.log(val_data['spacing'].float())

    print(f"Train: {train_cls.shape[0]} samples, embed_dim={train_cls.shape[1]}")
    print(f"Val:   {val_cls.shape[0]} samples")
    print(f"Head: {args.head}, epochs: {args.epochs}, lr: {args.lr}, batch_size: {args.batch_size}, seeds: {args.seeds}")
    print(f"Device: {args.device}")

    runs = train_cached_probe_seeds(
        seeds=args.seeds,
        train_features=train_cls,
        train_targets=train_spacing,
        val_features=val_cls,
        val_targets=val_spacing,
        head_factory=lambda input_dim: build_spacing_head(input_dim, args.head),
        objective=masked_l1_loss,
        evaluator=compute_spacing_metrics,
        selection=SPACING_SELECTION,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
    )
    all_metrics = [run.best_metrics for run in runs]
    all_weights = [
        {key: value.detach().cpu().clone() for key, value in run.model.state_dict().items()}
        for run in runs
    ]

    # Aggregate
    print("\n" + "=" * 60)
    print("RESULTS (across seeds)")
    print("=" * 60)
    metric_summary = aggregate_metric_records(all_metrics)
    summary = {
        'config': vars(args),
        'protocol': 'frozen_probe',
        'per_seed': all_metrics,
        'selection_policy': SPACING_SELECTION.policy_record(),
        'per_seed_selection': [
            {'seed': seed, **SPACING_SELECTION.selected_record(run.best_epoch, run.best_metrics)}
            for seed, run in enumerate(runs)
        ],
        'metrics': {'val': metric_summary},
        'summary': metric_summary,
    }
    for key in ('mae_mean', 'mae_depth', 'mae_height', 'mae_width'):
        metric = metric_summary[key]
        print(f'  {key}: {metric["mean"]:.4f} +/- {metric["std"]:.4f}')

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"spacing_{args.head}.json"
    results_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    print(f"\nResults saved to {results_path}")

    # Save model weights for all seeds
    weights_path = output_dir / f"spacing_{args.head}_weights.pt"
    torch.save(all_weights, weights_path)
    print(f"Weights saved to {weights_path}")


if __name__ == '__main__':
    main()
