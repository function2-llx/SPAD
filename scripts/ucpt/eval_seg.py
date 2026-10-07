"""Evaluate UCPT segmentation checkpoints with full-volume sliding-window inference."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import orjson
import torch
import torch.multiprocessing as mp
import yaml

from pumit.ucpt.input import InputNormalizer
from pumit.ucpt.seg.evaluation import (
    INTERPOLATION_ORDERS,
    SegEvalDataset,
    full_volume_metric_rows,
    generate_panel,
    load_checkpoint_metadata,
    load_segmentation_artifact,
    save_panel,
    sliding_window_logits,
    summarize_metric_rows,
)


def _assign_case_indices(samples: list[dict], num_workers: int) -> list[list[int]]:
    """Greedily balance workers by estimated inference cost."""
    assignments = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    costs = [
        sample['sliding_window']['num_windows'] * (1 + len(sample['classes']))
        for sample in samples
    ]
    ordered = sorted(
        range(len(samples)),
        key=lambda index: (-costs[index], index),
    )
    for index in ordered:
        rank = min(range(num_workers), key=lambda worker: (loads[worker], worker))
        assignments[rank].append(index)
        loads[rank] += costs[index]
    return assignments


def _visible_devices() -> list[int]:
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError('segmentation evaluation requires at least one visible CUDA device')
    return list(range(device_count))


def _evaluate_worker(
    rank: int,
    devices: list[int],
    checkpoint_path: str,
    panel_path: str,
    stacks: tuple[str, ...],
    assignments: list[list[int]],
    sw_batch_size: int,
    interpolation_order: str,
    shard_dir: str,
) -> None:
    device = torch.device('cuda', devices[rank])
    torch.cuda.set_device(device)
    metadata = load_checkpoint_metadata(Path(checkpoint_path))
    config = metadata['config']
    dataset = SegEvalDataset(
        Path(panel_path),
        data_root=Path(config['data']['data_root']),
        text_cache_path=Path(config['data']['text_cache_path']),
        class_captions_dir=Path(config['data']['class_captions_dir']),
        input_normalizer=InputNormalizer(config['data']['data_root']),
    )
    indices = assignments[rank]

    for stack in stacks:
        artifact, _ = load_segmentation_artifact(Path(checkpoint_path), stack=stack, device=device)
        shard_path = Path(shard_dir) / f'{stack}-rank-{rank}.jsonl'
        with open(shard_path, 'xb') as file, torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for case_index in indices:
                item = dataset[case_index]
                logits = sliding_window_logits(
                    artifact,
                    item,
                    device=device,
                    sw_batch_size=sw_batch_size,
                    interpolation_order=interpolation_order,
                )
                result = {
                    'case_index': case_index,
                    'rows': full_volume_metric_rows(
                        logits,
                        item,
                        data_root=dataset.data_root,
                        device=device,
                        stack=stack,
                        checkpoint_step=metadata['step'],
                        interpolation_order=interpolation_order,
                    ),
                }
                file.write(orjson.dumps(result))
                file.write(b'\n')
        del artifact
        torch.cuda.empty_cache()


def _merge_shards(shard_dir: Path, stack: str, num_workers: int, num_cases: int) -> list[dict]:
    case_rows: list[tuple[int, list[dict]]] = []
    for rank in range(num_workers):
        path = shard_dir / f'{stack}-rank-{rank}.jsonl'
        with open(path, 'rb') as file:
            for line in file:
                result = orjson.loads(line)
                case_rows.append((result['case_index'], result['rows']))
    case_rows.sort(key=lambda item: item[0])
    indices = [case_index for case_index, _ in case_rows]
    if indices != list(range(num_cases)):
        raise RuntimeError(f'incomplete or duplicate evaluation shards: got case indices {indices}')
    return [row for _, rows in case_rows for row in rows]


def _output_paths(output_dir: Path, run_id: str, step: int, stacks: tuple[str, ...]) -> dict[str, tuple[Path, Path]]:
    paths = {}
    for stack in stacks:
        stem = f'{run_id}-step-{step}-{stack}'
        paths[stack] = (
            output_dir / f'{stem}.jsonl',
            output_dir / f'{stem}-summary.json',
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/ucpt/seg-eval.yaml'))
    parser.add_argument('--stack', choices=('ema', 'online', 'both'), default='ema')
    parser.add_argument('--interpolation-order', choices=INTERPOLATION_ORDERS)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    metadata = load_checkpoint_metadata(args.checkpoint)
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f'--limit must be positive, got {args.limit}')
    with open(args.config) as file:
        panel_config = yaml.safe_load(file)
    checkpoint_config = metadata['config']
    max_cases = args.limit if args.limit is not None else panel_config.get('max_cases')
    overlap = float(panel_config['overlap'])
    sw_batch_size = int(panel_config['sw_batch_size'])
    interpolation_order = args.interpolation_order or panel_config['interpolation_order']
    panel = generate_panel(
        data_root=Path(checkpoint_config['data']['data_root']),
        generation_config_path=Path(panel_config['generation_config']),
        class_captions_dir=Path(checkpoint_config['data']['class_captions_dir']),
        panel=panel_config['panel'],
        seed=int(panel_config['seed']),
        max_cases=max_cases,
        k_neg=int(panel_config['k_neg']),
        overlap=overlap,
    )
    num_cases = len(panel['samples'])
    if num_cases == 0:
        raise ValueError('segmentation panel contains no cases')
    devices = _visible_devices()
    assignments = _assign_case_indices(panel['samples'], len(devices))
    stacks = ('ema', 'online') if args.stack == 'both' else (args.stack,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_bytes = orjson.dumps(panel, option=orjson.OPT_INDENT_2)
    panel_sha256 = hashlib.sha256(panel_bytes).hexdigest()
    panel_path = args.output_dir / f'panel-{panel_sha256[:16]}.json'
    output_paths = _output_paths(
        args.output_dir,
        metadata['run_id'],
        metadata['step'],
        stacks,
    )
    combined_path = (
        args.output_dir / f'{metadata["run_id"]}-step-{metadata["step"]}-summaries.json'
        if len(stacks) > 1 else None
    )
    planned_paths = [
        path
        for row_path, summary_path in output_paths.values()
        for path in (row_path, summary_path)
    ]
    if combined_path is not None:
        planned_paths.append(combined_path)
    existing = [path for path in planned_paths if path.exists()]
    if existing:
        raise FileExistsError(f'evaluation outputs already exist: {existing}')
    save_panel(panel, panel_path)

    with tempfile.TemporaryDirectory(prefix='.seg-eval-', dir=args.output_dir) as shard_dir:
        worker_args = (
            devices,
            str(args.checkpoint),
            str(panel_path),
            stacks,
            assignments,
            sw_batch_size,
            interpolation_order,
            shard_dir,
        )
        if len(devices) == 1:
            _evaluate_worker(0, *worker_args)
        else:
            mp.spawn(_evaluate_worker, args=worker_args, nprocs=len(devices), join=True)

        all_summaries = []
        for stack in stacks:
            rows = _merge_shards(Path(shard_dir), stack, len(devices), num_cases)
            row_path, summary_path = output_paths[stack]
            with open(row_path, 'xb') as file:
                for row in rows:
                    file.write(orjson.dumps(row))
                    file.write(b'\n')
            summary = {
                'checkpoint': str(args.checkpoint),
                'run_id': metadata['run_id'],
                'panel_path': str(panel_path),
                'panel_sha256': panel_sha256,
                'panel': panel['panel'],
                'step': metadata['step'],
                'stack': stack,
                'probability_threshold': 0.5,
                'overlap': overlap,
                'sw_batch_size': sw_batch_size,
                'interpolation_order': interpolation_order,
                'n_cases': num_cases,
                'n_windows': sum(sample['sliding_window']['num_windows'] for sample in panel['samples']),
                'devices': devices,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                **summarize_metric_rows(rows),
            }
            with open(summary_path, 'xb') as file:
                file.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
            all_summaries.append(summary)

        if combined_path is not None:
            with open(combined_path, 'xb') as file:
                file.write(orjson.dumps(all_summaries, option=orjson.OPT_INDENT_2))


if __name__ == '__main__':
    main()
