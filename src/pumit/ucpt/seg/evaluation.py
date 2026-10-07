"""Full-volume held-out evaluation for UCPT segmentation artifacts."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
import torch
import yaml
from monai.inferers import sliding_window_inference
from torch import Tensor, nn
import torch.nn.functional as F

from pumit.codec.config import MAX_DA
from pumit.data import build_training_data
from pumit.model.vit import ViT, ViTConfig
from pumit.text_prompt import SegmentationPromptResolver, class_captions_sha256
from pumit.ucpt.batch import seg_output_grid
from pumit.ucpt.input import InputNormalizer, normalize_input
from pumit.ucpt.mask import _load_mask
from pumit.ucpt.model import SegDecoderStack
from pumit.ucpt.sample_stream import _labeled_eligible

from .class_sampling import flatten_label_contract
from .label_contract import normalize_label_contract
from .text_encoding import TextEmbeddingCache


PANEL_FORMAT_VERSION = 2
INTERPOLATION_ORDERS = ('stitch-then-interpolate', 'interpolate-then-stitch')
_PATCH_SIZE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_rng(seed: int, dataset: str, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f'{seed}|{dataset}|{key}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def _balanced_subset(records: list[dict], max_cases: int | None, seed: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record['dataset'], record['modality'])].append(record)

    rng = np.random.default_rng(seed)
    for group in groups.values():
        group.sort(key=lambda record: record['key'])
        rng.shuffle(group)

    ordered_groups = sorted(groups)
    selected: list[dict] = []
    depth = 0
    while max_cases is None or len(selected) < max_cases:
        added = False
        for group_key in ordered_groups:
            group = groups[group_key]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if max_cases is not None and len(selected) == max_cases:
                    break
        if not added:
            break
        depth += 1
    return selected


def _sample_classes(record: dict, k_neg: int, rng: np.random.Generator) -> list[dict]:
    contract, _ = normalize_label_contract(record)
    positives, negatives = flatten_label_contract(contract)
    n_neg = min(k_neg, len(negatives))
    if n_neg:
        indices = rng.choice(len(negatives), n_neg, replace=False)
        sampled_negatives = [negatives[int(index)] for index in indices]
    else:
        sampled_negatives = []

    return [
        *({'source': source, 'name': name, 'is_positive': True} for source, name in positives),
        *({'source': source, 'name': name, 'is_positive': False} for source, name in sampled_negatives),
    ]


def _resolve_da(record: dict) -> int | None:
    shape = np.asarray(record['shape'], dtype=np.int64)
    spacing = np.asarray(record['spacing'], dtype=np.float64)
    if shape[0] == 1:
        return None
    if not np.isfinite(spacing).all():
        raise ValueError(f'3D record has non-finite spacing: {record["dataset"]}/{record["key"]}')
    ratio = float(np.clip(spacing[0] / spacing[1:].min(), 1, 1 << MAX_DA))
    return min(int(math.log2(ratio)), MAX_DA)


def _resolve_roi_size(record: dict, da: int | None, generation_config: dict) -> tuple[int, int, int]:
    shape = np.asarray(record['shape'], dtype=np.int64)
    is_2d = da is None
    choices = generation_config['size_xy_choices_2d'] if is_2d else generation_config['size_xy_choices']
    cap_key = 'labeled_size_xy_max_2d' if is_2d else 'labeled_size_xy_max'
    if (cap := generation_config.get(cap_key)) is not None:
        choices = [size for size in choices if size <= cap]
    if not choices:
        raise ValueError(f'{cap_key} excludes all crop sizes')
    eligible = [size for size in choices if size <= int(shape[1:].min())]
    size_xy = max(eligible) if eligible else min(choices)

    if is_2d:
        return (1, size_xy, size_xy)
    patch_d = _PATCH_SIZE >> da
    max_depth = int(generation_config['max_depth_per_da'][da])
    size_z = math.ceil(min(int(shape[0]), max_depth) / patch_d) * patch_d
    return (size_z, size_xy, size_xy)


def _sliding_geometry(
    shape: tuple[int, int, int],
    roi_size: tuple[int, int, int],
    da: int | None,
    overlap: float,
) -> dict:
    if not 0 <= overlap < 1:
        raise ValueError(f'overlap must be in [0, 1), got {overlap}')
    patch_d = _PATCH_SIZE >> (MAX_DA if da is None else da)
    alignment = (patch_d, _PATCH_SIZE, _PATCH_SIZE)
    scan_interval = tuple(
        min(roi, max(align, round(roi * (1 - overlap) / align) * align))
        for roi, align in zip(roi_size, alignment)
    )
    effective_overlap = tuple(1 - interval / roi for roi, interval in zip(roi_size, scan_interval))
    for axis, (roi, interval, align, axis_overlap) in enumerate(
        zip(roi_size, scan_interval, alignment, effective_overlap)
    ):
        if roi % align:
            raise ValueError(
                f'ROI must align with ViT patches on axis {axis}: {roi_size=} {alignment=}'
            )
        realized_interval = max(1, int(roi * (1 - axis_overlap)))
        if realized_interval != interval:
            raise RuntimeError(
                f'overlap does not reproduce the aligned scan interval on axis {axis}: '
                f'{roi=} {axis_overlap=} {interval=} {realized_interval=}'
            )
    padded_shape = tuple(
        roi if size <= roi else roi + math.ceil((size - roi) / interval) * interval
        for size, roi, interval in zip(shape, roi_size, scan_interval)
    )
    num_windows = math.prod(
        1 + (padded - roi) // interval
        for padded, roi, interval in zip(padded_shape, roi_size, scan_interval)
    )
    patch_grid = (
        padded_shape[0] // patch_d,
        padded_shape[1] // _PATCH_SIZE,
        padded_shape[2] // _PATCH_SIZE,
    )
    return {
        'roi_size': list(roi_size),
        'scan_interval': list(scan_interval),
        'overlap': list(effective_overlap),
        'padded_shape': list(padded_shape),
        'output_shape': list(seg_output_grid(da, patch_grid)),
        'num_windows': num_windows,
    }


def load_generation_config(path: Path) -> dict:
    """Read the stream generation config with its ``max_depth_per_da`` keys restored to ints."""
    with open(path) as file:
        generation_config = yaml.safe_load(file)
    generation_config['max_depth_per_da'] = {
        int(key): value for key, value in generation_config['max_depth_per_da'].items()
    }
    return generation_config


def panel_sample(
    record: dict,
    *,
    k_neg: int,
    rng: np.random.Generator,
    generation_config: dict,
    overlap: float,
) -> dict:
    """Build one panel sample from a data record: its prompted classes and full-volume sliding-window geometry."""
    da = _resolve_da(record)
    shape = tuple(int(value) for value in record['shape'])
    roi_size = _resolve_roi_size(record, da, generation_config)
    return {
        'dataset': record['dataset'],
        'key': record['key'],
        'img': record['img'],
        'shape': list(shape),
        'spacing': [
            None if not math.isfinite(float(value)) else float(value)
            for value in record['spacing']
        ],
        'modality': record['modality'],
        'original_split': record.get('split'),
        'classes': _sample_classes(record, k_neg, rng),
        'da': da,
        'sliding_window': _sliding_geometry(shape, roi_size, da, overlap),
    }


def generate_panel(
    *,
    data_root: Path,
    generation_config_path: Path,
    class_captions_dir: Path,
    panel: str,
    seed: int,
    max_cases: int | None,
    k_neg: int,
    overlap: float,
) -> dict:
    """Generate a deterministic full-volume segmentation panel.

    ``record-heldout`` uses segmentation-eligible records listed in the current per-dataset ``val.json`` files.
    ``official-test`` further restricts the panel to records whose original split is ``test``.
    """
    if panel not in {'record-heldout', 'official-test'}:
        raise ValueError(f'unknown panel {panel!r}')
    generation_config = load_generation_config(generation_config_path)

    _, val_data, _ = build_training_data(
        data_root=data_root,
        depth_tiers=None,
        verbose=False,
        max_da=MAX_DA,
    )
    records = [
        record
        for record in val_data.reset_index().to_dict('records')
        if _labeled_eligible(record)
        and flatten_label_contract(normalize_label_contract(record)[0])[0]
    ]
    if panel == 'official-test':
        records = [record for record in records if record.get('split') == 'test']
    records = _balanced_subset(records, max_cases, seed)

    datasets = sorted({record['dataset'] for record in records})
    resolver = SegmentationPromptResolver(class_captions_dir)
    samples = []
    for record in records:
        sample = panel_sample(
            record,
            k_neg=k_neg,
            rng=_record_rng(seed, record['dataset'], record['key']),
            generation_config=generation_config,
            overlap=overlap,
        )
        for cls in sample['classes']:
            resolver.get(cls['source'], cls['name'], record['modality'])
        samples.append(sample)

    source_files = []
    for dataset in datasets:
        for name in ('meta.json', 'val.json'):
            path = data_root / dataset / name
            source_files.append({'path': str(path), 'sha256': _sha256(path)})
    source_files.append({'path': str(generation_config_path), 'sha256': _sha256(generation_config_path)})
    source_files.append({
        'path': str(class_captions_dir),
        'sha256': class_captions_sha256(resolver.captions),
    })

    return {
        'format_version': PANEL_FORMAT_VERSION,
        'panel': panel,
        'heldout_unit': 'record',
        'seed': seed,
        'max_cases': max_cases,
        'k_neg': k_neg,
        'overlap': overlap,
        'inference_policy': 'full-volume sliding window with Gaussian logit blending',
        'known_limitations': [
            'Per-dataset val.json excludes exact records but does not guarantee patient-level separation.',
        ],
        'source_files': source_files,
        'samples': samples,
    }


def save_panel(panel: dict, path: Path) -> bool:
    """Write a new panel, or verify that the existing panel is byte-identical."""
    payload = orjson.dumps(panel, option=orjson.OPT_INDENT_2)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f'refusing to replace a different segmentation panel: {path}')
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging name: ranks (possibly on several nodes) may save one panel concurrently, and a shared name
    # lets one rank's truncating rewrite land in the inode another rank has already published.
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_bytes(payload)
    tmp.replace(path)
    return True


class SegEvalDataset:
    """Load full validation volumes and their frozen text queries."""

    def __init__(
        self,
        panel_path: Path,
        *,
        data_root: Path,
        text_cache_path: Path,
        class_captions_dir: Path,
        input_normalizer: InputNormalizer | None = None,
    ):
        panel = orjson.loads(panel_path.read_bytes())
        if panel.get('format_version') != PANEL_FORMAT_VERSION:
            raise ValueError(f'unsupported panel format: {panel.get("format_version")!r}')
        self.panel = panel
        self.samples = panel['samples']
        self.data_root = Path(data_root)
        self.text_cache = TextEmbeddingCache(text_cache_path)
        self.prompt_resolver = SegmentationPromptResolver(class_captions_dir)
        self.input_normalizer = input_normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image_np = np.load(sample['img'], mmap_mode='r')
        if tuple(image_np.shape[1:]) != tuple(sample['shape']):
            raise ValueError(
                f'image shape changed for {sample["dataset"]}/{sample["key"]}: '
                f'{tuple(image_np.shape[1:])} != {tuple(sample["shape"])}'
            )
        image = torch.from_numpy(np.array(image_np, dtype=np.float32, copy=True))
        padded_shape = sample['sliding_window']['padded_shape']
        pad = [
            0, padded_shape[2] - image.shape[3],
            0, padded_shape[1] - image.shape[2],
            0, padded_shape[0] - image.shape[1],
        ]
        if any(pad):
            image = F.pad(image.unsqueeze(0), pad, mode='replicate').squeeze(0)

        label_keys = [
            (cls['source'], cls['name'], sample['modality'])
            for cls in sample['classes']
        ]
        prompts = self.prompt_resolver.get_batch(label_keys)
        text_embeddings = self.text_cache.get_batch(prompts)
        text_valid_mask = self.text_cache.get_mask_batch(prompts)
        if not text_valid_mask.any(-1).all():
            raise ValueError(
                f'all-pad text query in {sample["dataset"]}/{sample["key"]}'
            )
        item = {
            'image': image,
            'da': sample['da'],
            'prompts': prompts,
            'text_embeddings': text_embeddings,
            'text_valid_mask': text_valid_mask,
            'sample': sample,
        }
        if self.input_normalizer is not None:
            item['input_scheme'] = self.input_normalizer.scheme_for(sample['dataset'], sample['key'])
        return item


class SegmentationArtifact(nn.Module):
    """EMA or online encoder-decoder pair used for checkpoint evaluation."""

    def __init__(self, vit: ViT, seg: SegDecoderStack):
        super().__init__()
        self.vit = vit
        self.seg = seg

    def forward(
        self,
        image: Tensor,
        da: int | None,
        text_embeddings: Tensor,
        text_valid_mask: Tensor,
    ) -> Tensor:
        """Predict concept masks for one crop or a batch of equal-sized crops."""
        single = image.ndim == 4
        if single:
            image = image.unsqueeze(0)
        if image.ndim != 5:
            raise ValueError(f'expected image shape (C,D,H,W) or (B,C,D,H,W), got {tuple(image.shape)}')

        patch_da = MAX_DA if da is None else da
        patch_d = _PATCH_SIZE >> patch_da
        grid = (
            image.shape[2] // patch_d,
            image.shape[3] // _PATCH_SIZE,
            image.shape[4] // _PATCH_SIZE,
        )
        _, patch_features = self.vit.encode_image(image, da=patch_da)
        feat_3d = patch_features.reshape(image.shape[0], *grid, self.vit.embed_dim).permute(0, 4, 1, 2, 3)
        logits = torch.stack([
            self.seg(features.unsqueeze(0), text_embeddings, text_valid_mask, da)
            for features in feat_3d
        ])
        return logits[0] if single else logits


def sliding_window_logits(
    artifact: SegmentationArtifact,
    item: dict,
    *,
    device: torch.device,
    sw_batch_size: int,
    interpolation_order: str,
) -> Tensor:
    """Run Gaussian-blended sliding-window inference and return fp32 logits on ``device``."""
    if sw_batch_size <= 0:
        raise ValueError(f'sw_batch_size must be positive, got {sw_batch_size}')
    if interpolation_order not in INTERPOLATION_ORDERS:
        raise ValueError(f'unknown interpolation order {interpolation_order!r}')
    text_embeddings = item['text_embeddings'].to(device)
    text_valid_mask = item['text_valid_mask'].to(device)

    def predictor(windows: Tensor) -> Tensor:
        if 'input_scheme' in item:
            windows = normalize_input(windows, item['input_scheme'], batched=True)
        else:
            windows = windows * 2 - 1
            if windows.shape[1] == 1:
                windows = windows.repeat(1, 3, 1, 1, 1)
            elif windows.shape[1] != 3:
                raise ValueError(f'expected one or three image channels, got {windows.shape[1]}')
        logits = artifact(windows, item['da'], text_embeddings, text_valid_mask)
        logits = logits.squeeze(2).float()
        if interpolation_order == 'interpolate-then-stitch':
            logits = F.interpolate(logits, size=windows.shape[-3:], mode='trilinear', align_corners=False)
        return logits.contiguous()

    logits = sliding_window_inference(
        inputs=item['image'].unsqueeze(0),
        roi_size=item['sample']['sliding_window']['roi_size'],
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=item['sample']['sliding_window']['overlap'],
        mode='gaussian',
        padding_mode='replicate',
        sw_device=device,
        device=device,
        progress=False,
    )
    spatial_shape = (
        item['sample']['sliding_window']['output_shape']
        if interpolation_order == 'stitch-then-interpolate'
        else item['sample']['sliding_window']['padded_shape']
    )
    expected = (1, len(item['sample']['classes']), *spatial_shape)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f'unexpected stitched-logit shape: {tuple(logits.shape)} != {expected}')
    return logits.squeeze(0)


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """Read the checkpoint identity and resolved config without constructing modules."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    return {
        'checkpoint': str(checkpoint_path),
        'run_id': checkpoint['run_id'],
        'step': int(checkpoint['step']),
        'config': checkpoint['config'],
    }


def load_segmentation_artifact(
    checkpoint_path: Path,
    *,
    stack: str,
    device: torch.device,
) -> tuple[SegmentationArtifact, dict]:
    """Load one strict encoder-decoder pair from a UCPT checkpoint."""
    if stack not in {'ema', 'online'}:
        raise ValueError(f'unknown stack {stack!r}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    config = checkpoint['config']
    model_config = config['model']
    vit = ViT(ViTConfig(
        hidden_size=model_config['embed_dim'],
        num_hidden_layers=model_config['depth'],
        num_attention_heads=model_config['num_heads'],
        intermediate_size=int(model_config['embed_dim'] * model_config['mlp_ratio']),
        num_register_tokens=model_config['n_register_tokens'],
        grad_ckpt=False,
    ))
    seg = SegDecoderStack(
        embed_dim=model_config['embed_dim'],
        text_embed_dim=model_config['text_embed_dim'],
        hidden_size=model_config['seg_hidden_size'],
    )
    state = {
        key.replace('_orig_mod.', ''): value
        for key, value in checkpoint['model'].items()
    }
    vit_prefix = 'teacher_vit.' if stack == 'ema' else 'vit.'
    seg_prefix = 'ema_seg.' if stack == 'ema' else 'seg.'
    vit.load_state_dict({
        key[len(vit_prefix):]: value for key, value in state.items() if key.startswith(vit_prefix)
    }, strict=True)
    seg.load_state_dict({
        key[len(seg_prefix):]: value for key, value in state.items() if key.startswith(seg_prefix)
    }, strict=True)
    artifact = SegmentationArtifact(vit, seg).to(device).eval()
    metadata = {
        'checkpoint': str(checkpoint_path),
        'run_id': checkpoint['run_id'],
        'step': int(checkpoint['step']),
        'stack': stack,
        'config': config,
    }
    return artifact, metadata


def _size_bin(fraction: float) -> str:
    if fraction < 0.001:
        return '<0.1%'
    if fraction < 0.01:
        return '0.1-1%'
    if fraction < 0.05:
        return '1-5%'
    return '>5%'


def _positive_overlap_metrics(tp: int, predicted: int, target: int) -> dict[str, float]:
    return {
        'dice': 2 * tp / (predicted + target),
        'iou': tp / (predicted + target - tp),
        'precision': tp / predicted if predicted else 0.0,
        'recall': tp / target,
    }


def full_volume_metric_rows(
    logits: Tensor,
    item: dict,
    *,
    data_root: Path,
    device: torch.device,
    stack: str,
    checkpoint_step: int,
    interpolation_order: str,
) -> list[dict]:
    """Compute native-grid metrics per concept."""
    if interpolation_order not in INTERPOLATION_ORDERS:
        raise ValueError(f'unknown interpolation order {interpolation_order!r}')
    sample = item['sample']
    shape = tuple(sample['shape'])
    padded_shape = tuple(sample['sliding_window']['padded_shape'])
    n_voxels = math.prod(shape)
    rows = []
    for logit, cls, prompt in zip(logits, sample['classes'], item['prompts']):
        if interpolation_order == 'stitch-then-interpolate':
            native_logit = F.interpolate(
                logit[None, None].to(device),
                size=padded_shape,
                mode='trilinear',
                align_corners=False,
            )[0, 0]
        else:
            if tuple(logit.shape) != padded_shape:
                raise RuntimeError(f'unexpected native-logit shape: {tuple(logit.shape)} != {padded_shape}')
            native_logit = logit.to(device)
        native_logit = native_logit[:shape[0], :shape[1], :shape[2]]
        prediction = native_logit >= 0
        predicted_sum = int(prediction.sum().item())

        target = None
        target_sum = 0
        if cls['is_positive']:
            mask = _load_mask(
                data_root,
                sample['dataset'],
                sample['key'],
                cls['source'],
                cls['name'],
                shape,
            )
            if mask is None:
                raise FileNotFoundError(
                    f'missing positive mask: dataset={sample["dataset"]!r} key={sample["key"]!r} '
                    f'source={cls["source"]!r} class={cls["name"]!r}'
                )
            target = torch.from_numpy(mask).to(device)
            target_sum = int(target.sum().item())
            if target_sum == 0:
                raise ValueError(
                    f'empty volume-positive mask: dataset={sample["dataset"]!r} key={sample["key"]!r} '
                    f'source={cls["source"]!r} class={cls["name"]!r}'
                )

        target_fraction = target_sum / n_voxels
        row = {
            'stack': stack,
            'checkpoint_step': checkpoint_step,
            'dataset': sample['dataset'],
            'key': sample['key'],
            'modality': sample['modality'],
            'original_split': sample['original_split'],
            'dims': 2 if item['da'] is None else 3,
            'da': item['da'],
            'source': cls['source'],
            'class_name': cls['name'],
            'prompt': prompt,
            'volume_positive': cls['is_positive'],
            'target_positive': cls['is_positive'],
            'target_fraction': target_fraction,
            'target_size_bin': _size_bin(target_fraction) if cls['is_positive'] else None,
            'predicted_foreground_fraction': predicted_sum / n_voxels,
            'false_positive': bool(predicted_sum) if not cls['is_positive'] else None,
        }
        if cls['is_positive']:
            tp = int(prediction[target].sum().item())
            row.update(_positive_overlap_metrics(tp, predicted_sum, target_sum))
        else:
            row.update({metric: None for metric in ('dice', 'iou', 'precision', 'recall')})
        rows.append(row)
    return rows


def _mean(rows: list[dict], metric: str) -> float | None:
    values = [row[metric] for row in rows if row[metric] is not None]
    return sum(values) / len(values) if values else None


def _macro(rows: list[dict], metric: str, group_key: str) -> float | None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            groups[str(row[group_key])].append(row)
    values = [_mean(group, metric) for group in groups.values()]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize_metric_rows(rows: list[dict]) -> dict:
    """Aggregate per-concept rows without allowing large datasets to dominate every view."""
    positive = [row for row in rows if row['target_positive']]
    negative = [row for row in rows if not row['target_positive']]
    metrics = ('dice', 'iou', 'precision', 'recall')
    summary = {
        'n_rows': len(rows),
        'n_positive': len(positive),
        'n_negative': len(negative),
        'positive_case_concept_macro': {metric: _mean(positive, metric) for metric in metrics},
        'positive_prompt_macro': {metric: _macro(positive, metric, 'prompt') for metric in metrics},
        'positive_dataset_macro': {metric: _macro(positive, metric, 'dataset') for metric in metrics},
        'negative': {
            'false_positive_concept_rate': _mean(negative, 'false_positive'),
            'predicted_foreground_fraction': _mean(negative, 'predicted_foreground_fraction'),
        },
    }

    strata = {
        'target_size_bin': positive,
        'dims': positive,
        'da': positive,
        'modality': positive,
        'dataset': positive,
    }
    summary['strata'] = {}
    for key, source_rows in strata.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in source_rows:
            groups[str(row[key])].append(row)
        summary['strata'][key] = {
            value: {'n': len(group), **{metric: _mean(group, metric) for metric in metrics}}
            for value, group in sorted(groups.items())
        }
    return summary
