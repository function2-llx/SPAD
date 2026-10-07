"""Generation-time class classification and replay-time segmentation targets."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pumit.segmentation_mask import pack_binary_mask
from pumit.text_prompt import SegmentationPromptResolver
from pumit.ucpt.affine import _affine_resample, stream_crop_size
from pumit.ucpt.fg_coords import FgCoordCache
from pumit.ucpt.mask import _load_mask_crop
from pumit.ucpt.mask_store import decode_positive_masks, encode_packed_positive_masks

from .class_sampling import RawClass, flatten_label_contract, sample_classes
from .label_contract import LabelContract
from .text_encoding import TextEmbeddingCache


def _bbox_intersects(
    bbox_start: np.ndarray,
    bbox_stop: np.ndarray,
    crop_start: np.ndarray,
    crop_stop: np.ndarray,
) -> bool:
    return bool(np.all(bbox_start < crop_stop) and np.all(crop_start < bbox_stop))


def _resample_mask_crops(
    crops: list[np.ndarray] | np.ndarray,
    affine_4x4: np.ndarray,
    output_size: list[int],
) -> torch.Tensor:
    if len(crops) == 0:
        return torch.empty((0, *output_size), dtype=torch.bool)
    crop = torch.from_numpy(crops if isinstance(crops, np.ndarray) else np.stack(crops))
    result = _affine_resample(
        crop,
        affine_4x4,
        output_size,
        interp_mode='nearest-exact',
        grid_mode='nearest',
        pad_mode='replicate',
    )
    return result.bool()


class CropClassSampler:
    """Classify every raw class on the frozen full-resolution crop, then sample queries."""

    def __init__(
        self,
        *,
        data_root: Path,
        fg_cache: FgCoordCache,
        positive_queries: int,
        negative_queries: int,
        min_positive_voxels: int,
        mask_batch_size: int,
    ):
        for name, value in {
            'positive_queries': positive_queries,
            'negative_queries': negative_queries,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer, got {value!r}')
        if positive_queries + negative_queries < 1:
            raise ValueError('at least one segmentation query must be enabled')
        if (
            isinstance(min_positive_voxels, bool)
            or not isinstance(min_positive_voxels, int)
            or min_positive_voxels < 1
        ):
            raise ValueError(
                f'min_positive_voxels must be a positive integer, got {min_positive_voxels!r}'
            )
        if isinstance(mask_batch_size, bool) or not isinstance(mask_batch_size, int) or mask_batch_size < 1:
            raise ValueError(f'mask_batch_size must be a positive integer, got {mask_batch_size!r}')
        self.data_root = data_root
        self.fg_cache = fg_cache
        self.positive_queries = positive_queries
        self.negative_queries = negative_queries
        self.min_positive_voxels = min_positive_voxels
        self.mask_batch_size = mask_batch_size
        self._mask_load_pool = ThreadPoolExecutor(
            max_workers=mask_batch_size,
            thread_name_prefix='seg-mask-load',
        )

    def _prepare(
        self,
        record: dict,
        label_contract: LabelContract,
        params: list,
    ) -> dict:
        positive_pairs, explicit_negative_pairs = flatten_label_contract(label_contract)
        sp = params[0]
        crop_size = stream_crop_size(sp)
        load_start = np.asarray(sp['load_slice_start'], dtype=np.int64)
        load_stop = np.asarray(sp['load_slice_stop'], dtype=np.int64)
        crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(load_start, load_stop))
        shape_3d = tuple(int(value) for value in record['shape'])
        affine_4x4 = np.asarray(sp['affine'], dtype=np.float64).reshape(4, 4)
        dataset = record['dataset']
        key = record['key']

        classified_positive: list[dict[str, str | bool | int]] = []
        classified_negative: list[dict[str, str | bool | int]] = [
            {
                'source': source,
                'name': name,
                'is_positive': False,
                'target_voxels': 0,
            }
            for source, name in explicit_negative_pairs
        ]
        unresolved: list[RawClass] = []
        for source, name in positive_pairs:
            try:
                bbox_start, bbox_stop = self.fg_cache.bbox(dataset, key, source, name)
            except KeyError:
                raise KeyError(
                    f'missing foreground sidecar entry: dataset={dataset!r} key={key!r} '
                    f'source={source!r} class={name!r}'
                ) from None
            item = {
                'source': source,
                'name': name,
                'is_positive': False,
                'target_voxels': 0,
            }
            if _bbox_intersects(bbox_start, bbox_stop, load_start, load_stop):
                unresolved.append((source, name))
            else:
                classified_negative.append(item)
        return {
            'affine_4x4': affine_4x4,
            'classified_negative': classified_negative,
            'classified_positive': classified_positive,
            'crop_size': crop_size,
            'crop_slices': crop_slices,
            'dataset': dataset,
            'explicit_negative_masks': len(explicit_negative_pairs),
            'key': key,
            'positive_masks': len(positive_pairs),
            'shape_3d': shape_3d,
            'unresolved': unresolved,
        }

    def estimate_work(
        self,
        record: dict,
        label_contract: LabelContract,
        params: list,
    ) -> dict[str, int]:
        """Estimate classification work without loading or resampling masks."""
        setup = self._prepare(record, label_contract, params)
        unresolved_masks = len(setup['unresolved'])
        return {
            'positive_masks': setup['positive_masks'],
            'explicit_negative_masks': setup['explicit_negative_masks'],
            'unresolved_masks': unresolved_masks,
            'resample_voxels': unresolved_masks * int(np.prod(setup['crop_size'])),
        }

    def sample_with_stats(
        self,
        record: dict,
        label_contract: LabelContract,
        params: list,
        rng: np.random.Generator,
        *,
        focus: RawClass | None,
    ) -> tuple[list[dict[str, str | bool | int]], dict[str, int]]:
        """Return final raw-class queries and mask-resampling work."""
        classes, _, stats = self._sample(
            record,
            label_contract,
            params,
            rng,
            focus=focus,
            materialize=False,
        )
        return classes, stats

    def sample_materialized_with_stats(
        self,
        record: dict,
        label_contract: LabelContract,
        params: list,
        rng: np.random.Generator,
        *,
        focus: RawClass | None,
    ) -> tuple[list[dict[str, str | bool | int]], bytes, dict[str, int]]:
        """Return final queries, their selected positive masks, and resampling work."""
        classes, frame, stats = self._sample(
            record,
            label_contract,
            params,
            rng,
            focus=focus,
            materialize=True,
        )
        assert frame is not None
        return classes, frame, stats

    def _sample(
        self,
        record: dict,
        label_contract: LabelContract,
        params: list,
        rng: np.random.Generator,
        *,
        focus: RawClass | None,
        materialize: bool,
    ) -> tuple[list[dict[str, str | bool | int]], bytes | None, dict[str, int]]:
        setup = self._prepare(record, label_contract, params)
        affine_4x4 = setup['affine_4x4']
        classified_negative = setup['classified_negative']
        classified_positive = setup['classified_positive']
        crop_size = setup['crop_size']
        crop_slices = setup['crop_slices']
        dataset = setup['dataset']
        key = setup['key']
        shape_3d = setup['shape_3d']
        unresolved = setup['unresolved']
        packed_positive_targets: dict[RawClass, np.ndarray] = {}

        for offset in range(0, len(unresolved), self.mask_batch_size):
            pairs = unresolved[offset:offset + self.mask_batch_size]

            def load_crop(pair: RawClass) -> np.ndarray:
                source, name = pair
                crop = _load_mask_crop(
                    data_root=self.data_root,
                    dataset=dataset,
                    key=key,
                    source=source,
                    cls_name=name,
                    shape_3d=shape_3d,
                    crop_slices=crop_slices,
                )
                if crop is None:
                    raise FileNotFoundError(
                        f'missing positive mask: dataset={dataset!r} key={key!r} '
                        f'source={source!r} class={name!r}'
                    )
                return crop

            crops = list(self._mask_load_pool.map(load_crop, pairs))
            with torch.inference_mode():
                targets = _resample_mask_crops(
                    crops,
                    affine_4x4,
                    crop_size,
                )
                counts = targets.flatten(1).count_nonzero(dim=1).cpu().tolist()
            for pair, target, count in zip(pairs, targets, counts, strict=True):
                source, name = pair
                item = {
                    'source': source,
                    'name': name,
                    'is_positive': count >= self.min_positive_voxels,
                    'target_voxels': count,
                }
                if count >= self.min_positive_voxels:
                    classified_positive.append(item)
                    if materialize:
                        packed_positive_targets[pair] = pack_binary_mask(target.cpu().numpy())
                elif count == 0:
                    classified_negative.append(item)

        classes = sample_classes(
            classified_positive,
            classified_negative,
            positive_queries=self.positive_queries,
            negative_queries=self.negative_queries,
            rng=rng,
            focus=focus,
        )
        frame = None
        if materialize:
            selected_packed = [
                packed_positive_targets[(str(item['source']), str(item['name']))]
                for item in classes
                if item['is_positive']
            ]
            frame = encode_packed_positive_masks(selected_packed, shape=crop_size)
        return classes, frame, {
            'positive_masks': setup['positive_masks'],
            'explicit_negative_masks': setup['explicit_negative_masks'],
            'unresolved_masks': len(unresolved),
            'resample_voxels': len(unresolved) * int(np.prod(crop_size)),
        }

    def __call__(
        self,
        record: dict,
        label_contract: LabelContract,
        params: list,
        rng: np.random.Generator,
        *,
        focus: RawClass | None,
    ) -> list[dict[str, str | bool | int]]:
        """Return final raw-class queries for one frozen spatial transform."""
        classes, _ = self.sample_with_stats(
            record,
            label_contract,
            params,
            rng,
            focus=focus,
        )
        return classes


def build_seg_payload(
    sample: dict,
    patch_grid: tuple[int, int, int],
    da: int | None,
    *,
    data_root: Path,
    prompt_resolver: SegmentationPromptResolver,
    text_cache: TextEmbeddingCache,
    text_rng: np.random.Generator | None = None,
    positive_mask_frame: bytes | None = None,
) -> dict:
    """Reconstruct selected full-resolution targets and text inputs for one frozen crop."""
    classes = sample['classes']
    dataset = sample['dataset']
    key = sample['key']
    modality = sample['modality']
    sp = sample['spatial'] if 'spatial' in sample else sample['params'][0]
    crop_size = stream_crop_size(sp)

    positive_indices = [index for index, item in enumerate(classes) if item['is_positive']]
    if positive_mask_frame is None:
        affine_4x4 = np.asarray(sp['affine'], dtype=np.float64).reshape(4, 4)
        load_start = sp['load_slice_start']
        load_stop = sp['load_slice_stop']
        crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(load_start, load_stop))
        shape_3d = tuple(int(value) for value in np.load(sample['img'], mmap_mode='r').shape[1:])
        positive_crops: list[np.ndarray] = []
        for index in positive_indices:
            item = classes[index]
            crop = _load_mask_crop(
                data_root=data_root,
                dataset=dataset,
                key=key,
                source=item['source'],
                cls_name=item['name'],
                shape_3d=shape_3d,
                crop_slices=crop_slices,
            )
            if crop is None:
                raise FileNotFoundError(
                    f'missing positive mask: dataset={dataset!r} key={key!r} '
                    f"source={item['source']!r} class={item['name']!r}"
                )
            positive_crops.append(crop)
        positive_targets = _resample_mask_crops(
            positive_crops,
            affine_4x4,
            crop_size,
        )
    else:
        positive_targets = decode_positive_masks(
            positive_mask_frame,
            count=len(positive_indices),
            shape=crop_size,
        )
    targets_by_index = {
        index: target for index, target in zip(positive_indices, positive_targets, strict=True)
    }

    masks: list[torch.Tensor] = []
    for index, item in enumerate(classes):
        if item['is_positive']:
            target = targets_by_index[index]
            actual = int(target.count_nonzero())
            if positive_mask_frame is not None and actual != item['target_voxels']:
                raise ValueError(
                    f'materialized positive target count mismatch: dataset={dataset!r} key={key!r} '
                    f'source={item["source"]!r} class={item["name"]!r} '
                    f'metadata={item["target_voxels"]} materialized={actual}'
                )
            # Boundary voxels from rotated nearest resampling can differ across CPU hosts. The generation-time count
            # establishes positivity; replay only requires that the selected positive target remains non-empty.
            if actual == 0:
                raise RuntimeError(
                    f'positive target vanished during replay: dataset={dataset!r} key={key!r} '
                    f'source={item["source"]!r} class={item["name"]!r} '
                    f'generation={item["target_voxels"]} replay={actual}'
                )
            masks.append(target)
        else:
            if item['target_voxels'] != 0:
                raise ValueError(f'negative class has nonzero target_voxels: {item!r}')
            masks.append(torch.zeros(crop_size, dtype=torch.bool))

    label_keys = [(item['source'], item['name'], modality) for item in classes]
    prompts = prompt_resolver.get_batch(label_keys, rng=text_rng)
    text = text_cache.get_batch(prompts)
    text_valid_mask = text_cache.get_mask_batch(prompts)
    assert text_valid_mask.any(-1).all(), 'a concept has an all-pad token mask'
    return {
        'target_masks': torch.stack(masks, dim=0),
        'text_embeddings': text,
        'text_valid_mask': text_valid_mask,
        'is_positive': torch.tensor(
            [bool(item['is_positive']) for item in classes],
            dtype=torch.bool,
        ),
        'patch_grid': patch_grid,
        'da': da,
        'prompts': prompts,
    }
