import orjson
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pumit.model.vit import ViT, ViTConfig
from pumit.transforms.patchify import patchify
import pumit.ucpt.seg.evaluation as evaluation
from pumit.ucpt.seg.evaluation import (
    _balanced_subset,
    _positive_overlap_metrics,
    _sliding_geometry,
    full_volume_metric_rows,
    save_panel,
    SegmentationArtifact,
    sliding_window_logits,
    summarize_metric_rows,
)
from scripts.ucpt.eval_seg import _assign_case_indices, _merge_shards, _visible_devices


def test_balanced_subset_is_deterministic_and_round_robin():
    records = [
        {'dataset': dataset, 'modality': modality, 'key': f'{dataset}-{index}'}
        for dataset, modality, count in [('a', 'CT', 5), ('b', 'MRI', 2), ('c', 'CT', 1)]
        for index in range(count)
    ]
    selected = _balanced_subset(records, max_cases=5, seed=7)
    repeated = _balanced_subset(records, max_cases=5, seed=7)

    assert [record['key'] for record in selected] == [record['key'] for record in repeated]
    assert {record['dataset'] for record in selected[:3]} == {'a', 'b', 'c'}


def test_save_panel_is_idempotent_but_does_not_replace(tmp_path):
    path = tmp_path / 'panel.json'
    panel = {'format_version': 2, 'samples': []}
    save_panel(panel, path)
    save_panel(panel, path)

    with pytest.raises(FileExistsError, match='refusing to replace'):
        save_panel({'format_version': 2, 'samples': [{}]}, path)


def test_visible_devices_uses_every_logical_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    assert _visible_devices() == [0, 1, 2, 3]


def test_visible_devices_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    with pytest.raises(RuntimeError, match='at least one visible CUDA device'):
        _visible_devices()


def test_case_assignments_are_disjoint_complete_and_cost_balanced():
    samples = [
        {
            'sliding_window': {'num_windows': num_windows},
            'classes': [{}] * num_concepts,
        }
        for num_windows, num_concepts in [(40, 1), (15, 3), (20, 1), (5, 3), (5, 1), (5, 1)]
    ]
    assignments = _assign_case_indices(samples, 3)
    assert sorted(index for shard in assignments for index in shard) == list(range(len(samples)))
    assert all(set(left).isdisjoint(right) for i, left in enumerate(assignments) for right in assignments[i + 1:])
    loads = [
        sum(
            samples[index]['sliding_window']['num_windows'] * (1 + len(samples[index]['classes']))
            for index in shard
        )
        for shard in assignments
    ]
    assert max(loads) - min(loads) <= 10


def test_merge_shards_restores_panel_order(tmp_path):
    num_cases = 7
    num_workers = 3
    assignments = _assign_case_indices(
        [
            {'sliding_window': {'num_windows': index + 1}, 'classes': [{}]}
            for index in range(num_cases)
        ],
        num_workers,
    )
    for rank in range(num_workers):
        path = tmp_path / f'ema-rank-{rank}.jsonl'
        with open(path, 'wb') as file:
            for case_index in assignments[rank]:
                file.write(orjson.dumps({
                    'case_index': case_index,
                    'rows': [{'case_index': case_index}],
                }))
                file.write(b'\n')

    rows = _merge_shards(tmp_path, 'ema', num_workers, num_cases)
    assert [row['case_index'] for row in rows] == list(range(num_cases))


def test_sliding_geometry_aligns_windows_and_output_grid():
    geometry = _sliding_geometry((100, 510, 770), (32, 256, 256), da=2, overlap=0.25)
    assert geometry == {
        'roi_size': [32, 256, 256],
        'scan_interval': [24, 192, 192],
        'overlap': [0.25, 0.25, 0.25],
        'padded_shape': [104, 640, 832],
        'output_shape': [104, 160, 208],
        'num_windows': 48,
    }


def test_sliding_geometry_handles_2d_depth_axis():
    geometry = _sliding_geometry((1, 4459, 6648), (1, 384, 384), da=None, overlap=0.25)
    assert geometry['scan_interval'] == [1, 288, 288]
    assert geometry['padded_shape'] == [1, 4704, 6720]
    assert geometry['num_windows'] == 368


@pytest.mark.parametrize('da', [0, 1, 2, 3, 4])
def test_direct_spad_patch_embedding_matches_training_patchify(da):
    torch.manual_seed(0)
    vit = ViT(ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_register_tokens=0,
    ))
    patch_d = 16 >> da
    image = torch.randn(3, patch_d * 2, 32, 32)
    patches, grid = patchify(image, da)

    training = vit.patch_embed(patches, da=0).reshape(-1, vit.embed_dim)
    direct = vit.embeddings(image.unsqueeze(0), da=da).reshape(-1, vit.embed_dim)

    assert grid == (2, 2, 2)
    torch.testing.assert_close(direct, training, atol=5e-5, rtol=5e-5)


class _FakeSeg(nn.Module):
    def forward(self, features, text_embeddings, text_valid_mask, schedule):
        k = text_embeddings.shape[0]
        return features[:, :1].expand(k, -1, *features.shape[2:])


class _FakeVit(nn.Module):
    embed_dim = 2

    def encode_image(self, images, da):
        patch_d = 16 >> da
        n_patches = (images.shape[2] // patch_d) * (images.shape[3] // 16) * (images.shape[4] // 16)
        values = images.mean(dim=(1, 2, 3, 4), keepdim=True).reshape(images.shape[0], 1, 1)
        return values[:, 0], values.expand(-1, n_patches, self.embed_dim)


def test_segmentation_artifact_batches_vit_windows():
    torch.manual_seed(0)
    artifact = SegmentationArtifact(_FakeVit(), _FakeSeg()).eval()
    images = torch.randn(2, 3, 16, 32, 32)
    text = torch.randn(3, 2, 8)
    valid = torch.ones(3, 2, dtype=torch.bool)

    batched = artifact(images, 0, text, valid)
    separate = torch.stack([artifact(image, 0, text, valid) for image in images])
    torch.testing.assert_close(batched, separate)


class _RecordingArtifact(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, images, da, text_embeddings, text_valid_mask):
        self.batch_sizes.append(images.shape[0])
        pooled = F.avg_pool3d(images.mean(1, keepdim=True), kernel_size=(1, 4, 4), stride=(1, 4, 4))
        return pooled[:, None].expand(-1, text_embeddings.shape[0], -1, -1, -1, -1)


@pytest.mark.parametrize(
    ('interpolation_order', 'expected_shape'),
    [
        ('stitch-then-interpolate', (2, 1, 10, 10)),
        ('interpolate-then-stitch', (2, 1, 40, 40)),
    ],
)
def test_sliding_window_logits_batches_and_stitches_lower_resolution_output(interpolation_order, expected_shape):
    artifact = _RecordingArtifact()
    item = {
        'image': torch.full((1, 1, 40, 40), 0.75),
        'da': None,
        'text_embeddings': torch.randn(2, 3, 8),
        'text_valid_mask': torch.ones(2, 3, dtype=torch.bool),
        'sample': {
            'classes': [{}, {}],
            'sliding_window': {
                'roi_size': [1, 16, 16],
                'overlap': [0.0, 0.25, 0.25],
                'padded_shape': [1, 40, 40],
                'output_shape': [1, 10, 10],
            },
        },
    }

    logits = sliding_window_logits(
        artifact,
        item,
        device=torch.device('cpu'),
        sw_batch_size=4,
        interpolation_order=interpolation_order,
    )
    assert artifact.batch_sizes == [4, 4, 1]
    assert logits.shape == expected_shape
    torch.testing.assert_close(logits, torch.full_like(logits, 0.5))


def test_full_volume_metrics_separate_positive_and_negative_queries(monkeypatch):
    positive_target = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool)
    monkeypatch.setattr(evaluation, '_load_mask', lambda *args, **kwargs: positive_target.numpy())
    item = {
        'da': None,
        'prompts': ['liver prompt', 'kidney prompt'],
        'sample': {
            'dataset': 'ds',
            'key': 'case',
            'modality': 'CT',
            'original_split': 'test',
            'shape': [1, 2, 2],
            'sliding_window': {'padded_shape': [1, 2, 2]},
            'classes': [
                {'source': 'src', 'name': 'liver', 'is_positive': True},
                {'source': 'src', 'name': 'kidney', 'is_positive': False},
            ],
        },
    }
    logits = torch.tensor([
        [[[10.0, -10.0], [-10.0, -10.0]]],
        [[[10.0, -10.0], [-10.0, -10.0]]],
    ])

    rows = full_volume_metric_rows(
        logits,
        item,
        data_root=None,
        device=torch.device('cpu'),
        stack='ema',
        checkpoint_step=100,
        interpolation_order='interpolate-then-stitch',
    )
    summary = summarize_metric_rows(rows)

    assert rows[0]['dice'] == pytest.approx(1.0)
    assert rows[1]['dice'] is None
    assert rows[1]['false_positive'] is True
    assert rows[1]['predicted_foreground_fraction'] == pytest.approx(0.25)
    assert summary['n_positive'] == 1
    assert summary['n_negative'] == 1
    assert summary['negative']['false_positive_concept_rate'] == pytest.approx(1.0)

    empty_prediction = logits.clone()
    empty_prediction[0].fill_(-10)
    empty_rows = full_volume_metric_rows(
        empty_prediction,
        item,
        data_root=None,
        device=torch.device('cpu'),
        stack='ema',
        checkpoint_step=100,
        interpolation_order='interpolate-then-stitch',
    )
    assert empty_rows[0]['precision'] == 0.0
    assert empty_rows[0]['recall'] == 0.0


def test_positive_overlap_metrics():
    assert _positive_overlap_metrics(tp=3, predicted=4, target=5) == {
        'dice': pytest.approx(2 / 3),
        'iou': pytest.approx(0.5),
        'precision': pytest.approx(0.75),
        'recall': pytest.approx(0.6),
    }
