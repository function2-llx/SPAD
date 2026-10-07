from dataclasses import asdict

import pytest
import torch

from pumit.ucpt.train.config import (
    UCPTModelConfig,
    UCPTOptimConfig,
    UCPTTrainConfig,
    parse_config,
)


def test_compile_cache_archive_uses_power_of_four_steps():
    from pumit.ucpt.train.__main__ import _compile_cache_archive_due

    assert not _compile_cache_archive_due(15)
    assert _compile_cache_archive_due(16)
    assert not _compile_cache_archive_due(32)
    assert _compile_cache_archive_due(64)
    assert not _compile_cache_archive_due(128)
    assert _compile_cache_archive_due(256)
    assert not _compile_cache_archive_due(257)
    assert _compile_cache_archive_due(4096)
    assert _compile_cache_archive_due(16384)
    assert not _compile_cache_archive_due(5000)


def test_inductor_compile_threads_depend_on_local_world_size():
    from pumit.ucpt.train.__main__ import _inductor_compile_threads

    assert _inductor_compile_threads(4) == 8
    assert _inductor_compile_threads(8) == 4
    with pytest.raises(ValueError, match='LOCAL_WORLD_SIZE must be positive'):
        _inductor_compile_threads(0)


def test_compile_cache_live_dir_is_namespaced_by_archive(tmp_path, monkeypatch):
    from pumit.ucpt.train.__main__ import _compile_cache_live_dir

    cache_root = tmp_path / 'live'
    monkeypatch.setenv('UCPT_COMPILE_CACHE_ROOT', str(cache_root))
    archive_a = tmp_path / 'a.tar.zst'
    archive_b = tmp_path / 'b.tar.zst'

    assert _compile_cache_live_dir(archive_a).parent == cache_root
    assert _compile_cache_live_dir(archive_a) == _compile_cache_live_dir(archive_a)
    assert _compile_cache_live_dir(archive_a) != _compile_cache_live_dir(archive_b)


def test_config_defaults():
    cfg = UCPTTrainConfig()
    assert cfg.data.virtual_lanes == 32
    assert cfg.model.latent_channels == 32
    assert cfg.model.recon_weight == 1.0
    assert cfg.model.seg_weight == 1.0
    assert cfg.optim.seg_lr_fraction == 0.3
    assert cfg.optim.ssl_lr_fraction == 1.0
    assert cfg.model.teacher_momentum == 0.996
    assert cfg.model.artifact_momentum == 0.996
    assert cfg.model.load_sam3_text_projection is False
    assert cfg.optim.cooldown_steps == 20000
    assert cfg.optim.min_lr == 1e-6
    assert cfg.run.compile_prewarm_batches == 2
    assert cfg.run.num_workers == 4 and cfg.run.augment_threads == 6
    assert cfg.run.numa_shared is True
    assert cfg.run.nccl_high_priority is True
    assert cfg.run.ddp_bucket_cap_mb == 50
    # No prototype-path fields.
    for forbidden in ('teacher_temp', 'distill_prototypes', 'center_momentum',
                      'patch_objective', 'proto_freeze_steps'):
        assert not hasattr(cfg.model, forbidden), f'prototype-path field {forbidden!r} survived'


def test_sam3_text_projection_requires_checkpoint():
    from pumit.ucpt.train.model_setup import build_model

    cfg = UCPTModelConfig(sam3_checkpoint='', load_sam3_text_projection=True)
    with pytest.raises(ValueError, match='requires sam3_checkpoint'):
        build_model(cfg, torch.device('cpu'))


def test_seg_prewarm_feat_matches_training_autograd_view():
    from pumit.ucpt.train.model_setup import _make_seg_prewarm_feat

    feat = _make_seg_prewarm_feat(
        d=4,
        h=8,
        w=8,
        embed_dim=16,
        device=torch.device('cpu'),
    )
    assert feat.shape == (1, 16, 4, 8, 8)
    assert feat.requires_grad
    feat.sum().backward()


def test_parse_config_reads_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / 'cfg.yaml'
    yaml_path.write_text(
        'optim:\n'
        '  lr: 3.0e-4\n'
        '  steps: 5000\n'
        '  seg_lr_fraction: 0.5\n'
    )
    monkeypatch.setattr('sys.argv', ['train.py', '--config', str(yaml_path)])
    cfg, no_compile, _cache_archive = parse_config()
    assert cfg.optim.lr == 3e-4
    assert cfg.optim.steps == 5000
    assert cfg.optim.seg_lr_fraction == 0.5
    assert no_compile is False


def test_parse_config_reads_optimizer_betas(tmp_path, monkeypatch):
    yaml_path = tmp_path / 'cfg.yaml'
    yaml_path.write_text(
        'optim:\n'
        '  lr: 1.0e-4\n'
        '  betas: [0.95, 0.999]\n'
    )
    monkeypatch.setattr('sys.argv', ['train.py', '--config', str(yaml_path)])
    cfg, _no_compile, _cache_archive = parse_config()
    assert cfg.optim.betas == (0.95, 0.999)
    assert cfg.optim.lr == 1.0e-4


def test_parse_config_applies_nested_cli_overrides(tmp_path, monkeypatch):
    yaml_path = tmp_path / 'cfg.yaml'
    yaml_path.write_text('optim:\n  lr: 1.0e-4\n')
    monkeypatch.setattr(
        'sys.argv',
        [
            'train.py',
            '--config', str(yaml_path),
            '--optim.lr', '2.0e-4',
            '--run.wandb_name', 'override',
            '--no-compile',
        ],
    )
    cfg, no_compile, _cache_archive = parse_config()
    assert cfg.optim.lr == 2.0e-4
    assert cfg.run.wandb_name == 'override'
    assert no_compile is True


def test_parse_config_merges_overlay_files(tmp_path, monkeypatch):
    base = tmp_path / 'base.yaml'
    base.write_text('model:\n  recon_weight: 1.0\nrun:\n  save_dir: baseline\n')
    overlay = tmp_path / 'overlay.yaml'
    overlay.write_text('model:\n  recon_weight: 0.0\nrun:\n  save_dir: ablation\n')
    monkeypatch.setattr(
        'sys.argv',
        ['train.py', '--config', str(base), '--config', str(overlay)],
    )

    cfg, _no_compile, _cache_archive = parse_config()

    assert cfg.model.recon_weight == 0.0
    assert cfg.run.save_dir == 'ablation'


def test_parse_config_rejects_unknown_keys(tmp_path, monkeypatch):
    yaml_path = tmp_path / 'cfg.yaml'
    yaml_path.write_text('optim:\n  lerning_rate: 1.0e-4\n')
    monkeypatch.setattr('sys.argv', ['train.py', '--config', str(yaml_path)])
    with pytest.raises(SystemExit):
        parse_config()


def test_build_model_structure(monkeypatch):
    """build_model returns a UCPTModel with 2 twins, no patch_distill_head,
    no distill_loss, prototype path fully absent. Uses tiny synthetic
    'pretrained' checkpoints so it runs on CPU without real weights."""
    import torch
    from safetensors.torch import save_file
    from pumit.model.vit import ViT, ViTConfig
    from pumit.ucpt.model import UCPTModel
    from pumit.ucpt.train.model_setup import build_model

    # Build a real ViT to get a valid state dict, save it as the 'pretrained' ckpt.
    vit_cfg = ViTConfig(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=256, num_register_tokens=2,
    )
    vit_sd = ViT(vit_cfg).state_dict()
    # Add the mask_token / inv_freq the loader whitelists as 'unexpected'.
    vit_sd['embeddings.mask_token'] = torch.zeros(1, 1, 128)
    import tempfile, os
    tmp = tempfile.mkdtemp()
    dino_path = os.path.join(tmp, 'dino.safetensors')
    save_file(vit_sd, dino_path)

    cfg = UCPTModelConfig(
        pretrained=dino_path, sam3_checkpoint='', embed_dim=128, depth=2,
        num_heads=4, mlp_ratio=2.0, n_register_tokens=2,
        recon_decoder_dim=64, recon_decoder_heads=4,
        patch_distill_decoder_dim=64, patch_distill_decoder_heads=4,
        latent_channels=16, cls_predictor_hidden=128, text_embed_dim=1152,
        seg_hidden_size=256, recon_weight=0.5, seg_weight=0.25, teacher_momentum=0.99,
        artifact_momentum=0.999, drop_path_rate=0.1,
    )
    device = torch.device('cpu')
    # SAM 3 checkpoint absent: build_model skips the SAM3 load when sam3_checkpoint is ''.
    model = build_model(cfg, device)
    assert isinstance(model, UCPTModel)
    assert model.recon_weight == 0.5
    assert model.seg_weight == 0.25
    assert model.teacher_momentum == 0.99
    assert model.artifact_momentum == 0.999
    assert model.vit.layer[-1].drop_path.drop_prob == pytest.approx(0.1)
    assert not hasattr(model, 'patch_distill_head')
    assert not hasattr(model, 'distill_loss')
    assert not hasattr(model, 'teacher_patch_distill_head')
    # 2 EMA twins.
    twin_roots = {k.split('.')[0] for k in model.state_dict()
                  if k.startswith(('teacher_', 'ema_'))}
    assert twin_roots == {'teacher_vit', 'ema_seg'}
    # Build-order invariant: twins are deepcopies of the LOADED student, so
    # they must equal the student at construction (before EMA diverges them).
    # A build_model that loaded after UCPTModel(...) would leave twins on init
    # weights and fail this.
    for p_vit, p_twin in zip(model.vit.parameters(), model.teacher_vit.parameters()):
        assert torch.equal(p_vit, p_twin), \
            'teacher_vit twin does not match the loaded vit (build-order broken)'
    for p_seg, p_twin in zip(model.seg.parameters(), model.ema_seg.parameters()):
        assert torch.equal(p_seg, p_twin), \
            'ema_seg twin does not match the loaded seg (build-order broken)'


def test_param_groups_6way():
    """6-way builder: vit/vit_no_decay/ssl/ssl_no_decay/seg/seg_no_decay.
    Twins + frozen pixel-decoder pair excluded by requires_grad. Every
    trainable seg.* in a seg group; every SSL-scaffolding param in an ssl group."""
    import torch
    from pumit.model.vit import ViT, ViTConfig
    from pumit.ucpt.model import UCPTModel, SegDecoderStack
    from pumit.ucpt.ssl.heads import (
        ReconDecoder, PatchDistillDecoder, ClsPredictor,
    )
    from pumit.ucpt.train.optim import _SSL_PREFIXES, build_param_groups

    vit_cfg = ViTConfig(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=256, num_register_tokens=2,
    )
    vit = ViT(vit_cfg)
    seg = SegDecoderStack(embed_dim=128, text_embed_dim=1152, hidden_size=256)
    model = UCPTModel(
        vit=vit,
        recon_decoder=ReconDecoder(encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, latent_channels=16),
        patch_distill_decoder=PatchDistillDecoder(encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, output_dim=128),
        cls_predictor=ClsPredictor(embed_dim=128, hidden_dim=128),
        seg=seg,
    )
    cfg = UCPTOptimConfig(lr=1.5e-4, seg_lr_fraction=0.3, ssl_lr_fraction=1.0, weight_decay=0.05)
    groups = build_param_groups(model, cfg)
    by_name = {g['group_name']: g for g in groups}
    assert set(by_name) == {'vit', 'vit_no_decay', 'ssl', 'ssl_no_decay', 'seg', 'seg_no_decay'}

    # LR scaling
    assert by_name['seg']['lr'] == pytest.approx(0.3 * 1.5e-4)
    assert by_name['seg_no_decay']['lr'] == pytest.approx(0.3 * 1.5e-4)
    assert by_name['ssl']['lr'] == pytest.approx(1.0 * 1.5e-4)
    assert by_name['vit']['lr'] == pytest.approx(1.5e-4)

    # weight_decay: decay groups = cfg.weight_decay, no_decay groups = 0
    assert by_name['vit']['weight_decay'] == 0.05
    assert by_name['vit_no_decay']['weight_decay'] == 0.0
    assert by_name['seg']['weight_decay'] == 0.05
    assert by_name['seg_no_decay']['weight_decay'] == 0.0

    # Coverage: every trainable param in exactly one group; twins + frozen pair excluded.
    all_trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    grouped = set()
    for g in groups:
        grouped.update(g['params_names'])
    assert all_trainable == grouped, 'trainable params lost or duplicated across groups'

    # Every trainable seg.* in a seg group; every SSL-scaffolding param in an ssl group.
    seg_trainable = {n for n in all_trainable if n.startswith('seg.')}
    seg_grouped = set(by_name['seg']['params_names']) | set(by_name['seg_no_decay']['params_names'])
    assert seg_trainable == seg_grouped, 'seg param leaked out of seg groups'

    ssl_trainable = {n for n in all_trainable if n.startswith(_SSL_PREFIXES)}
    ssl_grouped = set(by_name['ssl']['params_names']) | set(by_name['ssl_no_decay']['params_names'])
    assert ssl_trainable == ssl_grouped, 'ssl param leaked out of ssl groups'

    # The text resizer (2-D Linear weight) rides in the seg DECAY group.
    assert 'seg.text_encoder.resizer.weight' in by_name['seg']['params_names']
    assert 'seg.text_encoder.resizer.weight' not in by_name['seg_no_decay']['params_names']
    # Its bias (1-D) rides in the seg NO-DECAY group.
    assert 'seg.text_encoder.resizer.bias' in by_name['seg_no_decay']['params_names']

    # Twins excluded.
    for g in groups:
        for n in g['params_names']:
            assert not n.startswith(('teacher_', 'ema_')), f'twin param {n!r} reached optimizer'

    # No proto group.
    assert 'proto' not in by_name


def test_optimizer_scheduler_warmup_constant_and_cooldown(monkeypatch):
    import torch

    from pumit.ucpt.train.optim import make_optimizer_scheduler

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    params = [torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(()))]
    cfg = UCPTOptimConfig(steps=6, lr=1e-4, warmup_steps=2, cooldown_steps=2, min_lr=1e-6)
    optimizer, scheduler = make_optimizer_scheduler(cfg, [
        {'params': [params[0]], 'lr': cfg.lr},
        {'params': [params[1]], 'lr': cfg.lr * 0.5},
    ])

    observed_lrs = [scheduler.get_last_lr()]
    for _ in range(cfg.steps):
        for param in params:
            param.grad = torch.zeros_like(param)
        optimizer.step()
        scheduler.step()
        observed_lrs.append(scheduler.get_last_lr())

    assert observed_lrs[0] == pytest.approx([1e-6, 5e-7])
    assert observed_lrs[2] == pytest.approx([1e-4, 5e-5])
    assert observed_lrs[4] == pytest.approx([1e-4, 5e-5])
    assert observed_lrs[6] == pytest.approx([1e-6, 5e-7])


def test_step_loop_smoke_cpu(monkeypatch, tmp_path):
    """Exercise routed inputs, LLRD, packed drop path, EMA, and checkpoint resume."""
    import torch
    from pumit.model.vit import ViT, ViTConfig
    from pumit.ucpt.input import normalize_input
    from pumit.ucpt.model import UCPTModel, SegDecoderStack
    from pumit.ucpt.ssl.heads import (
        ReconDecoder, PatchDistillDecoder, ClsPredictor,
    )
    from pumit.ucpt.train.optim import build_param_groups, make_optimizer_scheduler

    # num_register_tokens=4 -> n_prefix=5, matching the _make_batch fixture's N_PREFIX.
    vit_cfg = ViTConfig(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=256, num_register_tokens=4, drop_path_rate=0.1,
    )
    vit = ViT(vit_cfg)
    seg = SegDecoderStack(embed_dim=128, text_embed_dim=1152, hidden_size=256)
    model = UCPTModel(
        vit=vit,
        recon_decoder=ReconDecoder(encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, latent_channels=16),
        patch_distill_decoder=PatchDistillDecoder(encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, output_dim=128),
        cls_predictor=ClsPredictor(embed_dim=128, hidden_dim=128),
        seg=seg,
    )
    cfg = UCPTOptimConfig(
        lr=1e-4, layer_decay=0.95, warmup_steps=1, cooldown_steps=0, steps=2, weight_decay=0.05,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    from tests.ucpt.conftest import use_cpu_attention_reference
    use_cpu_attention_reference(monkeypatch)

    groups = build_param_groups(model, cfg)
    optimizer, scheduler = make_optimizer_scheduler(cfg, groups)
    # On CUDA the optimizer must take the fused path (explicit fused=True accepts the
    # NoWeightDecayParameter subclass that default foreach dispatch rejects); the
    # optimizer.step() below then exercises fused acceptance of the parameter groups.
    if torch.cuda.is_available():
        assert optimizer.defaults['fused'] is True

    # Reuse the test_ucpt_model._make_batch fixture for a real UCPTBatch.
    from tests.ucpt.test_ucpt_model import _make_batch, _dtype, cast_ucpt_batch_dtype
    batch = _make_batch(n_labeled=1, k=2)
    medical = torch.linspace(-3, 5, batch.n_ssl_patches * 16 ** 3).reshape(-1, 1, 16, 16, 16)
    batch.patches[:batch.n_ssl_patches] = normalize_input(medical, 'zscore', batched=True)
    display = torch.rand_like(batch.patches[batch.n_ssl_patches:])
    display = (
        display - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    batch.patches[batch.n_ssl_patches:] = normalize_input(display, 'rgb', batched=True)
    dt = _dtype()
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)

    # One step.
    optimizer.zero_grad()
    out = model(batch)
    assert torch.isfinite(out.loss)
    assert out.seg_loss.item() > 0, 'seg_loss is 0 — labeled sample did not reach seg forward'
    out.loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        model.ema_update()

    # EMA moved the twins.
    # (twins started as deepcopies; after 1 ema_update toward the perturbed-by-backprop
    # student, they should differ from their init — hard to assert without a baseline,
    # so just assert no crash + finite.)

    # Checkpoint save + strict resume.
    from pumit.train_utils import save_checkpoint
    from pumit.ucpt.train.state import make_checkpoint_state, restore_checkpoint
    tmp = tmp_path
    state = make_checkpoint_state(
        run_id='test-run',
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        config=asdict(UCPTTrainConfig(optim=cfg)),
    )
    save_checkpoint(state, tmp)
    fresh = UCPTModel(
        vit=ViT(vit_cfg), recon_decoder=ReconDecoder(encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, latent_channels=16),
        patch_distill_decoder=PatchDistillDecoder(encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, output_dim=128),
        cls_predictor=ClsPredictor(embed_dim=128, hidden_dim=128),
        seg=SegDecoderStack(embed_dim=128, text_embed_dim=1152, hidden_size=256),
    ).to(device=device, dtype=model.vit.layer[0].attention.q_proj.weight.dtype)
    fresh_optimizer, fresh_scheduler = make_optimizer_scheduler(cfg, build_param_groups(fresh, cfg))
    restore_checkpoint(
        tmp / 'checkpoint-latest.pt',
        checkpoint_step=1,
        run_id='test-run',
        model=fresh,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        device=device,
    )
    # Resume round-trip: the loaded weights match the saved model (not just
    # structurally loadable — values too).
    assert torch.equal(
        model.vit.layer[0].attention.q_proj.weight,
        fresh.vit.layer[0].attention.q_proj.weight,
    ), 'resumed vit q_proj weight does not match the saved model'
