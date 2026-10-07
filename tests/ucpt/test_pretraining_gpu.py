"""CUDA compilation and NCCL checks for the continued-pretraining recipe."""

from copy import deepcopy
from dataclasses import fields
from datetime import timedelta
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from pumit.model.vit import ViT, ViTConfig
from pumit.ucpt.model import SegDecoderStack, UCPTModel
from pumit.ucpt.ssl.heads import ClsPredictor, PatchDistillDecoder, ReconDecoder
from pumit.ucpt.train.config import UCPTOptimConfig
from pumit.ucpt.train.optim import build_param_groups, make_optimizer_scheduler
from tests.ucpt.test_ucpt_model import _make_batch


def _vit_config(*, checkpoint_layers=None):
    return ViTConfig(
        hidden_size=128, num_hidden_layers=3, num_attention_heads=4,
        intermediate_size=256, num_register_tokens=4,
        drop_path_rate=0.1, grad_ckpt=True, grad_ckpt_first_n_layers=checkpoint_layers,
    )


def _mixed_model(*, checkpoint_layers=None):
    return UCPTModel(
        vit=ViT(_vit_config(checkpoint_layers=checkpoint_layers)),
        recon_decoder=ReconDecoder(
            encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, latent_channels=16,
        ),
        patch_distill_decoder=PatchDistillDecoder(
            encoder_dim=128, decoder_dim=64, depth=1, num_heads=4, output_dim=128,
        ),
        cls_predictor=ClsPredictor(embed_dim=128, hidden_dim=128),
        seg=SegDecoderStack(embed_dim=128, fusion_grad_ckpt=True),
    )


def _assert_finite_gradients(module):
    missing = [name for name, p in module.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f'trainable parameters without gradients: {missing}'
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA and xformers')
def test_compiled_packed_vit_bf16_cuda():
    device = torch.device('cuda:0')
    # Match the production compile recipe; eager checkpointing is exercised by the NCCL test.
    vit = ViT(_vit_config(checkpoint_layers=0), skip_embed=True).to(device).train()
    teacher = deepcopy(vit).requires_grad_(False).eval()
    eager_teacher = deepcopy(teacher)
    assert [getattr(block.drop_path, 'drop_prob', 0) for block in vit.layer] == pytest.approx([0, 0.05, 0.1])
    vit.compile(fullgraph=True, dynamic=True)
    teacher.compile(fullgraph=True, dynamic=True)

    for lengths in ([11, 17, 13], [17, 13, 11], [9, 15, 7, 13]):
        x = torch.randn(1, sum(lengths), 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        rope = torch.zeros(1, sum(lengths), 2, 32, device=device, dtype=torch.bfloat16)
        rope[..., 0, :] = 1
        bias = BlockDiagonalMask.from_seqlens(lengths, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = vit(x, rope, bias)
            target = teacher(x.detach(), rope, bias)
            expected_target = eager_teacher(x.detach(), rope, bias)
            loss = output.float().sin().mean()
        assert torch.isfinite(loss)
        torch.testing.assert_close(target, expected_target, atol=2e-2, rtol=2e-2)
        assert not target.requires_grad
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        _assert_finite_gradients(vit)
        assert all(p.grad is None for p in teacher.parameters())
        vit.zero_grad(set_to_none=True)


def _ddp_worker(rank, world_size, init_method, compile_vit):
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group(
        'nccl', rank=rank, world_size=world_size, init_method=init_method,
        timeout=timedelta(minutes=5), device_id=device,
    )
    try:
        torch.manual_seed(42)
        model = _mixed_model(checkpoint_layers=0 if compile_vit else None).to(device).train()
        if compile_vit:
            model.vit.compile(dynamic=True)
            model.teacher_vit.compile(dynamic=True)
        cfg = UCPTOptimConfig(
            lr=1e-4, layer_decay=0.95, ssl_lr_fraction=1.0, seg_lr_fraction=1.0,
            steps=2, warmup_steps=1, cooldown_steps=0,
        )
        groups = build_param_groups(model, cfg)
        rates = {name: group['lr'] for group in groups for name in group['params_names']}
        assert rates['vit.embeddings.patch_embeddings.weight'] == pytest.approx(1e-4 * 0.95 ** 3)
        assert rates['vit.layer.2.attention.q_proj.weight'] == pytest.approx(1e-4)
        optimizer, scheduler = make_optimizer_scheduler(cfg, groups)
        initial_patch = model.vit.patch_embed.weight.detach().clone()
        batches = []
        for step in range(2):
            torch.manual_seed(1000 + rank * 10 + step)
            batch = _make_batch(n_labeled=1 + (rank + step) % 2, k=2, n_ssl=1)
            batches.append(batch.to(device))
        if compile_vit:
            # Production warms both rank-local batches before installing DDP gradient hooks.
            with torch.random.fork_rng(devices=[device]):
                for batch in batches:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(batch, reduce_metrics=False)
                    output.loss.backward()
                    model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
        ddp = DDP(
            model, device_ids=[rank], gradient_as_bucket_view=True, batched_grad_copy=True,
            find_unused_parameters=False, static_graph=False,
        )

        for step, batch in enumerate(batches):
            torch.manual_seed(1000 + rank * 10 + step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = ddp(batch)
            assert all(torch.isfinite(getattr(output, field.name)).all() for field in fields(output))
            output.loss.backward()
            _assert_finite_gradients(model)
            assert not model.teacher_vit.training and not model.ema_seg.training
            assert all(p.grad is None for p in model.teacher_vit.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            model.ema_update()

            # Compare every online, frozen, and EMA parameter after each independent stochastic update.
            parameters = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            reference = parameters.clone()
            dist.broadcast(reference, src=0)
            assert torch.isfinite(parameters).all()
            torch.testing.assert_close(parameters, reference, rtol=0, atol=0)
        assert not torch.equal(initial_patch, model.vit.patch_embed.weight)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason='requires two CUDA devices and NCCL',
)
@pytest.mark.parametrize('compile_vit', [False, True], ids=['eager', 'compiled-vit'])
def test_mixed_ucpt_two_rank_nccl_step_and_ema(tmp_path, compile_vit):
    init_method = (tmp_path / 'nccl-init').as_uri()
    mp.spawn(_ddp_worker, args=(2, init_method, compile_vit), nprocs=2, join=True)
