"""UCPT model construction and segmentation decoder prewarming."""

import torch

from pumit.model.vit import ViT, ViTConfig
from pumit.ucpt.model import SegDecoderStack, UCPTModel
from pumit.ucpt.ssl.heads import ClsPredictor, PatchDistillDecoder, ReconDecoder

from .config import UCPTModelConfig


def _make_seg_prewarm_feat(
    d: int,
    h: int,
    w: int,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a decoder prewarm view matching the stride contract of encoder features."""
    n = d * h * w
    rows = torch.randn(
        n,
        embed_dim,
        device=device,
        requires_grad=True,
    )
    return rows.reshape(1, d, h, w, embed_dim).permute(0, 4, 1, 2, 3)


def build_model(cfg: UCPTModelConfig, device: torch.device) -> UCPTModel:
    import safetensors.torch as st
    from pumit.ucpt.seg.weight_load import load_sam3_weights

    if cfg.load_sam3_text_projection and not cfg.sam3_checkpoint:
        raise ValueError('load_sam3_text_projection requires sam3_checkpoint')

    # 1. DINOv3 ViT
    vit_config = ViTConfig(
        hidden_size=cfg.embed_dim,
        num_hidden_layers=cfg.depth,
        num_attention_heads=cfg.num_heads,
        intermediate_size=int(cfg.embed_dim * cfg.mlp_ratio),
        num_register_tokens=cfg.n_register_tokens,
        drop_path_rate=cfg.drop_path_rate,
        grad_ckpt=cfg.grad_ckpt,
        grad_ckpt_first_n_layers=cfg.grad_ckpt_first_n_layers,
    )
    vit = ViT(vit_config)
    pretrained_sd = st.load_file(cfg.pretrained)
    missing, unexpected = vit.load_state_dict(pretrained_sd, strict=False)
    allowed_unexpected = {'embeddings.mask_token', 'rope_embeddings.inv_freq'}
    actual_unexpected = set(unexpected) - allowed_unexpected
    if actual_unexpected:
        raise RuntimeError(f'Unexpected keys in DINOv3 checkpoint: {actual_unexpected}')
    if missing:
        raise RuntimeError(f'Missing keys loading DINOv3: {missing}')

    # 2. SAM 3 seg stack.
    seg = SegDecoderStack(
        embed_dim=cfg.embed_dim,
        text_embed_dim=cfg.text_embed_dim,
        hidden_size=cfg.seg_hidden_size,
        fusion_grad_ckpt=cfg.fusion_grad_ckpt,
    )
    if cfg.sam3_checkpoint:
        report = load_sam3_weights(
            cfg.sam3_checkpoint,
            neck=seg.neck,
            fusion=seg.fusion,
            head=seg.head,
            text_projection=seg.text_encoder.resizer if cfg.load_sam3_text_projection else None,
        )
        assert not report['unmatched_target'], (
            f'SAM 3 load left seg target keys unmatched: {report["unmatched_target"][:10]}'
        )

    # 3. SSL heads (no DistillHead / DistillLoss — JEPA only)
    recon_decoder = ReconDecoder(
        encoder_dim=cfg.embed_dim,
        decoder_dim=cfg.recon_decoder_dim,
        depth=cfg.recon_decoder_depth,
        num_heads=cfg.recon_decoder_heads,
        latent_channels=cfg.latent_channels,
    )
    patch_distill_decoder = PatchDistillDecoder(
        encoder_dim=cfg.embed_dim,
        decoder_dim=cfg.patch_distill_decoder_dim,
        depth=cfg.patch_distill_decoder_depth,
        num_heads=cfg.patch_distill_decoder_heads,
        output_dim=cfg.embed_dim,
    )
    cls_predictor = ClsPredictor(embed_dim=cfg.embed_dim, hidden_dim=cfg.cls_predictor_hidden)

    # 4. UCPTModel (deepcopies loaded weights into the 2 twins)
    model = UCPTModel(
        vit=vit,
        recon_decoder=recon_decoder,
        patch_distill_decoder=patch_distill_decoder,
        cls_predictor=cls_predictor,
        seg=seg,
        teacher_momentum=cfg.teacher_momentum,
        artifact_momentum=cfg.artifact_momentum,
        recon_weight=cfg.recon_weight,
        image_distill_weight=cfg.image_distill_weight,
        patch_distill_weight=cfg.patch_distill_weight,
        seg_weight=cfg.seg_weight,
    )
    return model.to(device)


def prewarm_seg_decoder(
    model: UCPTModel,
    text_cache_path: str,
    text_embed_dim: int,
    device: torch.device,
) -> None:
    """Compile the segmentation decoder guards used by training before DDP installs gradient hooks.

    Args:
        model: UCPT model whose fusion encoder and PixelDecoder have already been compiled.
        text_cache_path: Path to the text embedding cache.
        text_embed_dim: Text embedding width.
        device: Device holding the model and synthetic prewarm inputs.
    """
    from pumit.ucpt.seg.text_encoding import TextEmbeddingCache

    text_seq_len = TextEmbeddingCache(text_cache_path).seq_len
    embed = model.vit.embed_dim
    amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)
    fork_devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=fork_devices):
        for da in (None, 0, 1, 2, 3, 4):
            sizes = (((8, 8, 1), (72, 72, 1)) if da is None
                     else ((8, 8, 1), (8, 8, 4), (32, 32, 8)))
            for h, w, d in sizes:
                feat = _make_seg_prewarm_feat(d, h, w, embed, device)
                for k in (1, 4):  # PyTorch specializes singleton K separately from symbolic K>1.
                    text = torch.randn(k, text_seq_len, text_embed_dim, device=device)
                    valid_mask = torch.ones(k, text_seq_len, dtype=torch.bool, device=device)
                    with amp:
                        out = model.seg(feat, text, valid_mask, da)
                    out.float().sum().backward()
    model.zero_grad(set_to_none=True)
