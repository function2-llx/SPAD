"""Tests for codec checkpoint save (scaler state inclusion)."""

import torch

from scripts.codec.train import _save_checkpoint


class _Sched:
    def state_dict(self):
        return {'last_epoch': 7}


def test_save_checkpoint_includes_scaler(tmp_path):
    model = torch.nn.Linear(4, 4)
    ema_shadow = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cpu')

    ckpt_path = _save_checkpoint(
        tmp_path, 100, model, ema_shadow,
        optimizer, _Sched(),
        scaler=scaler,
        wandb_id=None,
        config={},
    )

    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'scaler' in saved
    assert saved['scaler'] == scaler.state_dict()
