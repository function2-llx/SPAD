"""Rank-zero model-weight EMA for nnU-Net trainers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from numbers import Real

import torch
import torch.distributed as dist
from torch import nn
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn.parallel import DistributedDataParallel


DEFAULT_MODEL_EMA_DECAY = 0.9998


def validate_model_ema_decay(decay: object) -> float | None:
    """Validate the optional EMA decay stored in an experiment plan."""
    if decay is None:
        return None
    if isinstance(decay, bool) or not isinstance(decay, Real):
        raise TypeError('model_ema_decay must be a number or null')
    decay = float(decay)
    if not 0 < decay < 1:
        raise ValueError('model_ema_decay must be between 0 and 1')
    return decay


class ModelEmaMixin:
    """Maintain an FP32 EMA on rank zero and make it available for validation."""

    model_ema_decay: float | None = None

    def initialize(self) -> None:
        super().initialize()
        if hasattr(self, '_model_ema_state'):
            return
        self._model_ema_num_updates = 0
        self._model_ema_state: dict[str, torch.Tensor] | None = None
        if self.model_ema_decay is not None and self._owns_model_ema():
            self._initialize_model_ema_state()

    def _owns_model_ema(self) -> bool:
        return getattr(self, 'global_rank', 0) == 0

    def _model_for_ema(self) -> nn.Module:
        model = self.network
        if isinstance(model, DistributedDataParallel):
            model = model.module
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
        return model

    def _live_model_state(self) -> dict[str, torch.Tensor]:
        return dict(self._model_for_ema().state_dict())

    @torch.no_grad()
    def _initialize_model_ema_state(self) -> None:
        self._model_ema_state = {
            name: (
                value.detach().to(dtype=torch.float32, copy=True)
                if value.is_floating_point()
                else value.detach().clone()
            )
            for name, value in self._live_model_state().items()
        }

    @torch.no_grad()
    def _update_model_ema(self) -> None:
        """Update the rank-zero shadow after one optimizer step."""
        if self.model_ema_decay is None or not self._owns_model_ema():
            return
        if self._model_ema_state is None:
            raise RuntimeError('model EMA was not initialized')

        live_state = self._live_model_state()
        if live_state.keys() != self._model_ema_state.keys():
            raise RuntimeError('live model and EMA state keys differ')
        interpolation_weight = 1 - self.model_ema_decay
        direct_ema = []
        direct_live = []
        for name, live_value in live_state.items():
            ema_value = self._model_ema_state[name]
            if not live_value.is_floating_point():
                ema_value.copy_(live_value)
            elif live_value.dtype == ema_value.dtype:
                direct_ema.append(ema_value)
                direct_live.append(live_value)
            else:
                source = live_value.to(dtype=ema_value.dtype)
                if self._model_ema_num_updates == 0:
                    ema_value.copy_(source)
                else:
                    ema_value.lerp_(source, interpolation_weight)

        if self._model_ema_num_updates == 0:
            torch._foreach_copy_(direct_ema, direct_live)
        else:
            torch._foreach_lerp_(
                direct_ema,
                direct_live,
                interpolation_weight,
            )
        self._model_ema_num_updates += 1

    def _augment_checkpoint_payload(self, payload: object) -> object:
        """Attach the rank-zero EMA to nnU-Net's checkpoint payload."""
        if self.model_ema_decay is None or not self._owns_model_ema():
            return payload
        if not isinstance(payload, dict):
            raise TypeError(
                f'nnU-Net checkpoint payload must be a dict, got '
                f'{type(payload).__name__}'
            )
        if self._model_ema_state is None:
            raise RuntimeError('model EMA was not initialized')
        payload['model_ema_state_dict'] = self._model_ema_state
        payload['model_ema_num_updates'] = self._model_ema_num_updates
        payload['model_ema_decay'] = self.model_ema_decay
        return payload

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if self.model_ema_decay is not None and self._owns_model_ema():
            checkpoint_for_ema = (
                torch.load(
                    filename_or_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
                if isinstance(filename_or_checkpoint, str)
                else filename_or_checkpoint
            )
            self._restore_model_ema_checkpoint(checkpoint_for_ema)

    @torch.no_grad()
    def _restore_model_ema_checkpoint(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, Mapping):
            raise TypeError('model EMA checkpoint must be a mapping')
        required_keys = {
            'model_ema_state_dict',
            'model_ema_num_updates',
            'model_ema_decay',
        }
        missing_keys = required_keys - checkpoint.keys()
        if missing_keys:
            raise ValueError(
                f'EMA-enabled plan cannot resume a checkpoint missing '
                f'{sorted(missing_keys)}'
            )
        checkpoint_decay = validate_model_ema_decay(checkpoint['model_ema_decay'])
        if checkpoint_decay != self.model_ema_decay:
            raise ValueError(
                f'checkpoint model EMA decay {checkpoint_decay} does not '
                f'match plan decay {self.model_ema_decay}'
            )
        num_updates = checkpoint['model_ema_num_updates']
        if (
            isinstance(num_updates, bool)
            or not isinstance(num_updates, int)
            or num_updates < 0
        ):
            raise ValueError('model_ema_num_updates must be a nonnegative integer')
        saved_state = checkpoint['model_ema_state_dict']
        if not isinstance(saved_state, Mapping):
            raise TypeError('model_ema_state_dict must be a mapping')
        if self._model_ema_state is None:
            self._initialize_model_ema_state()
        assert self._model_ema_state is not None
        if saved_state.keys() != self._model_ema_state.keys():
            raise ValueError('checkpoint and live model EMA state keys differ')
        for name, ema_value in self._model_ema_state.items():
            saved_value = saved_state[name]
            if not isinstance(saved_value, torch.Tensor):
                raise TypeError(f'model EMA value {name!r} must be a tensor')
            if saved_value.shape != ema_value.shape:
                raise ValueError(
                    f'model EMA value {name!r} has shape {saved_value.shape}, '
                    f'expected {ema_value.shape}'
                )
            ema_value.copy_(saved_value)
        self._model_ema_num_updates = num_updates

    @contextmanager
    def model_ema_weights(self) -> Iterator[None]:
        """Temporarily install rank-zero EMA weights on every DDP rank."""
        if self.model_ema_decay is None:
            raise RuntimeError('model EMA is disabled')
        live_state = self._live_model_state()
        # Host-side raw weights avoid a second full model copy on every validation GPU.
        raw_state = {
            name: value.detach().to('cpu', copy=True)
            for name, value in live_state.items()
        }
        try:
            if self._owns_model_ema():
                if self._model_ema_state is None:
                    raise RuntimeError('model EMA was not initialized')
                if self._model_ema_state.keys() != live_state.keys():
                    raise RuntimeError('live model and EMA state keys differ')
                with torch.no_grad():
                    for name, live_value in live_state.items():
                        live_value.copy_(self._model_ema_state[name])
            if dist.is_initialized():
                for live_value in live_state.values():
                    dist.broadcast(live_value, src=0)
            yield
        finally:
            with torch.no_grad():
                for name, live_value in live_state.items():
                    live_value.copy_(raw_state[name])
