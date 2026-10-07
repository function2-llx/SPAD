"""UCPT run identity, checkpoint, and automatic-resume state."""

import fcntl
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


RUN_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1


def make_checkpoint_state(
    *,
    run_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict,
) -> dict:
    """Build the canonical resume checkpoint payload."""
    return {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'run_id': run_id,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'config': config,
    }


def restore_checkpoint(
    checkpoint: Path,
    *,
    checkpoint_step: int,
    run_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> None:
    """Validate and restore a committed resume checkpoint."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get('format_version') != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f'checkpoint format mismatch: {state.get("format_version")!r}')
    if state.get('run_id') != run_id:
        raise ValueError('checkpoint run_id does not match run.yaml')
    if state.get('step') != checkpoint_step:
        raise ValueError(
            f'checkpoint step mismatch: filename={checkpoint_step}, state={state.get("step")!r}',
        )
    model_state = {key.replace('_orig_mod.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])


def _canonicalize(value):
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'run config contains unsupported value: {value!r}')


def _write_yaml_atomic(path: Path, value: dict) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with open(tmp, 'x') as f:
            yaml.safe_dump(value, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    wandb_id: str
    world_size: int
    config: dict

    def as_dict(self) -> dict:
        return {
            'format_version': RUN_FORMAT_VERSION,
            'run_id': self.run_id,
            'wandb_id': self.wandb_id,
            'world_size': self.world_size,
            'config': self.config,
        }


@dataclass(frozen=True)
class StartupState:
    manifest: RunManifest
    checkpoint: Path | None
    checkpoint_step: int
    new_run: bool

    @property
    def wandb_resume(self) -> str:
        if self.new_run:
            return 'never'
        if self.checkpoint is None:
            return 'allow'
        return 'must'


class RunLock:
    """Process-lifetime exclusive lock for one UCPT output directory."""

    def __init__(self, file):
        self._file = file

    @classmethod
    def acquire(cls, save_dir: Path) -> 'RunLock':
        save_dir.mkdir(parents=True, exist_ok=True)
        lock_path = save_dir / '.run.lock'
        file = open(lock_path, 'a')
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            file.close()
            raise RuntimeError(f'another process is using output directory {save_dir}') from None
        return cls(file)

    def close(self) -> None:
        fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()


def _load_manifest(path: Path) -> RunManifest:
    with open(path) as f:
        raw = yaml.safe_load(f)
    expected = {'format_version', 'run_id', 'wandb_id', 'world_size', 'config'}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f'invalid run manifest schema: {path}')
    if raw['format_version'] != RUN_FORMAT_VERSION:
        raise ValueError(
            f'unsupported run format {raw["format_version"]!r}, expected {RUN_FORMAT_VERSION}'
        )
    if not isinstance(raw['run_id'], str) or not raw['run_id']:
        raise ValueError(f'invalid run_id in {path}')
    if not isinstance(raw['wandb_id'], str) or not raw['wandb_id']:
        raise ValueError(f'invalid wandb_id in {path}')
    if not isinstance(raw['world_size'], int) or raw['world_size'] < 1:
        raise ValueError(f'invalid world_size in {path}')
    if not isinstance(raw['config'], dict):
        raise ValueError(f'invalid config in {path}')
    return RunManifest(
        run_id=raw['run_id'],
        wandb_id=raw['wandb_id'],
        world_size=raw['world_size'],
        config=raw['config'],
    )


def _find_committed_checkpoint(save_dir: Path) -> tuple[Path | None, int]:
    latest = save_dir / 'checkpoint-latest.pt'
    if not latest.is_symlink():
        if latest.exists():
            raise ValueError(f'latest checkpoint pointer is not a symlink: {latest}')
        return None, 0

    target = Path(os.readlink(latest))
    if target.is_absolute() or target.parent != Path('checkpoints'):
        raise ValueError(f'invalid latest checkpoint target: {target}')
    checkpoint = (save_dir / target).resolve(strict=True)
    if checkpoint.parent != (save_dir / 'checkpoints').resolve():
        raise ValueError(f'latest checkpoint escapes checkpoints directory: {latest}')
    prefix = 'checkpoint-'
    suffix = '.pt'
    if not checkpoint.name.startswith(prefix) or not checkpoint.name.endswith(suffix):
        raise ValueError(f'invalid checkpoint filename: {checkpoint.name}')
    step_text = checkpoint.name[len(prefix):-len(suffix)]
    if not step_text.isdigit() or int(step_text) < 1:
        raise ValueError(f'invalid checkpoint step: {checkpoint.name}')
    return checkpoint, int(step_text)


def prepare_run(
    save_dir: Path,
    config: dict,
    world_size: int,
    *,
    wandb_id: str,
) -> StartupState:
    """Create or validate a run and return its committed training state."""
    canonical_config = _canonicalize(config)
    manifest_path = save_dir / 'run.yaml'
    new_run = not manifest_path.exists()

    if new_run:
        unexpected = sorted(
            p.name
            for p in save_dir.iterdir()
            if p.name != '.run.lock'
            and not (p.name.startswith('.run.yaml.') and p.name.endswith('.tmp'))
        )
        if unexpected:
            raise RuntimeError(
                f'output directory has no run.yaml but is not empty: {save_dir} ({unexpected})'
            )
        manifest = RunManifest(
            run_id=uuid.uuid4().hex,
            wandb_id=wandb_id,
            world_size=world_size,
            config=canonical_config,
        )
        _write_yaml_atomic(manifest_path, manifest.as_dict())
    else:
        manifest = _load_manifest(manifest_path)
        if manifest.config != canonical_config:
            keys = sorted(
                key
                for key in manifest.config.keys() | canonical_config.keys()
                if manifest.config.get(key) != canonical_config.get(key)
            )
            raise ValueError(f'run config mismatch for {save_dir}: {keys}')

    checkpoint, checkpoint_step = _find_committed_checkpoint(save_dir)
    return StartupState(
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_step=checkpoint_step,
        new_run=new_run,
    )
