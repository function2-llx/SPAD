from __future__ import annotations

import os
from pathlib import Path
import subprocess

import torch

import pumit.nnunet.compile_cache as nnunet_compile_cache_module
import pumit.nnunet.checkpointing as checkpointing_module
from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.nnunet.compile_cache import CompileCacheMixin
from pumit.spad_unet.experiments.spad_universal import SPADUniversalTrainer


REPOSITORY_ROOT = Path(__file__).parents[1]
DATASET = 'Dataset591_SPADCTUniversalV2'
PLAN = 'SPADLauncherTestPlans'


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def _launcher_project(
    tmp_path: Path,
    visible_gpus: int,
) -> tuple[Path, dict[str, str], Path, Path, Path, Path]:
    launcher = tmp_path / 'scripts' / 'spad_unet' / 'launch.sh'
    _write_executable(
        launcher,
        (REPOSITORY_ROOT / 'scripts' / 'spad_unet' / 'launch.sh').read_text(),
    )
    plan_path = tmp_path / 'nnUNet_data' / 'preprocessed' / DATASET / f'{PLAN}.json'
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text('{}')

    bin_dir = tmp_path / 'bin'
    train_args = tmp_path / 'train-args.txt'
    cache_archive = tmp_path / 'cache-archive.txt'
    tile_step_size = tmp_path / 'tile-step-size.txt'
    compile_mode = tmp_path / 'compile-mode.txt'
    _write_executable(
        bin_dir / 'python',
        (
            '#!/usr/bin/env bash\n'
            'if [[ $2 == *json.load* ]]; then\n'
            '    printf "%s\\n" cross\n'
            'elif [[ $2 == *"tile step size"* ]]; then\n'
            '    /usr/bin/python3 -c "$2" "$3"\n'
            'else\n'
            f'    printf "%s\\n" {visible_gpus}\n'
            'fi\n'
        ),
    )
    _write_executable(
        bin_dir / 'nnUNetv2_train',
        (
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$@" > "$TRAIN_ARGS_PATH"\n'
            'printf "%s\\n" "${SPAD_UNIVERSAL_COMPILE_CACHE_ARCHIVE:-}" '
            '> "$CACHE_ARCHIVE_PATH"\n'
            'printf "%s\\n" "${SPAD_UNIVERSAL_TILE_STEP_SIZE:-}" '
            '> "$TILE_STEP_SIZE_PATH"\n'
            'printf "%s\\n" "${SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE:-}" '
            '>> "$TILE_STEP_SIZE_PATH"\n'
            'printf "%s\\n" "${SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS:-}" '
            '>> "$TILE_STEP_SIZE_PATH"\n'
            'printf "%s\\n" "${nnUNet_compile:-}" '
            '> "$COMPILE_MODE_PATH"\n'
        ),
    )
    _write_executable(
        bin_dir / 'uname',
        '#!/usr/bin/env bash\nprintf "%s\\n" x86_64\n',
    )
    env = {
        **os.environ,
        'PATH': f'{bin_dir}:{os.environ["PATH"]}',
        'TRAIN_ARGS_PATH': str(train_args),
        'CACHE_ARCHIVE_PATH': str(cache_archive),
        'TILE_STEP_SIZE_PATH': str(tile_step_size),
        'COMPILE_MODE_PATH': str(compile_mode),
    }
    return (
        launcher,
        env,
        train_args,
        cache_archive,
        tile_step_size,
        compile_mode,
    )


def test_launcher_derives_local_ranks_ignoring_cluster_context(tmp_path):
    launcher, env, train_args_path, _, _, _ = _launcher_project(
        tmp_path,
        visible_gpus=4,
    )
    env.update({'WORLD_SIZE': '2', 'RANK': '1'})

    result = subprocess.run(
        [launcher, DATASET, PLAN, '--trainer', 'SPADUniversalTrainer'],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert 'launching SPAD U-Net: local_ranks=4' in result.stdout
    train_args = train_args_path.read_text().splitlines()
    assert train_args[train_args.index('-num_gpus') + 1] == '4'


def test_launcher_treats_explicit_gpu_count_as_local_ranks(tmp_path):
    launcher, env, train_args_path, _, _, _ = _launcher_project(
        tmp_path,
        visible_gpus=4,
    )
    env.update({'WORLD_SIZE': '3', 'RANK': '2'})

    result = subprocess.run(
        [
            launcher,
            DATASET,
            PLAN,
            '--trainer',
            'SPADUniversalTrainer',
            '--gpus',
            '2',
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert 'launching SPAD U-Net: local_ranks=2' in result.stdout
    train_args = train_args_path.read_text().splitlines()
    assert train_args[train_args.index('-num_gpus') + 1] == '2'


def test_launcher_separates_compile_cache_by_machine_architecture(tmp_path):
    launcher, env, _, cache_archive_path, _, _ = _launcher_project(
        tmp_path,
        visible_gpus=8,
    )
    record_dir = tmp_path / 'records'

    subprocess.run(
        [
            launcher,
            DATASET,
            PLAN,
            '--trainer',
            'SPADUniversalTrainer',
            '--record-dir',
            record_dir,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert cache_archive_path.read_text().strip() == str(
        record_dir / 'spad-torchinductor-runtime-cache-x86.tar.zst'
    )


def test_launcher_names_and_exports_nondefault_tile_step(tmp_path):
    launcher, env, train_args_path, _, tile_step_path, _ = _launcher_project(
        tmp_path,
        visible_gpus=4,
    )

    subprocess.run(
        [
            launcher,
            DATASET,
            PLAN,
            '--trainer',
            'SPADUniversalTrainer',
            '--tile-step-size',
            '0.40',
            '--sliding-window-batch-size',
            '8',
            '--case-evaluation-workers',
            '4',
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert train_args_path.is_file()
    assert tile_step_path.read_text().splitlines() == ['0.4', '8', '4']


def test_launcher_uses_tile_step_specific_completion_marker(tmp_path):
    launcher, env, train_args_path, _, _, _ = _launcher_project(
        tmp_path,
        visible_gpus=4,
    )
    summary_path = (
        tmp_path
        / 'nnUNet_data'
        / 'results'
        / DATASET
        / f'SPADUniversalTrainer__{PLAN}__3d_fullres'
        / 'fold_0'
        / 'validation_cross_step0.4_batch8'
        / 'summary.json'
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text('{}\n')

    result = subprocess.run(
        [
            launcher,
            DATASET,
            PLAN,
            '--trainer',
            'SPADUniversalTrainer',
            '--tile-step-size',
            '0.4',
            '--sliding-window-batch-size',
            '8',
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert f'already complete: {summary_path}' in result.stdout
    assert not train_args_path.exists()


def test_launcher_uses_validation_only_without_compile_for_final_checkpoint(
    tmp_path,
):
    (
        launcher,
        env,
        train_args_path,
        _,
        _,
        compile_mode_path,
    ) = _launcher_project(tmp_path, visible_gpus=4)
    output_folder = (
        tmp_path
        / 'nnUNet_data'
        / 'results'
        / DATASET
        / f'SPADUniversalTrainer__{PLAN}__3d_fullres'
        / 'fold_0'
    )
    output_folder.mkdir(parents=True)
    (output_folder / 'checkpoint_final.pth').touch()

    subprocess.run(
        [launcher, DATASET, PLAN, '--trainer', 'SPADUniversalTrainer'],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '--val' in train_args_path.read_text().splitlines()
    assert '--c' not in train_args_path.read_text().splitlines()
    assert compile_mode_path.read_text().strip() == 'false'


def test_launcher_keeps_compile_for_incomplete_checkpoint(tmp_path):
    (
        launcher,
        env,
        train_args_path,
        _,
        _,
        compile_mode_path,
    ) = _launcher_project(tmp_path, visible_gpus=4)
    output_folder = (
        tmp_path
        / 'nnUNet_data'
        / 'results'
        / DATASET
        / f'SPADUniversalTrainer__{PLAN}__3d_fullres'
        / 'fold_0'
    )
    output_folder.mkdir(parents=True)
    (output_folder / 'checkpoint_latest.pth').touch()

    subprocess.run(
        [launcher, DATASET, PLAN, '--trainer', 'SPADUniversalTrainer'],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '--c' in train_args_path.read_text().splitlines()
    assert '--val' not in train_args_path.read_text().splitlines()
    assert compile_mode_path.read_text().strip() == 'true'


def test_spad_compile_paths_are_rotated_across_local_ranks():
    for local_world_size in (8, 4):
        assignments = [
            SPADUniversalTrainer._node_local_compile_warmup_indices(
                16,
                local_rank,
                local_world_size,
            )
            for local_rank in range(local_world_size)
        ]

        assert all(len(indices) == 16 for indices in assignments)
        assert all(sorted(indices) == list(range(16)) for indices in assignments)
        assert [indices[0] for indices in assignments] == list(
            range(local_world_size)
        )
        assert all(
            indices[round_index] == (round_index + local_rank) % 16
            for local_rank, indices in enumerate(assignments)
            for round_index in range(16)
        )


def test_spad_compile_warmup_uses_worker_local_topology(monkeypatch):
    trainer = object.__new__(SPADUniversalTrainer)
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '4')
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)

    assert trainer._compile_warmup_local_context() == (2, 4)


class _GlobalRankCheckpointBase:
    local_rank = 0
    global_rank = 4
    disable_checkpointing = False

    def __init__(self, output_folder: Path):
        self.output_folder = str(output_folder)
        self.current_epoch = 4
        self.base_save_calls = 0

    def save_checkpoint(self, filename: str) -> None:
        self.base_save_calls += 1
        if self.global_rank == 0:
            torch.save(
                {
                    '_best_ema': 0.5,
                    'logging': {'ema_fg_dice': [0.5]},
                },
                filename,
            )


def test_checkpoint_writer_is_selected_by_global_rank(tmp_path, monkeypatch):
    class Trainer(RetainPeriodicCheckpointsMixin, _GlobalRankCheckpointBase):
        pass

    trainer = Trainer(tmp_path)
    checkpoint_path = tmp_path / 'checkpoint_latest.pth'
    monkeypatch.setattr(checkpointing_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(checkpointing_module.dist, 'get_rank', lambda: 4)

    trainer.save_checkpoint(str(checkpoint_path))
    trainer._join_checkpoint_writer()

    assert trainer.local_rank == 0
    assert trainer.global_rank == 4
    assert trainer.base_save_calls == 1
    assert not checkpoint_path.exists()


def test_checkpoint_global_rank_falls_back_to_process_group(monkeypatch):
    monkeypatch.setattr(checkpointing_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(checkpointing_module.dist, 'get_rank', lambda: 4)

    assert checkpointing_module._global_rank() == 4


class _CompileCacheBase:
    global_rank = 4
    local_rank = 0
    was_initialized = False

    def __init__(self, output_folder: Path):
        self.output_folder_base = str(output_folder)
        self.messages = []

    def _do_i_compile(self) -> bool:
        return True

    def initialize(self) -> None:
        self.was_initialized = True

    def print_to_log_file(self, message: str) -> None:
        self.messages.append(message)


def test_nnunet_compile_cache_uses_node_local_extractor(
    tmp_path,
    monkeypatch,
):
    calls = []

    def extract(archive, cache_dir, **kwargs):
        calls.append((archive, cache_dir, kwargs))
        return Path(cache_dir), True

    class Trainer(CompileCacheMixin, _CompileCacheBase):
        pass

    archive = tmp_path / 'cache.tar.zst'
    cache_dir = tmp_path / 'cache'
    trainer = Trainer(tmp_path)
    trainer._compile_cache_paths = lambda: (archive, cache_dir)
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'get_rank', lambda: 4)
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(
        nnunet_compile_cache_module.dist,
        'broadcast_object_list',
        lambda status, src: status.__setitem__(0, False),
    )
    monkeypatch.setattr(
        nnunet_compile_cache_module,
        'extract_compile_cache',
        extract,
    )

    trainer.initialize()

    assert calls == [(
        archive,
        cache_dir,
        {'rank': 0, 'archive_exists': False},
    )]
    assert trainer._compile_cache_needs_archive


def test_nnunet_compile_cache_ranks_support_installed_nnunet(monkeypatch):
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'get_rank', lambda: 4)
    monkeypatch.setenv('LOCAL_RANK', '0')

    assert nnunet_compile_cache_module._global_rank() == 4
    assert nnunet_compile_cache_module._local_rank() == 0


def test_nnunet_compile_cache_uses_global_archive_writer(
    tmp_path,
    monkeypatch,
):
    calls = []

    class Trainer(CompileCacheMixin, _CompileCacheBase):
        pass

    trainer = Trainer(tmp_path)
    trainer._compile_cache_archive = tmp_path / 'cache.tar.zst'
    trainer._compile_cache_dir = tmp_path / 'cache'
    trainer._compile_cache_needs_archive = True
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'get_rank', lambda: 4)
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'barrier', lambda: None)
    monkeypatch.setattr(
        nnunet_compile_cache_module.dist,
        'broadcast_object_list',
        lambda status, src: status.__setitem__(0, True),
    )
    monkeypatch.setattr(
        nnunet_compile_cache_module,
        'distributed_archive_compile_cache',
        lambda archive, cache_dir, **kwargs: calls.append(
            (archive, cache_dir, kwargs)
        ),
    )

    trainer.archive_compile_cache_now()

    assert calls == [(
        trainer._compile_cache_archive,
        trainer._compile_cache_dir,
        {'rank': 4, 'best_effort': True},
    )]
    assert not trainer._compile_cache_needs_archive


def test_nnunet_compile_cache_force_refreshes_existing_archive(
    tmp_path,
    monkeypatch,
):
    calls = []

    class Trainer(CompileCacheMixin, _CompileCacheBase):
        pass

    trainer = Trainer(tmp_path)
    trainer._compile_cache_archive = tmp_path / 'cache.tar.zst'
    trainer._compile_cache_dir = tmp_path / 'cache'
    trainer._compile_cache_needs_archive = False
    monkeypatch.setattr(nnunet_compile_cache_module.dist, 'is_initialized', lambda: False)
    monkeypatch.setattr(
        nnunet_compile_cache_module,
        'distributed_archive_compile_cache',
        lambda archive, cache_dir, **kwargs: calls.append(
            (archive, cache_dir, kwargs)
        ),
    )
    monkeypatch.setattr(Path, 'is_file', lambda self: True)

    trainer.archive_compile_cache_now(force=True)

    assert calls == [(
        trainer._compile_cache_archive,
        trainer._compile_cache_dir,
        {'rank': 0, 'best_effort': True},
    )]
