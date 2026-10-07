import importlib
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


run_training_module = importlib.import_module('nnunetv2.run.run_training')


@pytest.fixture(autouse=True)
def visible_gpus(monkeypatch):
    monkeypatch.setattr(run_training_module.torch.cuda, 'device_count', lambda: 8)


def test_parent_spawns_local_workers_ignoring_cluster_context():
    environment = {
        'MASTER_ADDR': 'trainer-0',
        'MASTER_PORT': '23456',
        'RANK': '1',
        'WORLD_SIZE': '2',
    }
    with patch.dict(os.environ, environment, clear=True), \
            patch.object(run_training_module.mp, 'spawn') as spawn:
        run_training_module.run_training(
            'Dataset999_Test',
            '3d_fullres',
            0,
            plans_identifier='TestPlans',
            num_gpus=4,
        )

        assert os.environ['MASTER_ADDR'] == 'localhost'
        assert os.environ['MASTER_PORT'] == '23456'

    assert spawn.call_args.args[0] is run_training_module.run_ddp
    assert spawn.call_args.kwargs['nprocs'] == 4
    assert spawn.call_args.kwargs['join'] is True
    assert spawn.call_args.kwargs['args'][-1:] == (4,)


def test_single_gpu_ignores_cluster_context():
    environment = {
        'MASTER_ADDR': 'trainer-0',
        'RANK': '0',
        'WORLD_SIZE': '2',
    }
    with patch.dict(os.environ, environment, clear=True), \
            patch.object(run_training_module.mp, 'spawn') as spawn, \
            patch.object(run_training_module, 'get_trainer_from_args') as get_trainer, \
            patch.object(run_training_module, 'maybe_load_checkpoint'):
        run_training_module.run_training(
            'Dataset999_Test',
            '3d_fullres',
            0,
            plans_identifier='TestPlans',
            num_gpus=1,
        )

        assert 'MASTER_PORT' not in os.environ

    spawn.assert_not_called()
    get_trainer.return_value.run_training.assert_called_once_with()
    get_trainer.return_value.perform_actual_validation.assert_called_once_with(False)


def test_single_node_ddp_keeps_local_rendezvous_behavior():
    with patch.dict(os.environ, {'MASTER_ADDR': 'stale-address'}, clear=True), \
            patch.object(run_training_module, 'find_free_network_port', return_value=23456), \
            patch.object(run_training_module.mp, 'spawn') as spawn:
        run_training_module.run_training(
            'Dataset999_Test',
            '3d_fullres',
            0,
            plans_identifier='TestPlans',
            num_gpus=2,
        )

        assert os.environ['MASTER_ADDR'] == 'localhost'
        assert os.environ['MASTER_PORT'] == '23456'

    assert spawn.call_args.kwargs['args'][-1:] == (2,)


@pytest.mark.parametrize(
    ('environment', 'num_gpus', 'message'),
    [
        ({}, 0, 'num_gpus must be positive'),
        ({}, 9, 'exceeds the number of visible GPUs'),
    ],
)
def test_invalid_gpu_count_fails_before_spawn(
    environment,
    num_gpus,
    message,
):
    with patch.dict(os.environ, environment, clear=True), \
            patch.object(run_training_module.mp, 'spawn') as spawn, \
            pytest.raises(ValueError, match=message):
        run_training_module.run_training(
            'Dataset999_Test',
            '3d_fullres',
            0,
            plans_identifier='TestPlans',
            num_gpus=num_gpus,
        )

    spawn.assert_not_called()


def test_child_uses_local_rank_as_global_rank():
    trainer = MagicMock()
    trainer.output_folder = '/tmp/nnunet-test'
    with patch.dict(os.environ, {}, clear=True), \
            patch.object(run_training_module, 'setup_ddp') as setup_ddp, \
            patch.object(run_training_module.dist, 'is_initialized', return_value=True), \
            patch.object(run_training_module, 'cleanup_ddp') as cleanup_ddp, \
            patch.object(run_training_module.torch.cuda, 'set_device') as set_device, \
            patch.object(run_training_module, 'get_trainer_from_args', return_value=trainer), \
            patch.object(run_training_module, 'maybe_load_checkpoint'):
        run_training_module.run_ddp(
            2,
            'Dataset999_Test',
            '3d_fullres',
            0,
            'TestTrainer',
            'TestPlans',
            False,
            False,
            False,
            None,
            False,
            False,
            4,
        )

        assert os.environ['RANK'] == '2'
        assert os.environ['WORLD_SIZE'] == '4'
        assert os.environ['LOCAL_RANK'] == '2'
        assert os.environ['LOCAL_WORLD_SIZE'] == '4'

    setup_ddp.assert_called_once_with(2, 4)
    set_device.assert_called_once_with(torch.device('cuda', 2))
    trainer.run_training.assert_called_once_with()
    trainer.perform_actual_validation.assert_called_once_with(False)
    cleanup_ddp.assert_called_once_with()


def test_child_cleans_up_process_group_after_trainer_failure():
    trainer = MagicMock()
    trainer.output_folder = '/tmp/nnunet-test'
    trainer.run_training.side_effect = RuntimeError('training failed')
    with patch.dict(os.environ, {}, clear=True), \
            patch.object(run_training_module, 'setup_ddp'), \
            patch.object(run_training_module.dist, 'is_initialized', return_value=True), \
            patch.object(run_training_module, 'cleanup_ddp') as cleanup_ddp, \
            patch.object(run_training_module.torch.cuda, 'set_device'), \
            patch.object(run_training_module, 'get_trainer_from_args', return_value=trainer), \
            patch.object(run_training_module, 'maybe_load_checkpoint'), \
            pytest.raises(RuntimeError, match='training failed'):
        run_training_module.run_ddp(
            2,
            'Dataset999_Test',
            '3d_fullres',
            0,
            'TestTrainer',
            'TestPlans',
            False,
            False,
            False,
            None,
            False,
            False,
            4,
        )

    cleanup_ddp.assert_called_once_with()


def test_nonzero_global_rank_does_not_write_training_log():
    trainer = nnUNetTrainer.__new__(nnUNetTrainer)
    trainer.global_rank = 4
    trainer.local_rank = 0
    trainer.log_file = '/tmp/nnunet-test.log'

    with patch('builtins.open') as open_file:
        trainer.print_to_log_file('worker message', also_print_to_console=False)

    open_file.assert_not_called()


def test_only_global_rank_zero_creates_cross_validation_split():
    trainer = nnUNetTrainer.__new__(nnUNetTrainer)
    trainer.fold = 0
    trainer.is_ddp = True
    trainer.global_rank = 4
    trainer.preprocessed_dataset_folder_base = '/tmp/preprocessed'
    trainer.preprocessed_dataset_folder = '/tmp/preprocessed/configuration'
    trainer.folder_with_segs_from_previous_stage = None
    dataset = MagicMock(identifiers=['case_1', 'case_2'])
    trainer.dataset_class = MagicMock(return_value=dataset)
    trainer.print_to_log_file = MagicMock()
    splits = [{'train': ['case_1'], 'val': ['case_2']}]

    with patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.isfile', return_value=False), \
            patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.generate_crossval_split') as generate, \
            patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.save_json') as save, \
            patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.load_json', return_value=splits) as load, \
            patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.dist.barrier') as barrier:
        train_keys, validation_keys = trainer.do_split()

    generate.assert_not_called()
    save.assert_not_called()
    barrier.assert_called_once_with()
    load.assert_called_once_with('/tmp/preprocessed/splits_final.json')
    assert train_keys == ['case_1']
    assert validation_keys == ['case_2']


def test_ddp_checkpoint_maps_to_current_local_device():
    network = torch.nn.Linear(2, 1)
    checkpoint = {'network_weights': network.state_dict()}
    with patch('nnunetv2.run.load_pretrained_weights.dist.is_initialized', return_value=True), \
            patch('nnunetv2.run.load_pretrained_weights.torch.cuda.current_device', return_value=2), \
            patch('nnunetv2.run.load_pretrained_weights.torch.load', return_value=checkpoint) as load:
        load_pretrained_weights(network, '/tmp/checkpoint.pth')

    assert load.call_args.kwargs['map_location'] == torch.device('cuda', 2)
