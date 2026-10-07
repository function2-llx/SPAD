"""``nnUNetv2_train`` replacement whose fold argument may name a split.

Stock ``run_training`` casts every non-``'all'`` fold to ``int`` before it reaches the trainer, so named
label-fraction splits (resolved by ``_NamedFoldSplitMixin.do_split``) cannot pass through the official CLI.
This entry keeps the stock single-GPU and DDP dispatch by importing it and only exposes the argument
surface ``launch_one.sh`` uses.
"""

from __future__ import annotations

import argparse
import os

import torch
from torch import multiprocessing as mp
from torch.backends import cudnn

from nnunetv2.run.run_training import (
    find_free_network_port,
    get_trainer_from_args,
    maybe_load_checkpoint,
    run_ddp,
)


def parse_fold(fold: str) -> int | str:
    """Numeric folds keep stock integer semantics; anything else is a named split."""
    try:
        return int(fold)
    except ValueError:
        return fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id')
    parser.add_argument('configuration')
    parser.add_argument('fold', type=parse_fold)
    parser.add_argument('-tr', default='nnUNetTrainer')
    parser.add_argument('-p', default='nnUNetPlans')
    parser.add_argument('-pretrained_weights', default=None)
    parser.add_argument('-num_gpus', type=int, default=1)
    parser.add_argument('--c', action='store_true')
    args = parser.parse_args()

    # Stock run_training_entry's GPU-path thread configuration.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if args.num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = str(find_free_network_port())
        mp.spawn(
            run_ddp,
            args=(
                args.dataset_name_or_id,
                args.configuration,
                args.fold,
                args.tr,
                args.p,
                False,  # disable_checkpointing
                args.c,
                False,  # only_run_validation
                args.pretrained_weights,
                False,  # export_validation_probabilities
                False,  # val_with_best
                args.num_gpus,
            ),
            nprocs=args.num_gpus,
            join=True,
        )
        return

    trainer = get_trainer_from_args(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        args.tr,
        args.p,
        args.c,
    )
    maybe_load_checkpoint(trainer, args.c, False, args.pretrained_weights)
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True
    trainer.run_training()
    trainer.perform_actual_validation(False)


if __name__ == '__main__':
    main()
