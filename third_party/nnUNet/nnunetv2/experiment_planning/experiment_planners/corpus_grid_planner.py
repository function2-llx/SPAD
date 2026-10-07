from typing import List, Tuple, Union

import numpy as np

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    nnUNetPlannerResEncL,
)


class nnUNetPlannerResEncLCorpusGrid(nnUNetPlannerResEncL):
    """ResEnc L planner for the frozen SPAD U-Net Corpus-grid control."""

    target_spacing = (1.5, 1.0, 1.0)
    patch_size = (128, 192, 192)
    batch_size = 8
    gpu_memory_target_in_gb_default = 24.0
    # Subclasses freeze a different grid by overriding these four attributes only.
    plans_name_default = 'nnUNetResEncUNetLPlans1p5x1x1FOV192'

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = None,
        preprocessor_name: str = 'DefaultPreprocessor',
        plans_name: str = None,
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        gpu_memory_target_in_gb = (
            self.gpu_memory_target_in_gb_default
            if gpu_memory_target_in_gb is None
            else gpu_memory_target_in_gb
        )
        if gpu_memory_target_in_gb != self.gpu_memory_target_in_gb_default:
            raise ValueError(
                f'The Corpus-grid plan is frozen at the ResEnc L '
                f'{self.gpu_memory_target_in_gb_default:g} GB target, got '
                f'{gpu_memory_target_in_gb}'
            )
        if overwrite_target_spacing is not None and tuple(overwrite_target_spacing) != self.target_spacing:
            raise ValueError(
                f'The Corpus-grid target spacing is frozen at {self.target_spacing}, '
                f'got {tuple(overwrite_target_spacing)}'
            )
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name or self.plans_name_default,
            self.target_spacing,
            suppress_transpose,
        )
        self.lowres_creation_threshold = 0

    def generate_data_identifier(self, configuration_name: str) -> str:
        return f'{self.plans_identifier}_{configuration_name}'

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        if len(spacing) != 3 or not np.array_equal(np.asarray(spacing), np.asarray(self.target_spacing)):
            return super().get_plans_for_configuration(
                spacing,
                median_shape,
                data_identifier,
                approximate_n_voxels_dataset,
                _cache,
            )

        dataset_median_shape = np.asarray(median_shape)
        plan = super().get_plans_for_configuration(
            spacing,
            self.patch_size,
            data_identifier,
            approximate_n_voxels_dataset,
            _cache,
        )
        if tuple(plan['patch_size']) != self.patch_size:
            raise RuntimeError(
                f'The frozen {self.patch_size} patch exceeds the ResEnc L 24 GB planning target: '
                f'planner returned {tuple(plan["patch_size"])}'
            )
        plan['median_image_size_in_voxels'] = dataset_median_shape
        plan['batch_size'] = self.batch_size
        return plan

    def save_plans(self, plans):
        fullres = plans['configurations'].get('3d_fullres')
        if fullres is None:
            raise RuntimeError('The Corpus-grid planner requires a 3d_fullres configuration')
        if fullres['batch_dice']:
            raise RuntimeError('The Corpus-grid full-resolution configuration must disable batch Dice')
        super().save_plans(plans)


class nnUNetPlannerResEncLCorpusGridIso1(nnUNetPlannerResEncLCorpusGrid):
    """Corpus-grid control on an isotropic 1 mm grid, preserving the 192 mm physical FOV."""

    target_spacing = (1.0, 1.0, 1.0)
    patch_size = (192, 192, 192)
    batch_size = 4
    plans_name_default = 'nnUNetResEncUNetLPlans1x1x1FOV192'


class nnUNetPlannerResEncLCorpusGridIso1P224(
    nnUNetPlannerResEncLCorpusGrid
):
    """Corpus-grid control on a fixed isotropic 1 mm, 224-cube grid."""

    target_spacing = (1.0, 1.0, 1.0)
    patch_size = (224, 224, 224)
    batch_size = 2
    gpu_memory_target_in_gb_default = 40.0
    plans_name_default = 'nnUNetResEncUNetLPlans1x1x1P224'


class nnUNetPlannerResEncLCorpusGridIso0p9P224(
    nnUNetPlannerResEncLCorpusGrid
):
    """Corpus-grid control on a fixed isotropic 0.9 mm, 224-cube grid."""

    target_spacing = (0.9, 0.9, 0.9)
    patch_size = (224, 224, 224)
    batch_size = 2
    gpu_memory_target_in_gb_default = 40.0
    plans_name_default = 'nnUNetResEncUNetLPlans0p9x0p9x0p9P224'
