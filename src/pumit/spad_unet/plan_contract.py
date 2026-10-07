"""Plan-level contract for SPAD Universal experiments.

Declares the plan identifiers, dataset-objective/sampling axes, and the validation and weighting
rules that bind a plan to its replay stream and loss semantics. Shared by plan preparation
scripts and both universal trainers; importing it does not pull in the trainer stack.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from pumit.spad_unet.loss import (
    REGION_BALANCED_LOSS_NORMALIZATION,
    SAMPLE_LEVEL_LOSS_NORMALIZATIONS,
    SUPPORTED_LOSS_NORMALIZATIONS,
)
from pumit.spad_unet.replay import (
    COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    SQRT_DATASET_REPLAY_RULE,
)
from pumit.spad_unet.source_plans import (
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE0P9_FOV192_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_FOV224_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_P224_MAXB7_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_P224_MAXB7_PLANS_IDENTIFIER,
)

SPAD_UNIVERSAL_CONFIGURATION_NAME = '3d_fullres'
SPAD_UNIVERSAL_LEGACY_SOURCE_PLANS_IDENTIFIER = 'nnUNetResEncUNetLPlans'
SPAD_UNIVERSAL_INPLANE_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans7StageInplane'
)
SPAD_UNIVERSAL_SEVEN_STAGE_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans7StageFOV192'
)
SPAD_UNIVERSAL_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StagePlannedZ1x1FOV192'
)
SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIER = (
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER
)
SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_PLANNED_XY_SOURCE_PLANS_IDENTIFIER = (
    SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER
)
SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIERS = (
    SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE0P9_FOV192_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_FOV224_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_P224_MAXB7_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_PLANNED_XY_SOURCE_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_P224_MAXB7_PLANS_IDENTIFIER,
)
SPAD_UNIVERSAL_LEGACY_NATIVE_Z_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageNativeZ1x1FOV192'
)
# Fixed 1 mm isotropic grid at patch 224; every dataset trains at DA 0.
SPAD_UNIVERSAL_FIXED_ISO_P224_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans1x1x1P224'
)
SPAD_UNIVERSAL_SUPPORTED_SOURCE_PLANS_IDENTIFIERS = (
    SPAD_UNIVERSAL_LEGACY_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_INPLANE_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SEVEN_STAGE_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_LEGACY_NATIVE_Z_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_FIXED_ISO_P224_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SOURCE_PLANS_IDENTIFIER,
    *SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIERS,
)
SPAD_UNIVERSAL_PLANS_IDENTIFIER = 'SPADPlannedZSumGB4Plans'
SPAD_UNIVERSAL_LKR_PLANS_IDENTIFIER = 'SPADPlannedZLKRGB4Plans'
SPAD_UNIVERSAL_GLOBAL_BATCH_SIZE = 4
# Outer dataset objective: the target weighting over dataset-local losses.
# It is independent from dataset sampling (which records are presented) and
# from the dataset output contract (the dataset-local prediction semantics).
SPAD_UNIVERSAL_UNIFORM_REGION_OBJECTIVE = 'uniform_region'
SPAD_UNIVERSAL_SAMPLING_MATCHED_OBJECTIVE = 'sampling_matched'
SPAD_UNIVERSAL_UNIFORM_DATASET_OBJECTIVE = 'uniform_dataset'
SPAD_UNIVERSAL_DATASET_OBJECTIVES = (
    SPAD_UNIVERSAL_UNIFORM_REGION_OBJECTIVE,
    SPAD_UNIVERSAL_SAMPLING_MATCHED_OBJECTIVE,
    SPAD_UNIVERSAL_UNIFORM_DATASET_OBJECTIVE,
)
SPAD_UNIVERSAL_SQRT_DATASET_SAMPLING = 'sqrt_training_cases'
SPAD_UNIVERSAL_DATASET_SAMPLINGS = (SPAD_UNIVERSAL_SQRT_DATASET_SAMPLING,)
SPAD_UNIVERSAL_INFERENCE_MODES = (
    'floor',
    'ceil',
    'cross',
    'endpoints',
)
SPAD_UNIVERSAL_DA_DISCRETIZATIONS = ('stochastic', 'floor')


def _num_epochs_for_updates(
    num_updates: int,
    num_iterations_per_epoch: int,
) -> int:
    num_epochs, remainder = divmod(num_updates, num_iterations_per_epoch)
    if remainder:
        raise ValueError(
            f'experiment requests {num_updates} updates, which is not divisible '
            f'by {num_iterations_per_epoch} iterations per epoch'
        )
    return num_epochs


def validate_spad_universal_replay_metadata(
    metadata: Mapping[str, object],
    world_size: int,
) -> None:
    """Validate the replay contract a SPAD Universal plan declares."""
    global_batch_size = metadata.get('global_batch_size')
    if (
        not isinstance(global_batch_size, int)
        or isinstance(global_batch_size, bool)
        or global_batch_size < 1
    ):
        raise ValueError(
            f'SPAD Universal requires a positive integer global batch size, '
            f'got {global_batch_size}'
        )
    if 'replay_stream' not in metadata:
        if metadata.get('samples_per_dataset') != 1:
            raise ValueError(
                'SPAD Universal complementary replay requires samples_per_dataset=1'
            )
        if metadata.get('sampling_rule') != COMPLEMENTARY_FOREGROUND_REPLAY_RULE:
            raise ValueError(
                'SPAD Universal requires the complementary-foreground replay rule'
            )
    if metadata.get('loss_normalization') not in SUPPORTED_LOSS_NORMALIZATIONS:
        raise ValueError(
            'SPAD Universal requires an explicit supported loss normalization'
        )
    if world_size < 1 or global_batch_size % world_size:
        raise ValueError(
            f'global batch size {global_batch_size} is not divisible by '
            f'world size {world_size}'
        )


def resolve_spad_universal_dataset_objective(
    dataset_objective: object,
    loss_normalization: str,
) -> str:
    """Resolve the plan's outer dataset objective.

    Plans without the field keep their legacy semantics: the region-balanced
    normalization realizes uniform_region, and the sample-level normalizations
    average the sampled losses unweighted, which is sampling_matched.
    uniform_region is valid with every normalization: region-balanced realizes
    it implicitly, sample-level normalizations realize it through region-mass
    dataset weights.
    """
    if dataset_objective is None:
        return (
            SPAD_UNIVERSAL_UNIFORM_REGION_OBJECTIVE
            if loss_normalization == REGION_BALANCED_LOSS_NORMALIZATION
            else SPAD_UNIVERSAL_SAMPLING_MATCHED_OBJECTIVE
        )
    if dataset_objective not in SPAD_UNIVERSAL_DATASET_OBJECTIVES:
        raise ValueError(
            f'unsupported dataset objective {dataset_objective!r}; expected '
            f'one of {SPAD_UNIVERSAL_DATASET_OBJECTIVES}'
        )
    if dataset_objective == SPAD_UNIVERSAL_UNIFORM_REGION_OBJECTIVE:
        if loss_normalization not in SUPPORTED_LOSS_NORMALIZATIONS:
            raise ValueError(
                f'unsupported loss normalization {loss_normalization!r}; '
                f'expected one of {SUPPORTED_LOSS_NORMALIZATIONS}'
            )
    elif loss_normalization not in SAMPLE_LEVEL_LOSS_NORMALIZATIONS:
        raise ValueError(
            f'{dataset_objective} weights dataset-local sample losses and '
            f'requires a sample-level loss normalization, got '
            f'{loss_normalization!r}'
        )
    return dataset_objective


def resolve_sqrt_replay_dataset_probabilities(
    dataset_ids: tuple[str, ...],
    sampling_rule: str,
    dataset_sampling_probabilities: Mapping[str, object] | None,
) -> dict[str, float]:
    """Validate and return the sqrt replay's per-dataset sampling probabilities."""
    if sampling_rule != SQRT_DATASET_REPLAY_RULE:
        raise ValueError(
            f'{SPAD_UNIVERSAL_SQRT_DATASET_SAMPLING} requires a replay with '
            f'sampling_rule={SQRT_DATASET_REPLAY_RULE!r}, got {sampling_rule!r}'
        )
    if not isinstance(dataset_sampling_probabilities, Mapping) or set(
        dataset_sampling_probabilities
    ) != set(dataset_ids):
        raise ValueError(
            'replay dataset_sampling_probabilities must define every dataset '
            'exactly once'
        )
    probabilities = {}
    for dataset_id in dataset_ids:
        value = dataset_sampling_probabilities[dataset_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f'replay sampling probability for dataset {dataset_id} must '
                f'be positive and finite, got {value!r}'
            )
        probabilities[dataset_id] = float(value)
    if not math.isclose(sum(probabilities.values()), 1.0):
        raise ValueError(
            'replay dataset sampling probabilities must sum to one'
        )
    return probabilities


def validate_spad_universal_dataset_sampling(
    dataset_sampling: object,
    dataset_ids: tuple[str, ...],
    sampling_rule: str,
    dataset_sampling_probabilities: Mapping[str, object] | None,
) -> None:
    """Validate the plan's declared dataset-sampling axis against the replay."""
    if dataset_sampling is None:
        return
    if dataset_sampling not in SPAD_UNIVERSAL_DATASET_SAMPLINGS:
        raise ValueError(
            f'unsupported dataset sampling {dataset_sampling!r}; expected '
            f'one of {SPAD_UNIVERSAL_DATASET_SAMPLINGS}'
        )
    resolve_sqrt_replay_dataset_probabilities(
        dataset_ids,
        sampling_rule,
        dataset_sampling_probabilities,
    )


def spad_universal_dataset_loss_weights(
    dataset_objective: str,
    dataset_ids: tuple[str, ...],
    sampling_rule: str,
    dataset_sampling_probabilities: Mapping[str, object] | None,
    *,
    loss_normalization: str,
    dataset_region_counts: Mapping[str, int] | None,
) -> dict[str, float]:
    """Per-dataset objective weights multiplying dataset-local sample losses.

    uniform_dataset importance-corrects the sampled stream toward the uniform
    dataset mean: w_d = (1/D) / p_d, so E_{d~p}[w_d L_d] = (1/D) sum_d E[L_d].
    uniform_region under a sample-level normalization restores the legacy
    region mass instead: w_d = (C_d / C) / p_d, giving each of the C canonical
    regions expected objective mass 1/C. Both keep sum_d p_d w_d = 1, the unit
    expected sample weight. All other combinations leave the sampled losses
    unweighted; that includes uniform_region under the legacy region-balanced
    normalization, which realizes the same objective implicitly.
    """
    if dataset_objective not in SPAD_UNIVERSAL_DATASET_OBJECTIVES:
        raise ValueError(
            f'unsupported dataset objective {dataset_objective!r}; expected '
            f'one of {SPAD_UNIVERSAL_DATASET_OBJECTIVES}'
        )
    if loss_normalization not in SUPPORTED_LOSS_NORMALIZATIONS:
        raise ValueError(
            f'unsupported loss normalization {loss_normalization!r}; '
            f'expected one of {SUPPORTED_LOSS_NORMALIZATIONS}'
        )
    if dataset_objective == SPAD_UNIVERSAL_UNIFORM_DATASET_OBJECTIVE:
        probabilities = resolve_sqrt_replay_dataset_probabilities(
            dataset_ids,
            sampling_rule,
            dataset_sampling_probabilities,
        )
        dataset_weight = 1.0 / len(dataset_ids)
        return {
            dataset_id: dataset_weight / probability
            for dataset_id, probability in probabilities.items()
        }
    if (
        dataset_objective != SPAD_UNIVERSAL_UNIFORM_REGION_OBJECTIVE
        or loss_normalization not in SAMPLE_LEVEL_LOSS_NORMALIZATIONS
    ):
        return {}
    if sampling_rule == SQRT_DATASET_REPLAY_RULE:
        probabilities = resolve_sqrt_replay_dataset_probabilities(
            dataset_ids,
            sampling_rule,
            dataset_sampling_probabilities,
        )
    elif sampling_rule == COMPLEMENTARY_FOREGROUND_REPLAY_RULE:
        # The structured replay emits every dataset exactly once per group.
        probabilities = {
            dataset_id: 1.0 / len(dataset_ids)
            for dataset_id in dataset_ids
        }
    else:
        raise ValueError(
            f'sample-level {dataset_objective} requires dataset sampling '
            f'probabilities, which sampling_rule {sampling_rule!r} does not '
            f'define'
        )
    if not isinstance(dataset_region_counts, Mapping) or set(
        dataset_region_counts
    ) != set(dataset_ids):
        raise ValueError(
            f'sample-level {dataset_objective} requires dataset region counts '
            f'defining every dataset exactly once'
        )
    for dataset_id in dataset_ids:
        count = dataset_region_counts[dataset_id]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f'dataset {dataset_id} region count must be a positive '
                f'integer, got {count!r}'
            )
    total_regions = sum(
        dataset_region_counts[dataset_id] for dataset_id in dataset_ids
    )
    return {
        dataset_id: (
            dataset_region_counts[dataset_id] / total_regions
        ) / probabilities[dataset_id]
        for dataset_id in dataset_ids
    }


def validate_spad_universal_inference_mode(
    inference_mode: str,
    da_discretization: str,
) -> None:
    """Validate the full-volume inference mode against the plan's training route policy."""
    if inference_mode not in SPAD_UNIVERSAL_INFERENCE_MODES:
        raise ValueError(
            f'unsupported SPAD_UNIVERSAL_INFERENCE_MODE {inference_mode!r}; '
            f'expected one of {SPAD_UNIVERSAL_INFERENCE_MODES}'
        )
    if da_discretization == 'floor' and inference_mode not in (
        'floor',
        'endpoints',
    ):
        raise ValueError(
            'deterministic floor plans require floor or endpoint full-volume inference'
        )
