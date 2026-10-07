"""Source-level segmentation label contracts."""

from collections.abc import Iterable
from pathlib import Path

from pumit.codec.config import MAX_DA
from pumit.data import build_training_data


type LabelContract = dict[str, dict[str, list[str]]]
type LabelManifest = dict[tuple[str, str], LabelContract]


def labeled_eligible(record: dict) -> bool:
    """Return whether a dataset record can provide segmentation supervision."""
    if record.get('label') != True:  # noqa: E712 — records may carry np.bool_
        return False
    label_classes = record.get('label_classes', {})
    if not label_classes or not all(isinstance(info, dict) for info in label_classes.values()):
        return False
    return any(
        info.get(state)
        for info in label_classes.values()
        for state in ('positive', 'negative')
    )


def normalize_label_contract(record: dict) -> tuple[LabelContract, int]:
    """Validate and normalize one record's complete source-level class contract."""
    contract: LabelContract = {}
    pairs: set[tuple[str, str]] = set()
    context = f'{record["dataset"]}/{record["key"]}'
    for source in sorted(record['label_classes']):
        info = record['label_classes'][source]
        if not isinstance(info, dict):
            raise ValueError(f'{context}/{source}: label contract must be a mapping')
        states: dict[str, list[str]] = {}
        for state in ('positive', 'negative'):
            names = info.get(state, [])
            if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                raise ValueError(f'{context}/{source}: {state} must be a list of class names')
            states[state] = sorted(names)
        positive = states['positive']
        negative = states['negative']
        if len(positive) != len(set(positive)) or len(negative) != len(set(negative)):
            raise ValueError(f'{context}/{source}: duplicate class')
        overlap = set(positive) & set(negative)
        if overlap:
            raise ValueError(
                f'{context}/{source}: classes are both positive and negative: {sorted(overlap)}'
            )
        contract[source] = states
        for name in [*positive, *negative]:
            pair = source, name
            if pair in pairs:
                raise ValueError(f'{context}: duplicate class pair {pair!r}')
            pairs.add(pair)
    if not pairs:
        raise ValueError(f'{context}: labeled record has no supervised classes')
    return contract, len(pairs)


def build_label_manifest(records: Iterable[dict]) -> LabelManifest:
    """Build the training-time contract lookup keyed by ``(dataset, key)``."""
    manifest: LabelManifest = {}
    for record in records:
        if not labeled_eligible(record):
            continue
        key = record['dataset'], record['key']
        if key in manifest:
            raise ValueError(f'duplicate label manifest key: {key!r}')
        manifest[key], _ = normalize_label_contract(record)
    return manifest


def load_label_manifest(data_root: Path | str) -> LabelManifest:
    """Load current training records and build their source-level label lookup."""
    train_data, _, _ = build_training_data(
        data_root=Path(data_root),
        weight_fn=None,
        depth_tiers=None,
        verbose=False,
        max_da=MAX_DA,
    )
    return build_label_manifest(train_data.reset_index().to_dict('records'))
