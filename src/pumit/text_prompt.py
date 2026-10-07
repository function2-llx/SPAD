"""Text prompt construction for text-conditioned segmentation."""

import hashlib
from pathlib import Path

import numpy as np
import orjson


type ClassCaptions = dict[str, dict[str, list[str]]]


def load_class_captions(captions_dir: Path | str) -> ClassCaptions:
    """Load and validate one caption document per label source.

    Args:
        captions_dir: Directory containing ``<source>.json`` documents.

    Returns:
        Caption mappings ordered by source and raw class name.
    """
    captions_dir = Path(captions_dir)
    if not captions_dir.is_dir():
        raise NotADirectoryError(captions_dir)
    paths = sorted(captions_dir.glob('*.json'), key=lambda path: path.name)
    if not paths:
        raise ValueError(f'no caption definitions found in {captions_dir}')

    captions: ClassCaptions = {}
    for path in paths:
        document = orjson.loads(path.read_bytes())
        if set(document) != {'source', 'classes'}:
            raise ValueError(f'{path}: expected exactly "source" and "classes"')

        source = document['source']
        classes = document['classes']
        if not isinstance(source, str) or not source:
            raise ValueError(f'{path}: source must be a non-empty string')
        if path.name != f'{source}.json':
            raise ValueError(f'{path}: filename must match source {source!r}')
        if source in captions:
            raise ValueError(f'{path}: duplicate source {source!r}')
        if not isinstance(classes, dict) or not classes:
            raise ValueError(f'{path}: classes must be a non-empty object')

        validated_classes: dict[str, list[str]] = {}
        for class_name in sorted(classes):
            descriptions = classes[class_name]
            if not isinstance(class_name, str) or not class_name:
                raise ValueError(f'{path}: class names must be non-empty strings')
            if not isinstance(descriptions, list) or not descriptions:
                raise ValueError(f'{path}: {class_name!r} must have at least one description')
            if any(not isinstance(description, str) or not description.strip() for description in descriptions):
                raise ValueError(f'{path}: {class_name!r} descriptions must be non-empty strings')
            if len(descriptions) != len(set(descriptions)):
                raise ValueError(f'{path}: {class_name!r} contains duplicate descriptions')
            validated_classes[class_name] = descriptions
        captions[source] = validated_classes
    return captions


def class_captions_sha256(captions: ClassCaptions) -> str:
    """Hash the validated logical taxonomy independently of file formatting."""
    payload = orjson.dumps(captions, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


MODALITY_CONTEXTS: dict[str, str] = {
    'CT': 'a CT scan',
    'MRI': 'an MRI scan',
    'PET': 'a PET scan',
    'US': 'an ultrasound image',
    'X-ray': 'an X-ray image',
    'CBCT': 'a cone-beam CT scan',
    'fundus': 'a color fundus photograph',
    'dermoscopy': 'a dermoscopy image',
    'endoscopy': 'an endoscopy image',
    'histopathology': 'a histopathology image',
}

MODALITY_MAP: dict[str, str] = {
    'CT': 'CT',
    'MRI': 'MRI',
    'MRI/T1': 'MRI',
    'MRI/T2': 'MRI',
    'MRI/T1c': 'MRI',
    'MRI/T2-FLAIR': 'MRI',
    'MRI/DWI': 'MRI',
    'MRI/ADC': 'MRI',
    'MRI/FLAIR': 'MRI',
    'MRI/PD': 'MRI',
    'MRI/T1-dual/in': 'MRI',
    'MRI/T1-dual/out': 'MRI',
    'MRI/T2-SPIR': 'MRI',
    'MRI/T1-IR': 'MRI',
    'T2': 'MRI',
    'ADC': 'MRI',
    'PET': 'PET',
    'US': 'US',
    'XRA': 'X-ray',
    'X-ray': 'X-ray',
    'CBCT': 'CBCT',
    'fundus': 'fundus',
    'dermoscopy': 'dermoscopy',
    'endoscopy': 'endoscopy',
    'histopathology': 'histopathology',
}


def normalize_modality(modality: str) -> str:
    """Map a data modality string to its coarse prompt modality."""
    return MODALITY_MAP[modality]


def build_segmentation_prompt(description: str, modality: str) -> str:
    """Render the exact text supplied to the segmentation text encoder."""
    context = MODALITY_CONTEXTS[normalize_modality(modality)]
    return f'segmentation mask of {description} in {context}'


def build_segmentation_prompts(captions: ClassCaptions) -> list[str]:
    """Build the sorted, distinct prompts required by a caption taxonomy."""
    prompts: set[str] = set()
    for names in captions.values():
        for descriptions in names.values():
            for description in descriptions:
                for modality in MODALITY_CONTEXTS:
                    prompts.add(build_segmentation_prompt(description, modality))
    return sorted(prompts)


class SegmentationPromptResolver:
    """Resolve raw label identities to their exact segmentation prompts."""

    def __init__(self, captions_dir: Path | str):
        self.captions = load_class_captions(captions_dir)

    def get(
        self,
        source: str,
        class_name: str,
        modality: str,
        *,
        rng: np.random.Generator | None = None,
    ) -> str:
        """Resolve the canonical description, or sample a uniform text variant."""
        descriptions = self.captions[source][class_name]
        index = 0 if rng is None else int(rng.integers(len(descriptions)))
        description = descriptions[index]
        return build_segmentation_prompt(description, modality)

    def get_batch(
        self,
        keys: list[tuple[str, str, str]],
        *,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        return [
            self.get(source, class_name, modality, rng=rng)
            for source, class_name, modality in keys
        ]
