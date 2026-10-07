import json
from pathlib import Path

import pytest

from pumit.text_prompt import class_captions_sha256, load_class_captions


CAPTIONS_DIR = Path('src/pumit/ucpt/seg/class_captions')


def test_source_definitions_are_the_complete_runtime_taxonomy():
    captions = load_class_captions(CAPTIONS_DIR)

    assert len(captions) == 99
    assert sum(len(classes) for classes in captions.values()) == 805
    assert len(class_captions_sha256(captions)) == 64


def test_source_filename_must_match_source(tmp_path: Path):
    (tmp_path / 'wrong.json').write_text(json.dumps({
        'source': 'expected',
        'classes': {'class': ['description']},
    }))

    with pytest.raises(ValueError, match='filename must match source'):
        load_class_captions(tmp_path)


def test_duplicate_descriptions_are_rejected(tmp_path: Path):
    (tmp_path / 'source.json').write_text(json.dumps({
        'source': 'source',
        'classes': {'class': ['description', 'description']},
    }))

    with pytest.raises(ValueError, match='duplicate descriptions'):
        load_class_captions(tmp_path)


def test_caption_hash_ignores_json_formatting(tmp_path: Path):
    document = {
        'source': 'source',
        'classes': {'class': ['description']},
    }
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(document))
    first = class_captions_sha256(load_class_captions(tmp_path))

    path.write_text(json.dumps(document, indent=4))

    assert class_captions_sha256(load_class_captions(tmp_path)) == first
