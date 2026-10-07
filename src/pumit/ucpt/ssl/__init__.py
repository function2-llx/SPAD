"""UCPT SSL post-encoder heads (copied from pumit.ssl, archive-boundary rule)."""

from pumit.ucpt.ssl.heads import (
    ClsPredictor,
    PatchDistillDecoder,
    ReconDecoder,
    ssl_post,
)

__all__ = [
    'ReconDecoder',
    'PatchDistillDecoder',
    'ClsPredictor',
    'ssl_post',
]
