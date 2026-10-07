"""UCPT text-conditioned segmentation components (SAM 3-based, SPAD 3D)."""

from pumit.ucpt.seg.neck import SPADNeck
from pumit.ucpt.seg.decoder import FusionEncoder, SemanticHead, PixelDecoder
from pumit.ucpt.seg.loss import seg_loss
from pumit.ucpt.seg.text_encoding import TextEmbeddingCache, TextEncoder
from pumit.ucpt.seg.schedule import neck_da_schedule, StageDA

__all__ = [
    'SPADNeck', 'FusionEncoder', 'SemanticHead', 'PixelDecoder',
    'seg_loss', 'TextEmbeddingCache', 'TextEncoder', 'neck_da_schedule', 'StageDA',
]
