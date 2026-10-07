"""Experiment integrations for SPAD U-Net systems."""

from .corpus_grid import CorpusGridUniversalTrainer
from .spad import SPADUNetTrainer
from .spad_universal import SPADUniversalTrainer

__all__ = [
    'CorpusGridUniversalTrainer',
    'SPADUNetTrainer',
    'SPADUniversalTrainer',
]
