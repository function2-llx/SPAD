"""VoCo-v2 SwinUNETR-L encoder (feature size 96, ~206M): see ``voco`` for the shared design."""

from __future__ import annotations

from pumit.downstream.seg.backbones import voco

CONFIG_KEYS = frozenset()
OPTIONAL_CONFIG_KEYS = voco.OPTIONAL_CONFIG_KEYS

prepare_config, build_encoder, load_pretrained = voco.make_module_functions(96, 'VoCo_L_SSL_head.pt')
