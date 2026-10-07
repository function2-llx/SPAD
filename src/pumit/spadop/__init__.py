# spadop: spatially adaptive operation

# Deprecated: external interfaces (stream shards, config) use None for 2D.
# DA_2D is kept only for internal pool_d math (da < DA_2D means "3D sample").
# Will be replaced with None checks gradually.
DA_2D = 1 << 30

from .conv import *
from .resample import *
from .patch_embed import *
