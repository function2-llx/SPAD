from torch import nn

type tuple2_t[T] = tuple[T, T]
type tuple3_t[T] = tuple[T, T, T]
type param3_t[T] = T | tuple3_t[T]


class NoWeightDecayParameter(nn.Parameter):
    """Explicitly indicate that a parameter requires no weight decay."""
    pass
