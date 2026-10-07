"""Parameter-free 3D sine position encoding for the seg neck and fusion encoder.

The in-plane (H, W) part copies SAM 3's ``PositionEmbeddingSine`` exactly: coordinates ``arange(1, N+1)``
normalized to ``(0, 2*pi]`` by the grid extent, paired periods (each sin/cos pair shares one period),
interleaved ``[sin(t0), cos(t0), sin(t1), cos(t1), ...]`` layout, and the H (y) block concatenated before
the W (x) block. Depth is folded INTO both in-plane angles rather than carving a third channel block:

    angle_k(h, d) = h_norm / f_k + d / g_k

with ``f`` the in-plane period family and ``g`` a DISTINCT depth period family sharing the same pairing (so
each interleaved sin/cos pair still sees one angle; period = 1/frequency, the reciprocal of rope.py's
inv_freq). The depth coordinate uses the centered extent-normalized map
``2*pi * ((2*(l + 0.5) / D) - 1)`` (rope.py's convention, deliberately NOT SAM 3's one-based end-anchored
in-plane map): the single slice of a ``d == 1`` grid sits at coordinate 0, the depth term drops out in every
band, and the encoding equals SAM 3's 2D PE bit-for-bit, which keeps the SAM 3-pretrained fusion Q/K reading
the positional signal they were trained on. The distinct depth family avoids collapsing depth into a pure
in-plane phase shift, which would make ``(h, w, d)`` collide with ``(h + c', w + c', d - c)`` wherever the
two step sizes are commensurate.

Folding depth keeps the divisibility constraint at ``dim % 4`` (with at least two frequency pairs), so a
256-wide decoder fills exactly with no zero-padding.
"""

from functools import lru_cache
import math

import torch


@lru_cache(maxsize=None)
def _validate_config(dim: int, temperature: float, depth_temperature: float) -> None:
    """Validate scalar inputs and their realized fp32 period families on CPU."""
    if dim < 8 or dim % 4 != 0:
        raise ValueError(f'dim must be divisible by 4 and at least 8, got {dim}')
    for name, value in (('temperature', temperature), ('depth_temperature', depth_temperature)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive, got {value}')

    axis_dim = dim // 2
    k = torch.arange(axis_dim, dtype=torch.float32)
    period = temperature ** (2 * torch.div(k, 2, rounding_mode='floor') / axis_dim)
    period_depth = depth_temperature ** (2 * torch.div(k, 2, rounding_mode='floor') / axis_dim)
    if not period.isfinite().all() or not (period > 0).all():
        raise ValueError(f'temperature={temperature} yields a non-finite or non-positive period')
    if not period_depth.isfinite().all() or not (period_depth > 0).all():
        raise ValueError(f'depth_temperature={depth_temperature} yields a non-finite or non-positive period')
    if torch.equal(period, period_depth):
        raise ValueError(
            f'depth_temperature={depth_temperature} yields an fp32 depth period family identical to the '
            f'in-plane family (temperature={temperature}); use bases whose fp32 families differ.'
        )
    if torch.equal(period_depth, period_depth[0].expand_as(period_depth)):
        raise ValueError(
            f'depth_temperature={depth_temperature} yields only one fp32 depth period; use a wider encoding '
            f'or a base whose realized family has multiple periods.'
        )


def _sincos_interleave(angles: torch.Tensor) -> torch.Tensor:
    """SAM 3 interleave: sin of even channels, cos of odd, woven to ``[sin(t0), cos(t0), sin(t1), ...]``.

    Even/odd channels share a period (the paired-period construction), so each ``(sin, cos)`` pair sees one
    angle.
    """
    return torch.stack((angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1).flatten(-2)


def sine_pos_embed_3d(
    d: int, h: int, w: int,
    dim: int,
    device: torch.device,
    temperature: float = 10000.0,
    depth_temperature: float = 1000.0,
) -> torch.Tensor:
    """Build sine position encodings over a 3D grid.

    See the module docstring: SAM 3-exact in-plane form, depth folded into both in-plane angles with a
    distinct paired frequency family, exact SAM 3 2D reduction at ``d == 1``.

    Args:
        d: Grid depth.
        h: Grid height.
        w: Grid width.
        dim: Encoding width, divisible by four and at least eight.
        device: Output device.
        temperature: In-plane (H, W) frequency temperature.
        depth_temperature: Depth frequency temperature; must differ from ``temperature``. The default keeps
            the shared rule with rope.py (depth base = in-plane base / 10): the per-pair depth/in-plane
            frequency ratio sweeps one decade across pairs, which is what prevents any single displacement
            from nearly cancelling all pairs at once.

    Returns:
        Position encodings shaped ``(d * h * w, dim)`` in DHW row-major order.
    """
    # Config validation uses small CPU tensors and is cached for eager callers. The compiled per-sample path
    # skips it so no tensor-to-bool checks split the CUDA graph; training uses the validated fixed defaults.
    if not torch.compiler.is_compiling():
        _validate_config(dim, temperature, depth_temperature)
    axis_dim = dim // 2  # per-axis (H, W) block width; even because dim % 4 == 0

    # In-plane (H, W) coordinates, SAM 3's map (its y, x): arange(1, N+1) normalized by the grid extent to
    # (0, 2*pi]. The +eps is part of SAM 3's exact form; dropping it changes the fp32 bits and breaks the
    # d == 1 warm-start equality.
    eps = 1e-6
    scale = 2 * math.pi
    hh = torch.arange(1, h + 1, dtype=torch.float32, device=device)
    ww = torch.arange(1, w + 1, dtype=torch.float32, device=device)
    hh = hh / (hh[-1] + eps) * scale
    ww = ww / (ww[-1] + eps) * scale
    # Depth: centered extent-normalized map (rope.py's convention), NOT SAM 3's one-based end-anchored form.
    # The accepted inconsistency: in-plane must keep SAM 3's exact map (frozen by the warm start), while depth
    # needs the single slice at coordinate 0 so the depth term vanishes at d == 1 in every band (the
    # end-anchored map puts it at 2*pi, which is a full turn only where the frequency is exactly 1).
    dd = ((2 * (torch.arange(d, dtype=torch.float32, device=device) + 0.5) / d) - 1) * scale

    # Paired periods: each interleaved sin/cos pair shares one period (period = wavelength = 1/frequency; the
    # angle divides the coordinate by it, so it is the reciprocal of rope.py's inv_freq multiplier). The depth
    # family uses the same pairing so the pair still sees a single combined angle.
    k = torch.arange(axis_dim, dtype=torch.float32, device=device)
    period = temperature ** (2 * torch.div(k, 2, rounding_mode='floor') / axis_dim)
    period_depth = depth_temperature ** (2 * torch.div(k, 2, rounding_mode='floor') / axis_dim)

    # Depth folds in ADDITIVELY, so keep it separable: build the in-plane angles once over the (H, W) grid and
    # the depth offset once over D, then broadcast-add to (D, H*W, axis_dim). Meshgridding depth into the
    # product would recompute the same in-plane angle on every slice (and the same depth angle on every pixel)
    # before the sin/cos, discarding the separability that is the whole point of the additive fold-in.
    gh, gw = torch.meshgrid(hh, ww, indexing='ij')
    inplane_h = gh.reshape(-1)[:, None] / period   # (H*W, axis_dim)
    inplane_w = gw.reshape(-1)[:, None] / period
    depth_angles = dd[:, None] / period_depth       # (D, axis_dim)

    pos_h = (inplane_h[None] + depth_angles[:, None]).reshape(-1, axis_dim)  # (D*H*W, axis_dim)
    pos_w = (inplane_w[None] + depth_angles[:, None]).reshape(-1, axis_dim)
    # SAM 3 block order: H (its y) first, then W (its x).
    return torch.cat([_sincos_interleave(pos_h), _sincos_interleave(pos_w)], dim=1)
