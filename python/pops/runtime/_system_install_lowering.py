"""Poisson wall / boundary-condition lowerers for the System install seam.

Split out of :mod:`pops.runtime._system_install` for the 500-line cap (ADC-550): the two
module-level lowerers ``_lower_wall`` / ``_lower_bc`` that turn a typed
:mod:`pops.mesh.geometry` wall or a native boundary brick into the native ``(token, radius)`` /
``bc`` pair ``set_poisson`` passes to the compiled facade. ``_system_install`` re-imports both, so
``from pops.runtime._system_install import _lower_wall`` is unchanged.

Pure lowering: no ``_pops`` import, no numeric work.
"""
from __future__ import annotations

from typing import Any

from pops.runtime._numeric import native_real


def _lower_wall(wall: Any) -> Any:
    """Lower a Poisson ``wall`` to the native ``(wall_token, wall_radius)`` (Spec 5 sec.8.16).

    A typed :mod:`pops.mesh.geometry` wall lowers to its pair (``NoWall`` -> ``("none", 0.0)``,
    ``Disc`` -> ``("circle", radius)``); a non-wall typed geometry raises a clear ``TypeError``.
    Returns ``None`` when ``wall`` is a string: the caller keeps its own ``wall_radius=`` and the
    token is then route-validated by ``set_poisson`` (ADC-584) before the native call.
    """
    if isinstance(wall, str):
        return None
    lower_wall = getattr(wall, "lower_wall", None)
    if lower_wall is None:
        raise TypeError(
            "set_poisson: wall must be a 'none' / 'circle' string or a typed "
            "pops.mesh.geometry wall (NoWall / Disc), got %r" % (type(wall).__name__,))
    return lower_wall()


def _lower_bc(bc: Any) -> Any:
    """Lower a Poisson boundary condition to the native ``bc`` token (Spec 5 sec.14.2.6).

    A typed native boundary brick (``pops.Dirichlet()`` / ``pops.Neumann()`` / ``pops.Periodic()``)
    lowers to its token via its ``.bc`` attribute; a string (including ``"auto"``) passes through
    and is route-validated by ``set_poisson`` (ADC-584). A non-string, non-boundary value raises
    a clear ``TypeError``.
    """
    if isinstance(bc, str):
        return bc
    token = getattr(bc, "bc", None)  # native _Boundary brick carries its token on .bc
    if isinstance(token, str):
        return token
    raise TypeError(
        "set_poisson: bc must be an 'auto' / 'dirichlet' / 'neumann' / 'periodic' string or a typed "
        "native boundary brick (pops.Dirichlet() / Neumann() / Periodic()), got %r"
        % (type(bc).__name__,))


__all__ = ["_lower_wall", "_lower_bc"]


def _weno_kwargs(spatial):
    """ADC-645: WENO5(epsilon=...) rides along the Spatial; None (the default) forwards NOTHING so
    the native add_block keeps its kWenoEpsilon default (byte-identical historical call)."""
    weps = getattr(spatial, "weno_epsilon", None)
    return {} if weps is None else {
        "weno_epsilon": native_real(weps, where="System.add_block.weno_epsilon")}


def _mg_kwargs(rel_tol, max_cycles, min_coarse, pre_smooth, post_smooth, bottom_sweeps,
               coarse_threshold):
    """ADC-613/644: the GeometricMG V-cycle knobs, forwarded ONLY when set (None = unspecified ->
    not passed -> the native kMG*-sourced default, bit-identical); coarse_threshold is the ADC-644
    total-cell coarsening ceiling (0 = disabled)."""
    out = {}
    for key, val, cast in (("rel_tol", rel_tol, native_real), ("max_cycles", max_cycles, int),
                           ("min_coarse", min_coarse, int), ("pre_smooth", pre_smooth, int),
                           ("post_smooth", post_smooth, int), ("bottom_sweeps", bottom_sweeps, int),
                           ("coarse_threshold", coarse_threshold, int)):
        if val is not None:
            out[key] = (cast(val, where="System.set_poisson.%s" % key)
                        if cast is native_real else cast(val))
    return out
