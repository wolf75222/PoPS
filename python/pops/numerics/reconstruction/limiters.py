"""Executable slope-limiter descriptors derived from the native route catalog."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pops.descriptors import _native
from pops.numerics.indicator_stencils import FOURTH_ORDER_AXIS, SECOND_ORDER_AXIS
from pops.runtime.routes import LIMITER_MINMOD, LIMITER_VANLEER, Route


def _native_reconstruction_descriptor(
    route: Route, *, category: str, name: str | None = None, **options: Any
) -> Any:
    """Build one descriptor from an authenticated generated limiter route.

    Formal order and halo depth are catalogue facts.  Factories may add controls such as the
    WENO regulariser or the selected MUSCL limiter, but they cannot restate or override those
    structural facts.
    """
    if not isinstance(route, Route) or route.family != "limiter":
        raise TypeError("reconstruction descriptors require a generated limiter Route")
    metadata = route.metadata
    formal_order = metadata.get("formal_order")
    ghost_depth = metadata.get("n_ghost")
    muscl_compatible = metadata.get("muscl_compatible")
    if (isinstance(formal_order, bool) or not isinstance(formal_order, int)
            or isinstance(ghost_depth, bool) or not isinstance(ghost_depth, int)
            or not isinstance(muscl_compatible, bool)):
        raise RuntimeError("generated limiter route %s has an incomplete stencil contract" % route.id)
    gradient_stencil = FOURTH_ORDER_AXIS if formal_order >= 4 else SECOND_ORDER_AXIS
    return _native(
        name or route.token,
        route.native_entry,
        route.token,
        category=category,
        formal_order=formal_order,
        ghost_depth=ghost_depth,
        muscl_compatible=muscl_compatible,
        amr_gradient_stencil=gradient_stencil,
        **options,
    )

limiters = SimpleNamespace(
    Minmod=lambda: _native_reconstruction_descriptor(
        LIMITER_MINMOD, category="limiter"),
    VanLeer=lambda: _native_reconstruction_descriptor(
        LIMITER_VANLEER, category="limiter"),
)

# Spec 5: expose the limiters at module scope.
Minmod = limiters.Minmod
VanLeer = limiters.VanLeer

__all__ = ["limiters", "Minmod", "VanLeer"]
