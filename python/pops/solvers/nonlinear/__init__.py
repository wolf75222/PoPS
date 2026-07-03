"""pops.solvers.nonlinear -- the nonlinear-solver catalog (Spec 5 sec.5.7).

There is no standalone ``pops::Newton`` / ``pops::FixedPoint`` solver TYPE: Newton is the
implicit-stepper kernel (pops.time), and a fixed-point iteration is authored over the Krylov
primitives. So both are catalogued here as PLANNED descriptors (``available=False``, empty
native id) -- they name the slot without overclaiming a symbol. This is the ONE public home of
the ``solvers.Newton`` / ``solvers.FixedPoint`` entries formerly parked under
``pops.lib.solvers`` (that shim is removed; no second public path).

NOTE: the ``newton`` keyword of :mod:`pops.time` (the implicit time-stepper role) is a
DIFFERENT object from :func:`Newton` here (a solver descriptor); they share no namespace.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import _planned


def Newton(**options: Any) -> Any:
    """A planned Newton nonlinear solver descriptor (no native solver type yet; inert).

    Newton's iteration is realised by the implicit time-stepper kernel (pops.time), so there is
    no standalone ``pops::Newton`` -- this is a catalogued ``available=False`` slot.
    """
    return _planned("newton", "newton", category="solver", **options)


def FixedPoint(**options: Any) -> Any:
    """A planned fixed-point (Picard) nonlinear solver descriptor (no native type yet; inert)."""
    return _planned("fixed_point", "fixed_point", category="solver", **options)


__all__ = ["Newton", "FixedPoint"]
