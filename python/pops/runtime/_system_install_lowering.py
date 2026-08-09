"""Typed boundary-condition and numerical-option lowerers.

Split out of :mod:`pops.runtime._system_install` for the 500-line cap (ADC-550): the two
The boundary lowerer turns a typed native boundary brick into the ``bc`` token passed to the
private native seam. Embedded geometry follows the separate exact level-set authority.

Pure lowering: no ``_pops`` import, no numeric work.
"""

from __future__ import annotations

from typing import Any

from pops.runtime._numeric import native_real


def _lower_bc(bc: Any) -> Any:
    """Lower a Poisson boundary condition to the native ``bc`` token (Spec 5 sec.14.2.6).

    A typed native boundary descriptor lowers through its small ``.bc`` interface. Strings are
    rejected before route validation.
    """
    if isinstance(bc, str):
        raise TypeError(
            "set_poisson: bc must be a typed native boundary descriptor "
            "(Dirichlet / Neumann / Periodic); string selectors are not accepted"
        )
    token = getattr(bc, "bc", None)  # native _Boundary brick carries its token on .bc
    if isinstance(token, str):
        return token
    raise TypeError(
        "set_poisson: bc must be a typed native boundary descriptor "
        "(Dirichlet / Neumann / Periodic), got %s" % type(bc).__name__
    )


__all__ = ["_cartesian_cg_kwargs", "_lower_bc", "_weno_kwargs"]


def _weno_kwargs(spatial):
    """ADC-645: WENO5(epsilon=...) rides along the Spatial; None (the default) forwards NOTHING so
    the native ABI keeps its kWenoEpsilon default (byte-identical historical call)."""
    weps = getattr(spatial, "weno_epsilon", None)
    return (
        {}
        if weps is None
        else {"weno_epsilon": native_real(weps, where="System.add_equation.weno_epsilon")}
    )


def _cartesian_cg_kwargs(rel_tol, max_iterations):
    """Forward only authored exact-ranked Cartesian-CG controls to the native seam."""
    out = {}
    if rel_tol is not None:
        out["rel_tol"] = native_real(rel_tol, where="System.set_poisson.rel_tol")
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("System.set_poisson.max_iterations must be a Python int")
        out["max_iterations"] = max_iterations
    return out
