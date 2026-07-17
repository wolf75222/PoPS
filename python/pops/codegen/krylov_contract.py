"""Shared validation of the authenticated prepared-Krylov IR footprint."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_KRYLOV_METHODS = frozenset({"cg", "bicgstab", "gmres", "richardson"})
_PRECONDITIONED_METHODS = frozenset({"bicgstab", "gmres"})
_PRECONDITIONERS = frozenset({"identity", "geometric_mg"})
_FOOTPRINT_KEYS = frozenset({
    "components", "input_ghosts", "restart", "preconditioned",
})


def _exact_int(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            "solve_linear Krylov footprint %s must be an integer >= %d" % (label, minimum))
    return value


def _authenticated_operator_footprint(operator: Any) -> tuple[int, int]:
    """Return the independently authored operator shape consumed by Krylov.

    The solve node duplicates these two values so scratch inspection remains self-contained, but
    that duplicate is not an authority: codegen must bind it back to the typed operator declaration
    before allocating native fields.  Keep the import lazy so importing :mod:`pops.codegen` does not
    eagerly import the time DSL.
    """
    operator_attrs = getattr(operator, "attrs", None)
    if (
        getattr(operator, "op", None) != "matrix_free_operator"
        or not isinstance(operator_attrs, Mapping)
    ):
        raise ValueError("solve_linear requires an authenticated matrix_free_operator input")
    operator_components = _exact_int(
        operator_attrs.get("ncomp"), label="operator component count", minimum=1)

    from pops.time.stencil import StencilAccess
    stencil_access = operator_attrs.get("stencil_access")
    if type(stencil_access) is not StencilAccess:
        raise ValueError("solve_linear operator has no authenticated StencilAccess")
    operator_ghosts = _exact_int(
        stencil_access.required_ghost_depth,
        label="operator input_ghosts", minimum=0)
    return operator_components, operator_ghosts


def validated_krylov_footprint(
    attrs: Mapping[str, Any], *, operator: Any
) -> dict[str, Any]:
    """Return the exact canonical footprint or reject a malformed/tampered solve node.

    Code emission and inert scratch inspection consume this one validator so neither can coerce
    booleans/strings into plausible counts or silently disagree about method, restart, or actual
    preconditioner presence.
    """
    method = attrs.get("method")
    if method not in _KRYLOV_METHODS:
        raise ValueError("solve_linear has an unauthenticated Krylov method %r" % (method,))

    operator_components, operator_ghosts = _authenticated_operator_footprint(operator)
    components = _exact_int(attrs.get("ncomp"), label="operator component count", minimum=1)
    if components != operator_components:
        raise ValueError(
            "solve_linear component count disagrees with its authenticated operator")
    preconditioner = attrs.get("preconditioner")
    if preconditioner not in _PRECONDITIONERS:
        raise ValueError(
            "solve_linear has an unauthenticated prepared preconditioner %r" % (preconditioner,))
    preconditioned = preconditioner != "identity"
    if preconditioned and method not in _PRECONDITIONED_METHODS:
        raise ValueError(
            "solve_linear preconditioning is unavailable for %s" % method)

    raw_restart = attrs.get("restart")
    if method == "gmres":
        restart = _exact_int(raw_restart, label="GMRES restart", minimum=1)
    else:
        if raw_restart is not None:
            raise ValueError(
                "solve_linear restart belongs only to GMRES; got %r for %s"
                % (raw_restart, method))
        restart = 0

    footprint = attrs.get("krylov_footprint")
    if not isinstance(footprint, Mapping) or set(footprint) != _FOOTPRINT_KEYS:
        raise ValueError("solve_linear requires an exact typed Krylov footprint")
    footprint_components = _exact_int(
        footprint["components"], label="components", minimum=1)
    if footprint_components != components:
        raise ValueError("solve_linear Krylov footprint component count is unauthenticated")
    input_ghosts = _exact_int(
        footprint["input_ghosts"], label="input_ghosts", minimum=0)
    if input_ghosts != operator_ghosts:
        raise ValueError(
            "solve_linear Krylov footprint input_ghosts disagrees with its authenticated operator")
    footprint_restart = _exact_int(
        footprint["restart"], label="restart", minimum=0)
    if footprint_restart != restart:
        raise ValueError("solve_linear Krylov footprint restart disagrees with method controls")
    footprint_preconditioned = footprint["preconditioned"]
    if not isinstance(footprint_preconditioned, bool):
        raise ValueError("solve_linear Krylov footprint preconditioned must be a boolean")
    if footprint_preconditioned != preconditioned:
        raise ValueError(
            "solve_linear Krylov footprint disagrees with prepared preconditioner presence")

    return {
        "components": components,
        "input_ghosts": input_ghosts,
        "restart": restart,
        "preconditioned": preconditioned,
    }
