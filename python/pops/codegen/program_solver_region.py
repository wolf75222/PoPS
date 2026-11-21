"""A solver's region records no face expression (N9, ROM-90: the cost of a reference solve).

A reference solve (SV12) lowers its solver's body as a region. The region's value leaves it by a
raw copy (``PureFieldAlgebra::copy``) into ``sf_sol<id>``, a buffer whose face-record expression is
reset at its allocation and that no recorded operation writes, as the native CG writes its
solution into a scratch reset at allocation. b6's face-record expressions
(``FieldExchangeExpressions``) are kept for the publication of an accepted state; no field of a
solver's region is published, so every expression computed in the region is discarded at its
boundary. Computing them was the cost of the reference route: the faces of each StencilFlux apply
captured as records, and every combination of fields re-weighing them.

Inside a solver's region the lowering therefore emits the pure field algebra of the native solvers
(the same ``pops::saxpy`` / ``pops::lincomb`` kernels, so the values are bit for bit those of the
recorded algebra), captures no face record, brackets no write and attaches no affine expression.
The fields that the region's operations write are context scratches, reset at their access; a
carry of the region's loop is a structural copy whose expression no operation of the region reads.
The interface staging of a model's default transport (``stage_exchange_batch``) is not an
expression and is lowered as before.
"""
from __future__ import annotations

from typing import Any

SOLVER_REGION = ("solver_region",)


def enter_solver_region(var: dict[Any, Any], solve_id: int) -> None:
    """Mark ``var`` (the region's own copy of the emission map) as the region of solve ``solve_id``.

    Nested regions copy the map, so the mark holds for the solver's whole body."""
    var[SOLVER_REGION] = int(solve_id)
    var[("stencil_flux_expressions",)] = False


def in_solver_region(var: Any) -> bool:
    return var.get(SOLVER_REGION) is not None


def emit_axpy(var: Any, destination: str, coefficient: str, source: str,
              metadata: str | None = None) -> str:
    """``destination += coefficient * source``: recorded outside a solver's region, pure inside."""
    if in_solver_region(var):
        return "pops::PureFieldAlgebra::axpy(%s, %s, %s);" % (destination, coefficient, source)
    if metadata is None:
        return "ctx.axpy(%s, %s, %s);" % (destination, coefficient, source)
    return "ctx.axpy(%s, %s, %s, dt, %s);" % (destination, coefficient, source, metadata)


def emit_lincomb(var: Any, destination: str, left_coefficient: str, left: str,
                 right_coefficient: str, right: str) -> str:
    """``destination = a * left + b * right``: recorded outside a solver's region, pure inside."""
    call = ("pops::PureFieldAlgebra::lincomb" if in_solver_region(var) else "ctx.lincomb")
    return "%s(%s, %s, %s, %s, %s);" % (call, destination, left_coefficient, left,
                                         right_coefficient, right)
