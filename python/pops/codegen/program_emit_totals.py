"""The lowering of a cell total around today's rate (G03 lot R-b, ROM-21; note §4.3).

A state represented by ``cell_total(rank)`` (c = 1/cell.measure) stores U_K = ∫_K u. Each rate of
it is lowered in three statements: ``ctx.decode_cell_total`` writes the density ū = U/μ(K) into a
stage field, today's rate runs on ū (the boundary session fills the ghosts of ū by the laws of
the case, which are written on field values), and ``ctx.encode_cell_total_rate`` multiplies the
rate of ū by μ(K), which on a static configuration is the rate of U. An average emits nothing new,
so the generated C++ of every case without a total is unchanged (MG1).

The two native calls are a transitional route (``plan/routes_transitoires.md``): they go when N10
moves the face pass into generated regions and the scale becomes the Riesz ``map`` row of N16
§5.1.
"""
from __future__ import annotations

from typing import Any

# The rate operations that read the stored state and return its rate.
TOTAL_RATE_OPS = frozenset({"rhs", "diffusive_rhs"})
# The scratch subslot of the decoded density; the operations above use 0 to 3 and the named
# sources the following ones, so the decode takes a slot of the state scratch kind instead.
_DECODE_SUBSLOT = 7


def block_holds_totals(model: Any, block: Any) -> bool:
    """Whether the resolved numerics of ``block`` represent its state by cell totals."""
    numerics_for_block = getattr(model, "numerics_for_block", None)
    if block is None or not callable(numerics_for_block):
        return False
    try:
        numerics = numerics_for_block(block)
    except ValueError:
        return False
    states = getattr(numerics, "cell_total_states", None)
    return bool(states()) if callable(states) else False


def begin_total_rate(v: Any, var: dict, lines: list, *, model: Any, block: Any) -> Any:
    """Decode ū before a rate of a total and substitute it for the state; None otherwise."""
    if v.op not in TOTAL_RATE_OPS or v.attrs.get("path_conservative", False):
        return None
    if not block_holds_totals(model, v.block):
        return None
    if block is None:
        raise ValueError("the rate of a cell total requires its resolved block owner")
    state = v.inputs[0]
    stored = var[state.id]
    density = "cell_density_%d" % v.id
    lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%d, %d, %s);"
                 % (density, int(v.id), _DECODE_SUBSLOT, stored))
    lines.append("ctx.decode_cell_total(%s, %s);  // G03 R-b: the density of the total"
                 % (stored, density))
    var[state.id] = density
    return state, stored


def end_total_rate(v: Any, var: dict, lines: list, token: Any) -> None:
    """Restore the stored state and encode the rate of the total."""
    if token is None:
        return
    state, stored = token
    var[state.id] = stored
    lines.append("ctx.encode_cell_total_rate(%s);  // G03 R-b: the rate of the total" % var[v.id])


__all__ = ["TOTAL_RATE_OPS", "begin_total_rate", "block_holds_totals", "end_total_rate"]
