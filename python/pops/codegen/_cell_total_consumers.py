"""What a cell total is served by before the face pass moves into regions (G03 lot R-b, ROM-21).

A state represented by ``cell_total(rank)`` is lowered by decoding ū = U/μ(K), running today's
rate on ū and encoding the rate of U as μ(K) times it (``program_emit_totals.py``). That identity
holds on a static configuration, for the rates whose evaluation goes through the decoded state.
Every other consumer is refused at resolution, before any C++, by name and with the lot that
serves it (note §4.3, §13, §14; driver's rulings of 12/11 23:00 and 23:05):

- a step rule: the native step selection reads the stored state U, not ū (B1), so a total takes a
  fixed step until lot T teaches the selection the represented state;
- a hierarchy: restriction of a total is a sum and prolongation gives each child μ(child)·ū of
  its parent, so today's transfer is wrong on U (C1, G11);
- a moving configuration: the volume member of G03 lots M1 and M2;
- an initial value projected on the analytic route: the b6 ``ConservativeCellAverage`` gives
  averages; the encode of a projection waits for N16's table (plan/routes_transitoires.md, line 4);
- any Program operation other than the stored state, its linear combinations and the three rates
  that read the decoded state.
"""
from __future__ import annotations

from typing import Any

from pops.time._program.traced_method import NotYetElaborated

# The operations that a Program of a total may hold: the state, its combinations, and the rates
# that read the decoded density (program_emit_totals.TOTAL_RATE_OPS).
CELL_TOTAL_PROGRAM_OPS = frozenset({"state", "linear_combine", "rhs", "diffusive_rhs",
                                    "stencil_flux_divergence"})


class CellTotalNotYetElaborated(NotYetElaborated):
    """A consumer of a cell total that this lowering does not serve; ``reason`` names which."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _refuse(reason: str, block: str, what: str, lot: str) -> None:
    raise CellTotalNotYetElaborated(
        reason, "the state of block %r is represented by cell totals (cell_total, G03 lot R-b); "
        "%s is not served on a total yet; %s gives it" % (block, what, lot))


def total_blocks(plan: Any) -> tuple[Any, ...]:
    """The blocks of a resolved plan whose evolved state holds cell totals."""
    rows = []
    for block in plan.blocks:
        states = getattr(block.numerics, "cell_total_states", None)
        if callable(states) and states():
            rows.append(block)
    return tuple(rows)


def validate_cell_total_consumers(plan: Any) -> None:
    """Refuse, before any C++, every consumer of a cell total that its lowering does not serve."""
    from pops.codegen.program_lowerability import all_ops
    from pops.time._step.strategy import FixedDt

    blocks = total_blocks(plan)
    if not blocks:
        return
    name = blocks[0].name
    for block in blocks:
        totals = block.numerics.cell_total_states()
        if len(block.numerics.representations) != len(totals) or len(totals) != 1:
            _refuse("cell_total_mixed_block", block.name,
                    "a block that holds several states, or totals beside other representations",
                    "a later G03 lot with the record states of N8")
    if any(target == "amr_system" for target in plan.layout_targets.values()):
        _refuse("cell_total_hierarchy", name,
                "a hierarchy of levels, whose restriction of a total is a sum and whose "
                "prolongation gives each child mu(child) times the density of its parent", "G11")
    if plan.time is None:
        _refuse("cell_total_program", name, "a case without a Program", "the Program of the case")
    ops = sorted({value.op for value in all_ops(plan.time)} - CELL_TOTAL_PROGRAM_OPS)
    if any(op in ("geometry_state", "reynolds_update") for op in ops):
        _refuse("cell_total_moving", name, "a moving configuration, whose volume is a member of "
                "the candidate", "G03 lots M1 and M2")
    if ops:
        _refuse("cell_total_operation", name, "the Program operation(s) %s" % ", ".join(ops),
                "a later G03 lot (lot A for an accumulation, N16 lot 2 for a transform)")
    for value in all_ops(plan.time):
        if value.op == "rhs" and value.attrs.get("path_conservative", False):
            _refuse("cell_total_operation", name, "a path-conservative rate",
                    "a later G03 lot with the nonconservative products of N16")
    strategy = getattr(plan.time, "_step_strategy", None)
    if type(strategy) is not FixedDt:
        _refuse("cell_total_step_rule", name,
                "the step strategy %s: the native step selection reads the stored totals, not "
                "their density, so a total takes dt= a fixed step" % type(strategy).__name__,
                "G03 lot T")
    initial = plan.initial_condition_plan
    for binding in () if initial is None else initial.bindings:
        route = binding.source.options.to_data().get("native_route")
        if route != "bound_level_zero":
            _refuse("cell_total_initial_projection", name,
                    "the initial route %r, which projects cell averages; a total is bound as an "
                    "array of totals (BindArray)" % (route,),
                    "the N16 lot that retires the transitional table of CellAverage()")


__all__ = ["CELL_TOTAL_PROGRAM_OPS", "CellTotalNotYetElaborated", "total_blocks",
           "validate_cell_total_consumers"]
