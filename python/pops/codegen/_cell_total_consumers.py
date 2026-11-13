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
  averages; the projection by encode is N16 lot 2's (plan/routes_transitoires.md, line 4);
- any Program operation other than the stored state, its linear combinations, the rates that
  read the decoded state (``rhs``, ``diffusive_rhs`` and, since N15 lot FC-1, a named ``source``;
  a fused source, a pointwise solve or a local transform stays refused; a stencil flux is refused until a case runs it under a total) and
  the acceptance test ``finite()``, which reads the same on U as on ū (``finite_guard_values``);
- a consumer of the case that reads the state: a diagnostic reduces the stored totals weighed by
  the cell measure, Σ μ(K) U_K, not the inventory Σ U_K (row D4 of note §4.3), and a field
  output would write the totals U_K under the name of the state (obligation 1 of the record).
"""
from __future__ import annotations

from typing import Any

from pops.time._program.traced_method import NotYetElaborated

# The operations that a Program of a total may hold: the state, its combinations, and the rates
# that read the decoded density (program_emit_totals.TOTAL_RATE_OPS), a named source among them
# (N15 FaceFlux/CellRate lot FC-1).
CELL_TOTAL_PROGRAM_OPS = frozenset({"state", "linear_combine", "rhs", "diffusive_rhs", "source"})

# The owner of the projection of an initial value by the encode of its representation.
_PROJECTION_BY_ENCODE = ("N16 lot 2 (the projection of an initial value by the encode of its "
                         "representation: cell.integral, the sample, the Riesz map of a rule)")


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


def finite_guard_values(values: Any) -> frozenset[int]:
    """The ids of the values of the acceptance test ``finite()`` (``pops.Advance``, accept
    finite(), the default of the surface): ``acceptance_guard(guard="finite")`` of
    ``scalar_isfinite(reduce(kind="norm_inf_all"))``, each value read by the next one only.
    U = μ(K) ū with μ(K) finite and positive, so U is finite exactly when ū is: the test reads the
    same on a total as on its density, and a total serves it (N16, the lot that lifts the table for
    the three words). Any other reduction of a total is a diagnostic that reads U, refused."""
    values = list(values)
    readers: dict[int, list[Any]] = {}
    for value in values:
        for source in getattr(value, "inputs", ()) or ():
            readers.setdefault(getattr(source, "id", -1), []).append(value)
    served: set[int] = set()
    for guard in values:
        if guard.op != "acceptance_guard" or guard.attrs.get("guard") != "finite":
            continue
        for test in guard.inputs:
            if test.op != "scalar_isfinite" or len(test.inputs) != 1:
                continue
            (reduction,) = test.inputs
            if reduction.op != "reduce" or reduction.attrs.get("kind") != "norm_inf_all":
                continue
            if all(reader.op == "scalar_isfinite" for reader in readers.get(reduction.id, ())) \
                    and all(reader.op == "acceptance_guard" and reader.attrs.get("guard") ==
                            "finite" for reader in readers.get(test.id, ())):
                served.update((guard.id, test.id, reduction.id))
    return frozenset(served)


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
    values = list(all_ops(plan.time))
    guards = finite_guard_values(values)
    ops = sorted({value.op for value in values if value.id not in guards}
                 - CELL_TOTAL_PROGRAM_OPS)
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
    totals = {state for block in blocks for state in block.numerics.cell_total_states()}
    graph = getattr(plan, "consumer_graph", None)
    for node in () if graph is None else graph.nodes:
        if any(quantity.reference in totals for quantity in node.diagnostic_quantities):
            _refuse("cell_total_diagnostic", name,
                    "the diagnostic of consumer %r, which reduces the stored totals weighed by the "
                    "cell measure (the sum of mu(K) U_K, not the inventory, the sum of U_K: row D4 "
                    "of note 4.3)" % (node.qualified_id,),
                    "G03 lot M1, which reads the inventory of D4")
        if any(quantity.reference in totals for quantity in node.quantities):
            _refuse("cell_total_output", name,
                    "the output of consumer %r, which would write the totals U_K under the name of "
                    "the state" % (node.qualified_id,),
                    "G03 lot M1, which writes a total under its elaborated identity")
    initial = plan.initial_condition_plan
    for binding in () if initial is None else initial.bindings:
        route = binding.source.options.to_data().get("native_route")
        if route != "bound_level_zero":
            _refuse("cell_total_initial_projection", name,
                    "the initial route %r, which projects cell averages; a total is bound as an "
                    "array of totals (BindArray)" % (route,),
                    _PROJECTION_BY_ENCODE)


def validate_represented_initials(plan: Any) -> None:
    """A sample (``point_value``) or an average under a stated rule (``cell_average(rank,
    quadrature=Q)``) has the rate of an average, so the lowering serves it; its initial value
    differs from the average's by its projection only, the sample at the centre or the projection
    under Q, which the analytic route of b6 does not give (``ConservativeCellAverage``, line 4 of
    plan/routes_transitoires.md). Such a state is served bound as data (``BindArray``), and every
    other initial route is refused before any C++ (N16, the lot that lifts the table for the three
    words; review of #803, R1)."""
    from pops.representations.definition import cell_average, constant_mode_scale

    initial = plan.initial_condition_plan
    if initial is None:
        return
    for block in plan.blocks:
        for row in getattr(block.numerics, "representations", ()):
            method = row.method
            if constant_mode_scale(method) == "unit" or \
                    method.identity == cell_average(method.rank).identity:
                continue
            for binding in initial.bindings:
                if binding.subject != row.subject:
                    continue
                route = binding.source.options.to_data().get("native_route")
                if route != "bound_level_zero":
                    raise CellTotalNotYetElaborated(
                        "represented_initial_projection",
                        "the state of block %r is represented by %s (kind %r%s); the initial "
                        "route %r projects cell averages, not its encode, so it is bound as an "
                        "array of its coefficients (BindArray); %s gives the projection"
                        % (block.name, row.method.identity[-12:], method.kind,
                           "" if method.pairing is None else ", with a stated rule", route,
                           _PROJECTION_BY_ENCODE))


__all__ = ["CELL_TOTAL_PROGRAM_OPS", "CellTotalNotYetElaborated", "finite_guard_values",
           "total_blocks",
           "validate_cell_total_consumers", "validate_represented_initials"]
