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
- a cut cell: the scale kernel has the full grid cell measure, while the cut-cell update uses
  the embedded fluid volume;
- an initial value projected on the analytic route: the b6 ``ConservativeCellAverage`` gives
  averages; the projection by encode is N16 lot 2's (plan/routes_transitoires.md, line 4);
- any Program operation other than the stored state, its linear combinations, the three rates
  that read the decoded state (rhs, diffusive_rhs and a named source) and
  the Advance assessment ``finite()`` of the stored candidate coefficients (``finite_guard_values``);
  a stencil flux remains outside this set until a case runs it under a total;
- a prepared boundary component of another block that reads a total through its bound state
  route, which is outside the shared-interface scheduler and still reads accepted storage;
- a layout mapping that reads or overwrites total storage through a cell-average port;
- a consumer of the case that reads the state: a diagnostic reduces the stored totals weighed by
  the cell measure, Σ μ(K) U_K, not the inventory Σ U_K (row D4 of note §4.3), and a field
  output would write the totals U_K under the name of the state (obligation 1 of the record).
"""
from __future__ import annotations

from typing import Any

from pops.time._program.control import ADVANCE_ASSESSMENT_CONTRACT
from pops.time._program.traced_method import NotYetElaborated
from pops.time.solve_outcome import RejectAttempt

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


def _validate_external_storage_consumers(plan: Any, blocks: tuple[Any, ...]) -> None:
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan

    owners = {identity: block.name for block in blocks for identity in block.state_identities}
    for block in plan.blocks:
        if block.numerics is None:
            continue
        for boundary in block.numerics.boundaries:
            data = (boundary.compile_data if type(boundary) is CompiledBoundaryPlan
                    else boundary.compile_boundary_data())
            for row in data["component_region_templates"]:
                for identity in row["states"]:
                    owner = owners.get(identity)
                    if owner is not None and owner != block.name:
                        _refuse(
                            "cell_total_boundary_dependency", owner,
                            "the prepared boundary component of block %r that reads another "
                            "block's accepted storage without its stage density" % block.name,
                            "a later G03 lot with the decoded boundary state dependencies of N16",
                        )
    for mapping in plan.layout_plan.mappings:
        requirement = mapping.requirement
        for port in (requirement.source_port, requirement.target_port):
            owner = owners.get(port.subject.qualified_id)
            if owner is not None:
                _refuse(
                    "cell_total_layout_mapping", owner,
                    "the layout mapping %r with a state port that assumes cell averages"
                    % requirement.qualified_id,
                    "a later G03 lot with a representation-aware layout transfer",
                )


def _validate_volume_authority(plan: Any, blocks: tuple[Any, ...]) -> None:
    cut_layouts = {
        row.handle for row in plan.layout_plan.layouts
        if (row.options.get("embedded_boundary") or {}).get("transport", {}).get("mode") == "cutcell"
    }
    if not cut_layouts:
        return
    names = {block.name for block in blocks}
    for assignment in plan.layout_plan.assignments:
        if assignment.subject_kind == "block" and assignment.subject.local_id in names \
                and assignment.layout in cut_layouts:
            _refuse(
                "cell_total_cut_cell_measure", assignment.subject.local_id,
                "the cut-cell measure on layout %r: the scale kernel uses the full grid cell "
                "measure and requires the fluid volume instead" % assignment.layout.local_id,
                "a later G03 lot with the cut-cell measures of §4.3",
            )


def finite_guard_values(values: Any) -> frozenset[int]:
    """Identify the default Advance assessment of the exact candidate's stored coefficients.

    The Advance issues the guard and its RejectAttempt action. The all-component norm reads the
    candidate, and the norm and predicate are reserved for this assessment. Decoding U/μ(K) is
    outside this test and can overflow for finite U. Every other reduction remains unsupported.
    """
    values = tuple(values)
    readers: dict[int, list[int]] = {}
    for value in values:
        for source in value.inputs:
            readers.setdefault(source.id, []).append(value.id)
    served: set[int] = set()
    for guard in values:
        if (guard.op != "acceptance_guard" or guard.vtype != "state" or len(guard.inputs) != 2
                or set(guard.attrs) != {"guard", "action", "assessment"}
                or guard.attrs["guard"] != "finite"
                or guard.attrs["assessment"] != ADVANCE_ASSESSMENT_CONTRACT
                or type(guard.attrs["action"]) is not RejectAttempt):
            continue
        candidate, test = guard.inputs
        if (candidate.vtype != "state" or test.op != "scalar_isfinite" or test.vtype != "bool"
                or len(test.inputs) != 1 or readers.get(test.id) != [guard.id]):
            continue
        (reduction,) = test.inputs
        if (reduction.op != "reduce" or reduction.vtype != "scalar"
                or reduction.attrs.get("kind") != "norm_inf_all"
                or reduction.attrs.get("component_contract") != "pops.program.norm-inf-all@1"
                or len(reduction.inputs) != 1 or reduction.inputs[0] is not candidate
                or readers.get(reduction.id) != [test.id]):
            continue
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
    _validate_external_storage_consumers(plan, blocks)
    _validate_volume_authority(plan, blocks)
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
