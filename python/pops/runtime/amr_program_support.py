"""Resolved AMR Program capability query and fail-closed pre-artifact gate.

Program IR alone cannot prove an AMR route: hierarchy refinement, shared block interfaces and
authenticated field-provider routes are independent resolve-time authorities.  This module joins
those facts in :class:`AMRProgramSupportContext`, enumerates the Program operations that are
actually reachable, and reports whether every required AMR capability group is implemented.
``pops.resolve`` rejects a pending group before code generation; C++ deferred-operation checks stay
as defensive runtime backstops rather than the primary compatibility mechanism.

Three single sources of truth, none duplicated here:

  1. Op enumeration is the Program's own IR plus an exact resolved support context;
  2. the op -> capability-group map reuses the codegen's OWN op-group vocabulary
     (``program_emit_kernels._CONDENSED_OPS`` et al.), imported lazily -- not a hand list;
  3. the AMR support status derives from the ONE C++ source of truth,
     ``include/pops/runtime/program/amr_program_context.hpp``: every capability deferral there is an
     explicit ``deferred_op("<unambiguous-id>", ...)`` call mirrored in
     :data:`DEFERRED_GROUPS` and LOCKED by
     ``tests/python/architecture/test_amr_program_support_parity.py`` (the ``route_registry_parity``
     pattern). When ADC-631 / ADC-633 remove their throws, the header-derived deferred set shrinks,
     the parity test FORCES the mirror to shrink with it, and the affected group auto-greens -- with
     no edit to any ADC-634 file.

Deliberately IMPORT-FREE of the pops package at module scope (stdlib + typing only): the
architecture gate loads it standalone, without the compiled ``_pops`` module. The codegen op-group
sets and ``Program.ir_nodes`` are reached LAZILY inside the functions that need them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AMRProgramSupportContext:
    """Resolved execution facts required for a complete AMR capability query.

    A Program alone cannot reveal whether its blocks share an interface, whether the hierarchy can
    refine, or whether field-provider routes were authenticated.  Resolve constructs this value only
    after those independent authorities have been validated; callers cannot obtain a green route
    verdict from Program IR alone.
    """

    hierarchy_level_count: int
    frozen_hierarchy: bool
    shared_block_interfaces: bool
    field_routes_validated: bool

    def __post_init__(self) -> None:
        if type(self.hierarchy_level_count) is not int:
            raise TypeError("AMRProgramSupportContext.hierarchy_level_count must be int")
        if self.hierarchy_level_count < 1:
            raise ValueError("AMRProgramSupportContext.hierarchy_level_count must be positive")
        for name in (
            "frozen_hierarchy", "shared_block_interfaces", "field_routes_validated",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("AMRProgramSupportContext.%s must be bool" % name)

    @property
    def refined_hierarchy(self) -> bool:
        """Whether the resolved hierarchy can materialize a fine level."""
        return self.hierarchy_level_count > 1

    @property
    def supports_shared_interface_fragments(self) -> bool:
        """Whether the installed ledger route serves this resolved hierarchy policy."""
        if self.frozen_hierarchy:
            return True
        # Resolve validation admits only the exact scheduled-regrid policy on this branch.
        return self.hierarchy_level_count >= 2


class AMRProgramSupportError(ValueError):
    """Resolved AMR execution facts are incompatible with a reachable Program operation."""

# --- Capability groups: the ONE mirror of the AmrProgramContext deferral surface ----------------
# Each group names (a) the AmrProgramContext C++ methods that FAIL LOUD for it -- the header-derived
# deferred identifiers the parity test locks against amr_program_context.hpp -- and (b) the Python
# IR op names that route into that group. ``issue`` is the follow-up that greens the group (None for
# a group with no scheduled implementation). A group whose ``header_methods`` is EMPTY is served
# today (green); a non-empty one is pending. The header_methods are exactly the unambiguous string
# identifiers passed to ``deferred_op("<name>", ...)``. Ordinary validation/runtime exceptions are
# deliberately outside this capability mirror. This is the SINGLE place the
# AMR support status is declared; :func:`deferred_groups` / :func:`amr_program_op_support` read it.
#
# op_source names the codegen op-group set this group's ir_ops mirror (documentary: the parity of the
# ir_ops against those codegen sets is asserted by the support parity test, so the emit vocabulary
# stays the single source and this table cannot silently drift from it).
DEFERRED_GROUPS: dict = {
    "condensed": {
        # ADC-633 WIRED the condensed-implicit Program on the hierarchy and ADC-637 made the generic
        # condensed_* ops the sole route: per-level assembly runs through AmrProgramContext::grid_context /
        # assembly_target / assembly_source, and solve_prepared_linear dispatches flat->prepared BiCGStab
        # / refined->composite FAC. No throw stub remains, so header_methods is EMPTY -> the group is GREEN.
        "issue": "ADC-633",
        "op_source": "program_emit_kernels._CONDENSED_OPS",
        "ir_ops": frozenset({"condensed_coeffs", "condensed_rhs", "condensed_reconstruct"}),
        "header_methods": frozenset(),
    },
    "named_flux": {
        "issue": None,
        "op_source": "program_emit_kernels._named_fluxes (rhs with named fluxes)",
        "ir_ops": frozenset({"neg_div_flux_into"}),
        "header_methods": frozenset({"neg_div_flux_into"}),
    },
    "projection": {
        "issue": None,
        "op_source": "program_emit_kernels._ALLOWED_OPS['project']",
        "ir_ops": frozenset({"project"}),
        "header_methods": frozenset(),
    },
    "coupled_solve": {
        "issue": None,
        "op_source": "Program IR solve_fields_from_blocks -> program_emit_ops "
                     "ctx.solve_fields_from_blocks_at",
        "ir_ops": frozenset({"solve_fields_from_blocks"}),
        "header_methods": frozenset(),
    },
    "named_field_solve": {
        "issue": None,
        "op_source": "Program IR solve_fields -> program_emit_ops "
                     "ctx.solve_fields_from_state_at",
        "ir_ops": frozenset({"solve_fields"}),
        "header_methods": frozenset(),
    },
    "unqualified_coupled_solve": {
        "issue": None,
        "op_source": "not representable in final Program IR (field identity is mandatory)",
        "ir_ops": frozenset(),
        "header_methods": frozenset({"solve_fields_from_blocks_default"}),
    },
    "schedule_control": {
        "issue": None,
        "op_source": "program_emit_schedule schedule_decision(..., cache_backed=false)",
        "ir_ops": frozenset(),  # scheduling is an attr on an op node, not a distinct IR op
        "header_methods": frozenset(),
    },
    "schedule_cache": {
        "issue": None,
        "op_source": "program_emit_schedule cache_* actions and checkpointed field Hold",
        "ir_ops": frozenset(),  # scheduling is an attr on an op node, not a distinct IR op
        "header_methods": frozenset({"cache_should_update", "cache_store_aux", "cache_restore_aux",
                                    "cache_store_scratch", "cache_restore_scratch",
                                    "cache_accumulate_dt", "cache_effective_dt"}),
    },
}


def header_deferred_methods() -> frozenset:
    """The FULL set of AmrProgramContext deferral method identifiers this mirror declares.

    The union of every group's ``header_methods`` -- the mirror of the deferral surface in
    ``amr_program_context.hpp``. ``test_amr_program_support_parity`` parses the header and asserts the
    parsed set equals this one, so the mirror cannot drift from the C++ source of truth.
    """
    methods: set = set()
    for group in DEFERRED_GROUPS.values():
        methods |= set(group["header_methods"])
    return frozenset(methods)


def deferred_groups() -> dict:
    """The per-group AMR Program support status: ``{group: "green" | "pending:ADC-6xx"}``.

    A group with no declared deferral method (``header_methods`` empty) is served on the AMR Program
    path today (``"green"``); a group that still defers reports ``"pending:<issue>"`` (or a bare
    ``"pending"`` when no follow-up issue is scheduled). Read-only: it derives entirely from
    :data:`DEFERRED_GROUPS`, the single mirror of the header.
    """
    status = {}
    for name, group in DEFERRED_GROUPS.items():
        status[name] = _group_status(group)
    return status


def amr_program_op_support(
    program: Any, *, context: AMRProgramSupportContext,
) -> dict:
    """Report the AMR Program op support for the ops @p program actually USES: ``{group: status}``.

    Enumerates @p program's ops via its own ``ir_nodes()`` (the ``_pops``-free IR walk) plus the
    named-flux / scheduled derivations that are ATTRS on an op rather than a distinct op, maps each
    used op to its capability group via :data:`DEFERRED_GROUPS`, and returns the support status of
    every group the Program touches (``"green"`` when the AMR path serves it, ``"pending:ADC-6xx"``
    when a follow-up must land first). A group the Program does not use is OMITTED, so an all-explicit
    SSPRK2 Program returns ``{}`` (nothing pending: every op it uses is served) -- an empty report is
    the fully-green report. Scheduling is classified from the live Program values rather than the
    inspection summary: the latter deliberately renders a Schedule as its type name and therefore
    cannot prove whether lowering needs the checkpointed hierarchy cache. NO mutation occurs.
    """
    if type(context) is not AMRProgramSupportContext:
        raise TypeError(
            "amr_program_op_support requires the resolved AMRProgramSupportContext; "
            "Program IR alone is insufficient")
    if not context.field_routes_validated:
        raise ValueError(
            "amr_program_op_support requires authenticated resolved field-provider routes")
    used_groups = _used_groups(program, context=context)
    return {name: _group_status(DEFERRED_GROUPS[name]) for name in sorted(used_groups)}


def _group_status(group: dict) -> str:
    """The status string of one group: ``"green"`` when it defers nothing, else ``"pending[:issue]"``."""
    if not group["header_methods"]:
        return "green"
    issue = group.get("issue")
    return "pending:%s" % issue if issue else "pending"


def _used_groups(program: Any, *, context: AMRProgramSupportContext) -> set:
    """The capability groups the ops of @p program map into.

    Walks ``program.ir_nodes()`` (each node's ``op``) and maps a used op to its group via the
    ``ir_ops`` membership in :data:`DEFERRED_GROUPS`. A ``rhs`` node carrying NAMED fluxes maps into
    the ``named_flux`` group (the named-flux -div path), a ``solve_fields`` node carrying a
    ``field`` attr into ``named_field_solve``, and a scheduled node into the exact
    ``schedule_control`` / ``schedule_cache`` group. Schedule classification consumes the original
    Program values because the inspection summary intentionally does not retain Schedule objects.
    Every mapping reads the IR only; it binds / dlopens nothing.
    """
    op_to_group = {}
    for name, group in DEFERRED_GROUPS.items():
        for op in group["ir_ops"]:
            op_to_group[op] = name
    used: set = set()
    nodes = _ir_nodes(program)
    for node in nodes:
        op = node.get("op")
        attrs = node.get("attrs") or {}
        if op in op_to_group:
            used.add(op_to_group[op])
        # A rhs with named fluxes (not the default flux) lowers to the deferred named-flux -div seam.
        if op == "rhs" and _has_named_fluxes(attrs):
            used.add("named_flux")
        # The canonical IR op is solve_fields; code generation alone lowers that operation to the
        # exact C++ AmrProgramContext::solve_fields_from_state_at seam.
        if op == "solve_fields" and attrs.get("field"):
            used.add("named_field_solve")
    for value in _scheduled_values(program, nodes):
        group = _schedule_group(value, context=context)
        if group is not None:
            used.add(group)
    return used


def _scheduled_values(program: Any, nodes: Any) -> list[Any]:
    """Return the live scheduled Program values, or refuse insufficient inspection evidence.

    :meth:`Program.ir_nodes` makes its attrs JSON-friendly and consequently reduces a Schedule to
    ``"Schedule"``.  The AMR capability gate must inspect the original value attrs to distinguish a
    cache-free ``schedule_decision(..., false)`` from a checkpointed cache dependency; treating that
    summary as either one would manufacture an unsupported verdict.
    """
    if not any(node["attrs"].get("schedule") is not None for node in nodes):
        return []
    try:
        from pops.codegen.program_lowerability import all_ops
        values = list(all_ops(program))
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "AMR Program schedule capability requires live Program value attrs"
        ) from exc
    scheduled = []
    for value in values:
        attrs = getattr(value, "attrs", None)
        if not isinstance(attrs, dict):
            raise TypeError("AMR Program value attrs must be a mapping")
        if attrs.get("schedule") is not None:
            scheduled.append(value)
    if not scheduled:
        raise TypeError(
            "AMR Program inspection reports a schedule without a live Program schedule value"
        )
    return scheduled


def _schedule_group(value: Any, *, context: AMRProgramSupportContext) -> str | None:
    """Classify one validated scheduled value by the exact lowering actions it reaches.

    ``schedule_decision(..., false)`` is an implemented AMR control seam.  Every emitted cache
    action remains unavailable until the cache participates in hierarchy checkpoint/regrid state.
    A field ``Hold`` also belongs to that unavailable provider contract even though its raw aux store
    is intentionally omitted by codegen.  Field ``Zero`` and ``AccumulateDt`` remain a distinct
    ProviderPack-freshness refusal, matching the emitter rather than pretending they are cacheable.
    """
    from pops.codegen.program_emit_kernels import _AUX_OUTPUT_OPS
    from pops.codegen.program_emit_schedule import _lower_schedule_ir
    from pops.time._schedule.api import ScheduleAction, ScheduleDueKind, ScheduleTimeline

    schedule = value.attrs["schedule"]
    lowering = _lower_schedule_ir(value, schedule)
    if lowering.domain.timeline is ScheduleTimeline.AMR_LEVEL:
        level = lowering.domain.level
        if level is None or level < 0 or level >= context.hierarchy_level_count:
            raise AMRProgramSupportError(
                "AMRLevel schedule on node %r selects level %r outside the resolved hierarchy "
                "[0, %d)" % (value.name, level, context.hierarchy_level_count)
            )
    # This is the one schedule shape for which the emitter returns before producing a due test or
    # schedule_decision seam. It must not advertise a control capability that generated code never
    # reaches. Always on ClockTick, Stage and AMRLevel still emits schedule_domain_occurs.
    if (
        lowering.due.kind is ScheduleDueKind.ALWAYS
        and lowering.domain.timeline is ScheduleTimeline.ACCEPTED_STEP
    ):
        return None
    actions = lowering.off.before_due + lowering.off.after_due + lowering.off_cadence
    is_aux = value.op in _AUX_OUTPUT_OPS
    if is_aux and any(
        action in {ScheduleAction.ZERO, ScheduleAction.ACCUMULATE_DT} for action in actions
    ):
        raise NotImplementedError(
            "scheduled field outputs must be governed by their typed ProviderPack freshness "
            "transaction; raw Program auxiliary caching/zeroing is not a supported route"
        )
    cache_actions = {
        ScheduleAction.EFFECTIVE_DT,
        ScheduleAction.STORE,
        ScheduleAction.ACCUMULATE_DT,
        ScheduleAction.RESTORE,
    }
    if any(action in cache_actions for action in actions):
        return "schedule_cache"
    return "schedule_control"


def _has_named_fluxes(attrs: dict) -> bool:
    """True when a ``rhs`` op's ``fluxes`` attr names non-default fluxes (the deferred named-flux path).

    Mirrors ``program_emit_kernels._named_fluxes``: ``None`` / ``["default"]`` is the default -div F
    path (``rhs_into``, served on AMR); any named flux routes into the deferred ``neg_div_flux_into``
    seam. The ir_nodes attr summary renders a list as a list, so this reads it directly.
    """
    fluxes = attrs.get("fluxes")
    if not fluxes or fluxes == ["default"]:
        return False
    return any(f != "default" for f in fluxes)


def _ir_nodes(program: Any) -> Any:
    """Return a validated copy of the Program's machine-readable IR nodes.

    A missing or malformed inspection surface is unknown capability evidence and therefore a hard
    error.  Treating it as an empty Program would incorrectly manufacture a green AMR verdict.
    """
    ir_nodes = getattr(program, "ir_nodes", None)
    if not callable(ir_nodes):
        raise TypeError("AMR Program capability query requires Program.ir_nodes()")
    # Deferred operations may be legal only inside a control/solver subregion.  The historical flat
    # report cannot expose them (notably rhs_jacvec is always inside matrix_free_operator.apply_block),
    # so accepting it here would manufacture a green AMR verdict for an operation reached at runtime.
    nodes = ir_nodes(recursive=True)
    if not isinstance(nodes, list):
        raise TypeError("Program.ir_nodes(recursive=True) must return a list")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError("Program.ir_nodes(recursive=True)[%d] must be a mapping" % index)
        op = node.get("op")
        attrs = node.get("attrs")
        if not isinstance(op, str) or not op:
            raise TypeError(
                "Program.ir_nodes(recursive=True)[%d].op must be a non-empty string" % index)
        if not isinstance(attrs, dict):
            raise TypeError(
                "Program.ir_nodes(recursive=True)[%d].attrs must be a mapping" % index)
    return list(nodes)


__all__ = ["AMRProgramSupportContext", "AMRProgramSupportError", "DEFERRED_GROUPS",
           "header_deferred_methods", "deferred_groups", "amr_program_op_support"]
