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
  3. explicit C++ capability deferrals, when a representable generated route reaches one, use
     ``deferred_op("<unambiguous-id>", ...)`` and are mirrored in :data:`DEFERRED_GROUPS`.
     Resolve-time refusals for routes that cannot be represented safely in an artifact are declared
     directly in the same table instead of inventing phantom C++ methods. Ordinary provider/runtime
     errors continue to use ``unavailable_`` and are not capability declarations.

Deliberately IMPORT-FREE of the pops package at module scope (stdlib + typing only): the
architecture gate loads it standalone, without the compiled ``_pops`` module. The codegen op-group
sets and ``Program.ir_nodes`` are reached LAZILY inside the functions that need them.
"""
from __future__ import annotations

from collections.abc import Mapping
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
    topology_rematerializer_validated: bool = False

    def __post_init__(self) -> None:
        if type(self.hierarchy_level_count) is not int:
            raise TypeError("AMRProgramSupportContext.hierarchy_level_count must be int")
        if self.hierarchy_level_count < 1:
            raise ValueError("AMRProgramSupportContext.hierarchy_level_count must be positive")
        for name in (
            "frozen_hierarchy", "shared_block_interfaces", "field_routes_validated",
            "topology_rematerializer_validated",
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
# Each group names (a) any representable AmrProgramContext C++ route that calls
# ``deferred_op("<name>", ...)`` and (b) the Python IR operations/attributes that route into the
# group. ``status`` records a pre-artifact refusal which has no representable C++ call site. An
# empty ``header_methods`` set therefore says only that there is no runtime deferral; it does not
# manufacture a green verdict for an upstream-refused route. Ordinary validation/runtime
# exceptions are deliberately outside this capability mirror. This is the SINGLE place the AMR
# support status is declared; :func:`deferred_groups` / :func:`amr_program_op_support` read it.
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
        # The Cartesian implementation is live. A resolved shared-interface provider is a
        # narrower unsupported envelope and is classified contextually by
        # ``_contextual_group_status`` rather than masquerading as a global throw stub here.
        "issue": None,
        "op_source": "program_emit_kernels._named_fluxes (rhs with named fluxes)",
        "ir_ops": frozenset({"neg_div_flux_into"}),
        "header_methods": frozenset(),
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
    "schedule_due": {
        "issue": None,
        "op_source": "program_emit_schedule Every/AtStart/Program When due primitives",
        "ir_ops": frozenset(),  # scheduling is an attr on an op node, not a distinct IR op
        "header_methods": frozenset(),
    },
    "schedule_domain": {
        "issue": None,
        "op_source": "program_emit_schedule Stage/ClockTick/AMRLevel domains",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_zero": {
        "issue": None,
        "op_source": "program_emit_schedule cache-free scratch Zero action",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_error": {
        "issue": None,
        "op_source": "program_emit_schedule cache-free Error action",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_field_skip": {
        "issue": None,
        "op_source": "program_emit_schedule initially-due field Skip with retained ProviderPack",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_field_hold": {
        "issue": None,
        "op_source": "program_emit_schedule field Hold with retained ProviderPack",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_field_skip_unprepared": {
        "issue": None,
        "status": "pending:initial_provider_pack",
        "op_source": "program_emit_schedule field Skip without a proven initial due occurrence",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_scratch_skip": {
        "issue": None,
        "status": "pending:prepared_scratch_state",
        "op_source": "program_emit_schedule scratch Skip requiring accepted transactional state",
        "ir_ops": frozenset(),
        "header_methods": frozenset(),
    },
    "schedule_cache": {
        "issue": None,
        "status": "pending:checkpointed_hierarchy_cache",
        "op_source": "program_emit_schedule scratch cache_* actions",
        "ir_ops": frozenset(),  # scheduling is an attr on an op node, not a distinct IR op
        "header_methods": frozenset(),
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

    A group may be pending either because a representable generated route has an explicit C++
    deferral or because resolve refuses an unsafe route before an artifact can represent it. Read-
    only: the result derives entirely from :data:`DEFERRED_GROUPS`.
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
    every group the Program touches (``"green"`` when the resolved AMR path serves it,
    ``"pending:<reason>"`` when a follow-up or a narrower execution envelope must land first). A
    group the Program does not use is OMITTED, so an all-explicit SSPRK2 Program returns ``{}``
    (nothing pending: every op it uses is served) -- an empty report is the fully-green report.
    Scheduling is classified from the live Program values rather than the inspection summary: the
    latter deliberately renders a Schedule as its type name and therefore cannot prove whether
    lowering needs the checkpointed hierarchy cache. NO mutation occurs.
    """
    if type(context) is not AMRProgramSupportContext:
        raise TypeError(
            "amr_program_op_support requires the resolved AMRProgramSupportContext; "
            "Program IR alone is insufficient")
    if not context.field_routes_validated:
        raise ValueError(
            "amr_program_op_support requires authenticated resolved field-provider routes")
    used_groups = _used_groups(program, context=context)
    return {
        name: _contextual_group_status(name, DEFERRED_GROUPS[name], context=context)
        for name in sorted(used_groups)
    }


def _group_status(group: dict) -> str:
    """Return one group's explicit upstream refusal or representable runtime deferral status."""
    declared = group.get("status")
    if declared is not None:
        if not isinstance(declared, str) or not declared.startswith("pending"):
            raise TypeError("AMR Program group status must be pending text")
        return declared
    if not group["header_methods"]:
        return "green"
    issue = group.get("issue")
    return "pending:%s" % issue if issue else "pending"


def _contextual_group_status(
    name: str,
    group: dict,
    *,
    context: AMRProgramSupportContext,
) -> str:
    """Narrow a globally live group to the exact resolved execution envelope.

    Named cell-centered fluxes are implemented on Cartesian AMR carriers, including ordinary
    multi-block carriers. The native context deliberately refuses an installed shared topological
    interface provider, however, so the resolve-time query must report that exact envelope as
    non-green before code generation rather than publishing a capability the runtime will reject.
    """
    status = _group_status(group)
    if status == "green" and name == "named_flux" and context.shared_block_interfaces:
        return "pending:shared_block_interfaces"
    if status == "green" and name in {"schedule_field_hold", "schedule_field_skip"} \
            and not context.frozen_hierarchy \
            and not context.topology_rematerializer_validated:
        return "pending:dynamic_hierarchy_provider_pack"
    return status


def _used_groups(program: Any, *, context: AMRProgramSupportContext) -> set:
    """The capability groups the ops of @p program map into.

    Walks ``program.ir_nodes()`` (each node's ``op``) and maps a used op to its group via the
    ``ir_ops`` membership in :data:`DEFERRED_GROUPS`. A ``rhs`` node carrying NAMED fluxes maps into
    the ``named_flux`` group (the named-flux -div path), a ``solve_fields`` node carrying a
    ``field`` attr into ``named_field_solve``, and a scheduled node into the exact
    exact schedule due/domain/policy groups. Schedule classification consumes the original Program
    values because the inspection summary intentionally does not retain Schedule objects.
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
        # A rhs with named fluxes (not the default flux) lowers to the named-flux -div group.
        if op == "rhs" and _has_named_fluxes(attrs):
            used.add("named_flux")
        # The canonical IR op is solve_fields; code generation alone lowers that operation to the
        # exact C++ AmrProgramContext::solve_fields_from_state_at seam.
        if op == "solve_fields" and attrs.get("field"):
            used.add("named_field_solve")
    for value, ancestry in _scheduled_values(program, nodes):
        used.update(_schedule_groups(value, ancestry=ancestry, context=context))
    return used


def _scheduled_values(program: Any, nodes: Any) -> list[tuple[Any, tuple[tuple[int, str], ...]]]:
    """Return live scheduled values with authenticated structured-control ancestry.

    :meth:`Program.ir_nodes` makes its attrs JSON-friendly and consequently reduces a Schedule to
    ``"Schedule"``.  The AMR capability gate must inspect the original value attrs to distinguish a
    cache-free ``schedule_decision(..., false)`` from a checkpointed cache dependency; treating that
    summary as either one would manufacture an unsupported verdict. Flattening recursive values is
    also insufficient for field ``Skip``: a node in a lazy branch or loop is not proven to execute on
    the first macro step, even when its AcceptedStep trigger itself is initially due.
    """
    if not any(node["attrs"].get("schedule") is not None for node in nodes):
        return []
    try:
        from pops.codegen.program_lowerability import all_ops_with_ancestry
        values = list(all_ops_with_ancestry(program))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "AMR Program schedule capability requires live Program values with authenticated "
            "structured ancestry"
        ) from exc
    scheduled = []
    for value, ancestry in values:
        attrs = getattr(value, "attrs", None)
        if not isinstance(attrs, Mapping):
            raise TypeError("AMR Program value attrs must be a mapping")
        if attrs.get("schedule") is not None:
            scheduled.append((value, ancestry))
    if not scheduled:
        raise TypeError(
            "AMR Program inspection reports a schedule without a live Program schedule value"
        )
    return scheduled


def _schedule_groups(
    value: Any,
    *,
    ancestry: tuple[tuple[int, str], ...],
    context: AMRProgramSupportContext,
) -> set[str]:
    """Classify one validated scheduled value into exact due/domain/policy facets.

    Cache-free cadence, qualified domains, and off-cadence actions are deliberately separate so a
    green due primitive cannot overclaim a pending persistence policy. Every scratch cache action
    remains unavailable until the cache participates in hierarchy checkpoint/regrid state. Field
    ``Hold`` retains its accepted ProviderPack and is a distinct capability. Scratch
    ``Skip`` is not a cache hit: it requires prepared accepted scratch state and rollback. Field
    ``Skip`` is honest only on a resolved frozen hierarchy where the ProviderPack remains retained,
    and only a root-region AcceptedStep Every/AtStart proves that the first invocation prepares that
    value. A due trigger nested in lazy structured control does not prove its owner executes.
    """
    from pops.codegen.program_emit_kernels import _AUX_OUTPUT_OPS
    from pops.codegen.program_emit_schedule import _lower_schedule_ir
    from pops.time._schedule.api import (
        ScheduleAction,
        ScheduleComment,
        ScheduleDueKind,
        ScheduleTimeline,
    )

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
        return set()
    policy = lowering.off
    actions = policy.before_due + policy.after_due + policy.off_cadence
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
    field_hold = is_aux and {ScheduleAction.STORE, ScheduleAction.RESTORE}.issubset(set(actions))
    if any(action in cache_actions for action in actions) and not field_hold:
        return {"schedule_cache"}

    groups: set[str] = set()
    if field_hold:
        groups.add("schedule_field_hold")
    if lowering.due.kind is not ScheduleDueKind.ALWAYS:
        groups.add("schedule_due")
    if lowering.domain.timeline is not ScheduleTimeline.ACCEPTED_STEP:
        groups.add("schedule_domain")
    if ScheduleAction.ZERO in actions:
        groups.add("schedule_zero")
    if ScheduleAction.ERROR in actions:
        groups.add("schedule_error")
    if lowering.off.comment is ScheduleComment.SKIP:
        if not is_aux:
            groups.add("schedule_scratch_skip")
        elif (
            not ancestry
            and lowering.domain.timeline is ScheduleTimeline.ACCEPTED_STEP
            and lowering.due.kind
            in {ScheduleDueKind.CACHE_PERIOD, ScheduleDueKind.MACRO_STEP_ZERO}
        ):
            groups.add("schedule_field_skip")
        else:
            groups.add("schedule_field_skip_unprepared")
    return groups


def _has_named_fluxes(attrs: Mapping[str, Any]) -> bool:
    """True when a ``rhs`` op's ``fluxes`` attr names non-default fluxes.

    Mirrors ``program_emit_kernels._named_fluxes``: ``None`` / ``["default"]`` is the default -div F
    path (``rhs_into``); any named flux routes into the generic named-flux capability group. The
    ir_nodes attr summary renders a list as a list, so this reads it directly.
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
        if not isinstance(node, Mapping):
            raise TypeError("Program.ir_nodes(recursive=True)[%d] must be a mapping" % index)
        op = node.get("op")
        attrs = node.get("attrs")
        if not isinstance(op, str) or not op:
            raise TypeError(
                "Program.ir_nodes(recursive=True)[%d].op must be a non-empty string" % index)
        if not isinstance(attrs, Mapping):
            raise TypeError(
                "Program.ir_nodes(recursive=True)[%d].attrs must be a mapping" % index)
    return list(nodes)


__all__ = ["AMRProgramSupportContext", "AMRProgramSupportError", "DEFERRED_GROUPS",
           "header_deferred_methods", "deferred_groups", "amr_program_op_support"]
