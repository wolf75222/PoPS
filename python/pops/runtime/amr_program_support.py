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

    refined_hierarchy: bool
    shared_block_interfaces: bool
    field_routes_validated: bool

    def __post_init__(self) -> None:
        for name in (
            "refined_hierarchy", "shared_block_interfaces", "field_routes_validated",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("AMRProgramSupportContext.%s must be bool" % name)

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
        "op_source": "program_emit_kernels._AUX_OUTPUT_OPS['solve_fields_from_blocks']",
        "ir_ops": frozenset({"solve_fields_from_blocks"}),
        "header_methods": frozenset(),
    },
    "named_field_solve": {
        "issue": None,
        "op_source": "Program IR solve_fields -> program_emit_ops ctx.solve_fields_from_state",
        "ir_ops": frozenset({"solve_fields"}),
        "header_methods": frozenset(),
    },
    "unqualified_field_solve": {
        "issue": None,
        "op_source": "not representable in final Program IR (field identity is mandatory)",
        "ir_ops": frozenset(),
        "header_methods": frozenset({"solve_fields_from_state_default"}),
    },
    "unqualified_coupled_solve": {
        "issue": None,
        "op_source": "not representable in final Program IR (field identity is mandatory)",
        "ir_ops": frozenset(),
        "header_methods": frozenset({"solve_fields_from_blocks_default"}),
    },
    "refined_shared_block_interfaces": {
        "issue": None,
        "op_source": "captured rhs_group with shared block interfaces on a refined hierarchy",
        "ir_ops": frozenset(),
        "header_methods": frozenset({"refined_shared_block_interfaces"}),
    },
    "fine_level_field_perturbation": {
        "issue": None,
        "op_source": "field-provider perturbation inside an implicit solve",
        "ir_ops": frozenset(),
        "header_methods": frozenset({"solve_fields_from_state_at_fine_level"}),
    },
    "scheduler": {
        "issue": None,
        "op_source": "program_emit_schedule (held / scheduled cache_* seams)",
        "ir_ops": frozenset(),  # scheduling is an attr on an op node, not a distinct IR op
        "header_methods": frozenset({"cache_should_update", "cache_store_aux", "cache_restore_aux",
                                    "cache_store_scratch", "cache_restore_scratch",
                                    "cache_accumulate_dt", "cache_effective_dt", "scheduler_error"}),
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
    the fully-green report. NO refusal and NO mutation: this is a capability query, the route compiles
    and installs the Program regardless.
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
    ``field`` attr into ``named_field_solve``, and a node carrying a schedule attr into ``scheduler``
    -- the attr-borne derivations the codegen lowers into the same deferred seams. Every mapping
    reads the IR only; it binds / dlopens nothing.
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
        # C++ AmrProgramContext::solve_fields_from_state seam.
        if op == "solve_fields" and attrs.get("field"):
            used.add("named_field_solve")
        # A held / scheduled node lowers to the deferred scheduler cache seams.
        if attrs.get("schedule") is not None:
            used.add("scheduler")
        # A field-coupled finite-difference Jacobian re-solves the provider at a perturbed state.
        # AmrProgramContext serves this on the coarse level, but cannot do so on a fine level until
        # a composite stage solver exists.  This is conditional on resolved hierarchy evidence, not
        # a property that Program IR can decide alone.
        if op == "rhs_jacvec" and attrs.get("field_coupled") is True \
                and context.refined_hierarchy:
            used.add("fine_level_field_perturbation")
    if context.refined_hierarchy and context.shared_block_interfaces:
        used.add("refined_shared_block_interfaces")
    return used


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


__all__ = ["AMRProgramSupportContext", "DEFERRED_GROUPS", "header_deferred_methods",
           "deferred_groups", "amr_program_op_support"]
