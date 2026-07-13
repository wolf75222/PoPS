"""AMR Program op-support capability query (ADC-634): which ops a Program uses run on AMR.

The clean AMR route (``pops.compile(problem, layout=<structured AMR descriptor>)`` with a whole-system time
``Program``) compiles + installs + runs for ANY Program: a body using a deferred op still
COMPILES against ``AmrProgramContext`` (the signatures match ``ProgramContext`` byte-for-byte) and
throws the honest ``AmrProgramContext`` backstop only when that op is reached at run. ADC-634 adds
NO compile-time gate and NO route refusal. This module is the read-only CAPABILITY QUERY that
drives the Spec 6 matrix and ``inspect``: it enumerates the ops a Program actually uses (from its
IR) and reports, per capability group, whether the AMR Program path serves it today (``"green"``)
or defers it to a follow-up issue (``"pending:ADC-6xx"``).

Three single sources of truth, none duplicated here:

  1. Op enumeration is the Program's own IR (``Program.ir_nodes()`` -- the ``_pops``-free op walk);
  2. the op -> capability-group map reuses the codegen's OWN op-group vocabulary
     (``program_emit_kernels._CONDENSED_OPS`` et al.), imported lazily -- not a hand list;
  3. the AMR support status derives from the ONE C++ source of truth,
     ``include/pops/runtime/program/amr_program_context.hpp``: every deferral there (a
     ``deferred_op(...)`` op, a ``history_deferred`` method, an inline-throw method) is mirrored in
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

from typing import Any

# --- Capability groups: the ONE mirror of the AmrProgramContext deferral surface ----------------
# Each group names (a) the AmrProgramContext C++ methods that FAIL LOUD for it -- the header-derived
# deferred identifiers the parity test locks against amr_program_context.hpp -- and (b) the Python
# IR op names that route into that group. ``issue`` is the follow-up that greens the group (None for
# a group with no scheduled implementation). A group whose ``header_methods`` is EMPTY is served
# today (green); a non-empty one is pending. The header_methods are: the string literals passed to
# ``deferred_op("<name>", ...)``, the method names that call ``history_deferred(...)``, and the
# method names whose body throws inline for a deferral (apply_projection / the named
# solve_fields_from_state / solve_fields_from_blocks / scheduler_error). This is the SINGLE place the
# AMR support status is declared; :func:`deferred_groups` / :func:`amr_program_op_support` read it.
#
# op_source names the codegen op-group set this group's ir_ops mirror (documentary: the parity of the
# ir_ops against those codegen sets is asserted by the support parity test, so the emit vocabulary
# stays the single source and this table cannot silently drift from it).
DEFERRED_GROUPS: dict = {
    "condensed": {
        # ADC-633 WIRED the condensed-implicit Program on the hierarchy and ADC-637 made the generic
        # condensed_* ops the sole route: per-level assembly runs through AmrProgramContext::grid_context /
        # assembly_target / assembly_source, and solve_linear_matfree dispatches flat->matrix-free BiCGStab
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
        "header_methods": frozenset({"apply_projection"}),
    },
    "coupled_solve": {
        "issue": None,
        "op_source": "program_emit_kernels._AUX_OUTPUT_OPS['solve_fields_from_blocks']",
        "ir_ops": frozenset({"solve_fields_from_blocks"}),
        "header_methods": frozenset({"solve_fields_from_blocks"}),
    },
    "named_field_solve": {
        "issue": None,
        "op_source": "program_emit_ops.solve_fields_from_state(field, ...)",
        "ir_ops": frozenset({"solve_fields_from_state"}),
        "header_methods": frozenset({"solve_fields_from_state"}),
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


def amr_program_op_support(program: Any) -> dict:
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
    used_groups = _used_groups(program)
    return {name: _group_status(DEFERRED_GROUPS[name]) for name in sorted(used_groups)}


def _group_status(group: dict) -> str:
    """The status string of one group: ``"green"`` when it defers nothing, else ``"pending[:issue]"``."""
    if not group["header_methods"]:
        return "green"
    issue = group.get("issue")
    return "pending:%s" % issue if issue else "pending"


def _used_groups(program: Any) -> set:
    """The capability groups the ops of @p program map into.

    Walks ``program.ir_nodes()`` (each node's ``op``) and maps a used op to its group via the
    ``ir_ops`` membership in :data:`DEFERRED_GROUPS`. A ``rhs`` node carrying NAMED fluxes maps into
    the ``named_flux`` group (the named-flux -div path), a ``solve_fields_from_state`` carrying a
    ``field`` attr into ``named_field_solve``, and a node carrying a schedule attr into ``scheduler``
    -- the attr-borne derivations the codegen lowers into the same deferred seams. Every mapping
    reads the IR only; it binds / dlopens nothing.
    """
    op_to_group = {}
    for name, group in DEFERRED_GROUPS.items():
        for op in group["ir_ops"]:
            op_to_group[op] = name
    used: set = set()
    for node in _ir_nodes(program):
        op = node.get("op")
        attrs = node.get("attrs") or {}
        if op in op_to_group:
            used.add(op_to_group[op])
        # A rhs with named fluxes (not the default flux) lowers to the deferred named-flux -div seam.
        if op == "rhs" and _has_named_fluxes(attrs):
            used.add("named_flux")
        # A per-stage named-field re-solve (solve_fields_from_state with a field name) -> named_field.
        if op == "solve_fields_from_state" and attrs.get("field"):
            used.add("named_field_solve")
        # A held / scheduled node lowers to the deferred scheduler cache seams.
        if attrs.get("schedule") is not None:
            used.add("scheduler")
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
    """The Program's IR nodes (``program.ir_nodes()``), or ``[]`` for an object that exposes none.

    Reads the machine-readable, ``_pops``-free op walk the authoring layer already provides; a handle
    that is not a Program (no ``ir_nodes``) yields no ops (an empty support report, never an error).
    """
    ir_nodes = getattr(program, "ir_nodes", None)
    return ir_nodes() if callable(ir_nodes) else []


__all__ = ["DEFERRED_GROUPS", "header_deferred_methods", "deferred_groups",
           "amr_program_op_support"]
