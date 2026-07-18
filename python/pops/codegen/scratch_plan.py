"""pops.codegen.scratch_plan -- INERT scratch liveness plan of a time Program (Spec 5 sec.13.11.3).

The scratch-plan inspection surface (acceptance criterion #38, epic ADC-479): a value class
:class:`ScratchPlan` plus the pure builder :func:`build_scratch_plan` that runs a LIVENESS analysis
over a lowered ``pops.time.Program`` IR and reports, BEFORE any bind / run / allocation:

  - the per-category scratch-buffer counts the step body needs -- ``state`` scratch (the staged
    states a ``linear_combine`` / local solve produces), ``rhs`` scratch (the rate buffers a
    ``rhs`` / ``source`` / ``apply`` / ``coupled_rate`` fills) and ``scalar_field`` scratch (the
    single-component fields a ``solve_linear`` / ``cell_compare`` / ``where`` produces);
  - the REUSED buffers -- pairs of scratch nodes whose SSA live ranges are PROVABLY disjoint (the
    earlier one's last read strictly precedes the later one's definition), so the codegen MAY share
    one buffer between them. This is computed, not assumed: it reuses the same greedy live-range
    allocation as ``Program.buffer_reuse_report`` (a sound minimum-buffer estimate);
  - the REJECTED reuse -- scratch nodes that could NOT collapse onto an existing buffer, each with
    an inspectable REASON (their live ranges overlap a still-live occupant, or -- for an aux-reading
    rhs/source/apply straddling a field solve -- the buffer carries a value the next reader needs);
  - the PERSISTENT solver bundles -- solution storage, matrix-free apply scratch/frozen resources,
    prepared affine problem/preconditioner fields, Krylov work vectors and optional multigrid
    hierarchy. These live for the whole solve (not a step-body scratch), so they are reported
    separately. Python reports only the structural Krylov lower bound: the native method provider
    computes its exact workspace from the runtime vector distribution and metric before solve.
    Conditional boundary-JVP and topology-dependent multigrid storage are labelled rather than
    hidden.

Nothing here binds, dlopens, allocates or reads a runtime array: the builder reads the Program IR
(the same SSA value list ``_ir_hash`` digests) and the carried model's component counts only. It
imports ``pops.time`` lazily (in-function) so the codegen layering stays acyclic and the module is
``numpy`` / ``_pops``-free at module scope (cf. tests/python/architecture/test_import_graph.py).
"""
from __future__ import annotations

import json
from typing import Any

from pops.solvers._prepared_preconditioner_registry import (
    prepared_preconditioner_allocation_plan_from_identity,
)
from pops.codegen.krylov_contract import validated_krylov_footprint

# Scratch-allocating op families. A scratch buffer is owned by one flat node; we bucket each by the
# vtype of the buffer it stages so a caller sees the state / rhs / scalar-field split the spec asks
# for. These mirror Program._SCRATCH_OPS; the bucketing is by the produced vtype, not the op name.
_RHS_OPS = ("rhs", "source", "apply", "coupled_rate")
_STATE_SCRATCH_OPS = (
    "linear_combine", "solve_local_linear", "solve_local_nonlinear",
    "solve_coupled_implicit", "where")
_SCALAR_FIELD_OPS = ("solve_linear", "scalar_field", "cell_compare")
# A linear_source is a pure operator DECLARATION node (vtype 'operator', no allocated buffer): it
# appears in the Program's _SCRATCH_OPS for the liveness walk but stages no MultiFab, so it is its
# own family ('operator') and never inflates the state / rhs / scalar-field scratch counts.
_OPERATOR_DECL_OPS = ("linear_source",)

# Solve ops that own a PERSISTENT (whole-solve) buffer set, distinct from the step-body scratch.
_KRYLOV_OPS = ("solve_linear",)
_ELLIPTIC_OPS = ("solve_fields", "solve_fields_from_blocks")

_MULTIGRID_LEVELS_NOTE = "~4/3 of the fine grid (geometric V-cycle hierarchy)"
_MULTIGRID_BUFFER_NOTE = "geometric MG hierarchy " + _MULTIGRID_LEVELS_NOTE


class ScratchPlan:
    """An INERT scratch-liveness plan for a compiled time Program (Spec 5 sec.13.11.3, #38).

    A plain value describing the step body's scratch budget BEFORE any run: the per-category buffer
    counts, the provably-reusable buffers, the rejected reuse (with reasons) and the persistent
    solver buffers. It allocates nothing and reads no runtime array. ``str(plan)`` is a readable
    report; :meth:`to_dict` / :meth:`to_json` serialise it.

    Attributes:
      categories: ``{"state": n, "rhs": n, "scalar_field": n}`` -- the raw scratch-buffer count per
        family, BEFORE reuse (one buffer per scratch-allocating node).
      scratch_count: the total number of step-body scratch buffers before reuse.
      buffer_count: the MINIMUM number of distinct buffers after sound (disjoint-liverange) reuse.
      reused: a list of ``{"scratch", "op", "buffer", "shares_with"}`` -- each scratch that landed on
        a RECYCLED buffer, naming the earlier scratch(es) it shares that buffer with. PROVABLE: their
        live ranges do not overlap.
      rejected: a list of ``{"scratch", "op", "reason"}`` -- a scratch that could NOT reuse an
        existing buffer, with the inspectable reason (a still-live occupant, or an aux/field barrier).
      persistent: disjoint Krylov-operator, solve and multigrid allocation owners that live for a
        whole solve. Exact problem/apply sub-counts and the provider-owned Krylov workspace lower
        bound are reported separately; AMR owns one instance per materialized level.
      conservative: True iff any reported persistent figure is conditional, provider-owned, or
        topology-dependent. Step-body reuse and explicitly scoped fixed sub-counts remain exact.
      notes: inspectable assumptions -- exactly which figures are exact vs conservative.
    """

    def __init__(self, *, program_name: Any, categories: Any, scratch_count: Any, buffer_count: Any,
                 reused: Any, rejected: Any, persistent: Any, notes: Any, conservative: Any) -> None:
        self.program_name = program_name
        self.categories = dict(categories)
        self.scratch_count = int(scratch_count)
        self.buffer_count = int(buffer_count)
        self.reused = [dict(r) for r in reused]
        self.rejected = [dict(r) for r in rejected]
        self.persistent = [dict(p) for p in persistent]
        self.notes = list(notes)
        self.conservative = bool(conservative)

    @property
    def buffers_saved(self) -> Any:
        """How many step-body buffers the sound reuse eliminates (scratch_count - buffer_count)."""
        return self.scratch_count - self.buffer_count

    def to_dict(self) -> dict:
        """A plain-dict view of every field (JSON-ready)."""
        return {"program": self.program_name,
                "categories": dict(self.categories),
                "scratch_count": self.scratch_count,
                "buffer_count": self.buffer_count,
                "buffers_saved": self.buffers_saved,
                "reused": [dict(r) for r in self.reused],
                "rejected": [dict(r) for r in self.rejected],
                "persistent": [dict(p) for p in self.persistent],
                "conservative": self.conservative,
                "notes": list(self.notes)}

    def to_json(self, path: Any = None, *, indent: Any = 2) -> Any:
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self) -> str:
        lines = ["scratch plan for Program %r (liveness, %s)"
                 % (self.program_name or "problem",
                    "conservative on persistent buffers" if self.conservative else "exact")]
        lines.append("  scratch categories (before reuse):")
        headline = ("state", "rhs", "scalar_field")
        for name in headline:
            lines.append("    %-13s %d" % (name, self.categories.get(name, 0)))
        for name in sorted(k for k in self.categories if k not in headline):
            lines.append("    %-13s %d" % (name, self.categories[name]))
        lines.append("  step-body buffers: %d allocated, %d after sound reuse (%d saved)"
                     % (self.scratch_count, self.buffer_count, self.buffers_saved))
        if self.reused:
            lines.append("  reused buffers (live ranges proven disjoint):")
            for r in self.reused:
                lines.append("    %-13s (%s) -> buffer %d, shares with %s"
                             % (r["scratch"], r["op"], r["buffer"], ", ".join(r["shares_with"])))
        if self.rejected:
            lines.append("  rejected reuse:")
            for r in self.rejected:
                lines.append("    %-13s (%s): %s" % (r["scratch"], r["op"], r["reason"]))
        if self.persistent:
            qualification = "mixed exact/conservative" if self.conservative else "exact"
            lines.append("  persistent solver buffers (whole-solve, %s):" % qualification)
            for p in self.persistent:
                maximum = p.get("buffers_max", p["buffers"])
                if maximum is None:
                    count = ">=%d" % p["buffers"]
                else:
                    count = (str(p["buffers"]) if maximum == p["buffers"]
                             else "%d..%d" % (p["buffers"], maximum))
                lines.append("    %-13s %s x%s  (%s)"
                             % (p["name"], p["kind"], count, p["note"]))
        if self.notes:
            lines.append("  notes:")
            for note in self.notes:
                lines.append("    - %s" % note)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return ("ScratchPlan(scratch=%d, buffers=%d, reused=%d, rejected=%d, persistent=%d)"
                % (self.scratch_count, self.buffer_count, len(self.reused), len(self.rejected),
                   len(self.persistent)))


def _scratch_family(op: Any, vtype: Any) -> str:
    """The category a scratch-allocating node belongs to: ``state`` / ``rhs`` / ``scalar_field``.

    Bucket by the buffer family the op stages. The rhs family (rate buffers) and the state-scratch
    family (staged states) are split by op so the report mirrors the spec's state-scratch / rhs-
    scratch / scalar-field split; a ``linear_source`` is an operator DECLARATION (no buffer) and is
    bucketed separately so it inflates no real category. The produced ``vtype`` is the tiebreaker for
    any residual op."""
    if op in _OPERATOR_DECL_OPS:
        return "operator"
    if op in _RHS_OPS:
        return "rhs"
    if op in _STATE_SCRATCH_OPS:
        return "state"
    if op in _SCALAR_FIELD_OPS:
        return "scalar_field"
    # A residual scratch op (none today fall through, but be explicit): bucket by produced vtype.
    if vtype == "scalar_field":
        return "scalar_field"
    return "state"


def build_scratch_plan(program: Any, model: Any = None) -> ScratchPlan:
    """Build the :class:`ScratchPlan` of a lowered ``pops.time.Program`` (Spec 5 sec.13.11.3, #38).

    Runs the liveness / buffer-reuse analysis the Program already exposes
    (``Program.scratch_liveness`` / ``Program.buffer_reuse_report``) and turns it into an inspectable
    scratch plan. The reuse it reports is EXACT (two scratches share a buffer only when their SSA
    live ranges are provably disjoint); persistent Krylov bundles are split into exact/bounded typed
    categories, while topology-dependent multigrid storage remains conservative. It
    reads the IR only -- no bind, no dlopen, no allocation.

    @p program a lowered ``pops.time.Program`` (or a ``CompiledProblem`` carrying one).
    @p model optional physical model (unused for the structural plan; accepted for symmetry with the
       other inspection builders and a future per-block component-count breakdown). The IR node
       structure alone determines the scratch plan.
    """
    program = _resolve_program(program)
    if program is None:
        raise ValueError("build_scratch_plan: no Program to analyze (the handle carries none)")

    liveness = program.scratch_liveness()
    reuse = program.buffer_reuse_report()
    by_name = {r["name"]: r for r in liveness}
    assignment = reuse["assignment"]              # scratch name -> buffer index

    # --- per-category scratch counts (before reuse): one buffer per scratch-allocating node. The
    # 'operator' bucket holds linear_source declaration nodes (no MultiFab); it is reported only when
    # non-empty so the common state / rhs / scalar-field triple stays the headline. ---
    categories = {"state": 0, "rhs": 0, "scalar_field": 0, "operator": 0}
    op_of = {}
    for r in liveness:
        fam = _scratch_family(r["op"], _vtype_of(program, r["name"]))
        categories[fam] += 1
        op_of[r["name"]] = r["op"]
    if categories["operator"] == 0:
        del categories["operator"]

    # --- reused buffers: a scratch that landed on a buffer an EARLIER scratch already occupied. The
    # greedy allocator only recycles a buffer whose occupant is dead before the new def, so a shared
    # buffer is a PROOF the two ranges are disjoint. List, per recycled scratch, the earlier
    # scratch(es) on the same buffer with an earlier def (its predecessors on that slot). ---
    ranges_in_order = sorted(liveness, key=lambda r: r["def_index"])
    occupants = {}        # buffer index -> [scratch names, in def order]
    reused = []
    for r in ranges_in_order:
        name = r["name"]
        slot = assignment[name]
        prior = list(occupants.get(slot, []))
        if prior:
            reused.append({"scratch": name, "op": op_of[name], "buffer": slot,
                           "shares_with": prior})
        occupants.setdefault(slot, []).append(name)

    # --- rejected reuse: a scratch that could NOT recycle an earlier buffer. For each scratch placed
    # on a FRESH buffer (no prior occupant), name why no existing buffer was free at its def: every
    # already-allocated buffer's current occupant was still live (its range overlaps), so sharing
    # would alias two simultaneously-live values. This is the honest, inspectable reason; we also flag
    # the aux/field-barrier case (an rhs/source/apply reading the shared aux across a field solve is
    # not even CSE-equal, let alone buffer-shareable). ---
    rejected = _rejected_reuse(ranges_in_order, assignment, op_of, by_name, program)

    # --- persistent solver owners: disjoint operator, solve and multigrid resources. These live for
    # a whole solve, not a step-body scratch, so they are reported separately. ---
    persistent = _persistent_solver_buffers(program)

    notes = [
        "step-body scratch reuse is EXACT: two scratches share a buffer only when their SSA live "
        "ranges (def_index .. last_use_index) are provably disjoint (Program.buffer_reuse_report).",
        "the codegen MAY keep more buffers than buffer_count; this plan reports the sound MINIMUM, "
        "so it is a lower bound on the buffers the .so allocates, not a prediction of its choices.",
    ]
    conservative = any(not item.get("exact", False) for item in persistent)
    if persistent:
        notes.append(
            "prepared Krylov solution/problem/preconditioner structure is derived from typed IR; "
            "the native method provider computes its exact workspace from the runtime vector "
            "distribution and metric before solve. Conditional boundary-JVP buffers and geometric "
            "multigrid storage remain conservative (%s)."
            % _MULTIGRID_LEVELS_NOTE)

    return ScratchPlan(program_name=getattr(program, "name", None),
                       categories=categories, scratch_count=reuse["scratch_count"],
                       buffer_count=reuse["buffer_count"], reused=reused, rejected=rejected,
                       persistent=persistent, notes=notes, conservative=conservative)


def _rejected_reuse(ranges_in_order: Any, assignment: Any, op_of: Any, by_name: Any,
                    program: Any) -> list:
    """The scratches that could not reuse an earlier buffer, each with an inspectable reason.

    Walk the scratches in def order, tracking the live range currently on each buffer (the greedy
    allocator's state). A scratch on a FRESH buffer (not a recycled one) was rejected from reuse:
    name the still-live occupant(s) that blocked it, or -- for an rhs/source/apply that reads the
    shared aux across an intervening field solve -- the aux/field barrier. The very first scratch (no
    buffer exists yet) is not a rejection (there was nothing to reuse), so it is skipped."""
    field_barriers = _field_solve_indices(program)
    occupant_end = {}     # buffer index -> last_use_index of its current occupant
    rejected = []
    n_buffers = 0
    for r in ranges_in_order:
        name = r["name"]
        slot = assignment[name]
        if slot >= n_buffers:
            # A genuinely fresh buffer was allocated. If ANY earlier buffer existed, reuse was
            # considered and rejected (every occupant was still live); explain why.
            if n_buffers > 0:
                blockers = [b for b, end in occupant_end.items() if end >= r["def_index"]]
                reason = _reject_reason(name, r, op_of, field_barriers, blockers, by_name,
                                        assignment)
                rejected.append({"scratch": name, "op": op_of[name], "reason": reason})
            n_buffers = slot + 1
        occupant_end[slot] = r["last_use_index"]
    return rejected


def _reject_reason(name: Any, r: Any, op_of: Any, field_barriers: Any, blockers: Any, by_name: Any,
                   assignment: Any) -> str:
    """A human reason a scratch landed on a fresh buffer instead of reusing one.

    Primary reason: every already-allocated buffer's occupant was still live at this scratch's def
    (its range overlaps), so reuse would alias two simultaneously-live values. We also surface the
    aux/field-barrier note for an rhs/source/apply scratch whose live range straddles a field solve
    (it reads the freshly-solved aux, so it is not even value-equal to an earlier same-input rate --
    a documented CSE / reuse barrier in Program._PURE_OPS)."""
    aux_note = ""
    if op_of[name] in _RHS_OPS:
        for fidx in field_barriers:
            if r["def_index"] <= fidx <= r["last_use_index"]:
                aux_note = (" (an rhs/source/apply reads the shared aux a field solve writes, so its "
                            "buffer is not value-equal to an earlier same-input rate)")
                break
    if blockers:
        return ("all %d existing buffer(s) were still live at def index %d (their ranges overlap; "
                "sharing would alias two simultaneously-live scratches)%s"
                % (len(blockers), r["def_index"], aux_note))
    return ("a fresh buffer was required at def index %d%s" % (r["def_index"], aux_note))


def _field_solve_indices(program: Any) -> list:
    """Flat indices of the field-solve nodes (the aux/field barriers) in the step body."""
    out = []
    for i, v in enumerate(getattr(program, "_values", [])):
        if v.op in _ELLIPTIC_OPS:
            out.append(i)
    return out


def _operator_bundle_footprint(operator: Any) -> dict[str, int]:
    """Return the exact/bounded persistent fields owned by one matrix-free operator.

    Codegen owns one field for every authored scalar scratch, one result accumulator, four frozen
    coefficient fields per distinct tensor-coefficient bundle, and four fixed fields per
    ``rhs_jacvec``. A boundary-linearized jacvec conditionally owns two more fields whose presence
    is known only after native provider installation, hence the explicit min/max range.
    """
    attrs = getattr(operator, "attrs", None)
    attrs_get = getattr(attrs, "get", None)
    block = attrs_get("apply_block") if callable(attrs_get) else None
    if block is None:
        raise ValueError("solve_linear operator has no authenticated apply block")
    if not isinstance(block, (list, tuple)):
        raise ValueError("solve_linear operator apply block is not an authenticated value sequence")
    scalar_scratch = sum(getattr(value, "op", None) == "scalar_field" for value in block)
    coefficient_bundles = {
        getattr(value.inputs[2], "id", None)
        for value in block
        if getattr(value, "op", None) == "apply_laplacian_coeff"
    }
    if None in coefficient_bundles:
        raise ValueError("coefficiented matrix-free operator has no authenticated bundle identity")
    jacvec_count = sum(getattr(value, "op", None) == "rhs_jacvec" for value in block)
    fixed = scalar_scratch + 1 + 4 * len(coefficient_bundles) + 4 * jacvec_count
    return {
        "apply_scratch_buffers": scalar_scratch,
        "apply_accumulator_buffers": 1,
        "frozen_resource_buffers": 4 * len(coefficient_bundles),
        "jacvec_fixed_buffers": 4 * jacvec_count,
        "jacvec_conditional_buffers": 2 * jacvec_count,
        "operator_buffers_min": fixed,
        "operator_buffers_max": fixed + 2 * jacvec_count,
    }


def _persistent_program_values(program: Any):
    """Yield persistent-owner candidates in every structured region exactly once.

    Control-flow values are not members of ``Program._values``.  A solve nested in a while condition
    or body, range/subcycle body, or either branch arm nevertheless owns install-lifetime
    solution/problem/workspace storage.  Inspection therefore follows every authenticated structured
    block just as code emission does. SSA ids are global to one Program; deduplicating them prevents a
    shared/referenced node from being reported twice while preserving distinct authored solves that
    reuse one operator.
    """
    block_keys = ("cond_block", "body_block", "true_block", "false_block")
    seen_ids = set()

    def walk(values: Any):
        for value in values:
            value_id = getattr(value, "id", None)
            if not isinstance(value_id, int):
                raise ValueError("persistent scratch inspection found a value without an SSA id")
            if value_id in seen_ids:
                continue
            seen_ids.add(value_id)
            yield value
            attrs = getattr(value, "attrs", None)
            attrs_get = getattr(attrs, "get", None)
            if not callable(attrs_get):
                continue
            for key in block_keys:
                block = attrs_get(key)
                if block is None:
                    continue
                if not isinstance(block, (list, tuple)):
                    raise ValueError(
                        "%s has no authenticated %s for scratch inspection"
                        % (getattr(value, "op", "control-flow value"), key))
                yield from walk(block)

    yield from walk(getattr(program, "_values", ()))


def _persistent_solver_buffers(program: Any) -> list:
    """Allocation-owner records for prepared Krylov and multigrid resources.

    Each generated owner appears exactly once: matrix-free apply resources belong to their operator,
    live condensed coefficients to the coefficient node, and solution/problem/preconditioner/workspace
    fields to their solve. This avoids double-counting when two solves intentionally reuse one
    operator. A ``solve_fields`` elliptic hierarchy remains topology-dependent. All records are per
    materialized runtime level and rematerialize only after an authenticated topology change.
    """
    persistent = []
    values = list(_persistent_program_values(program))
    operator_bundles = {
        value.id: _operator_bundle_footprint(value)
        for value in values
        if value.op == "matrix_free_operator"
    }
    for v in values:
        if v.op == "matrix_free_operator":
            operator_bundle = _operator_bundle_footprint(v)
            conditional = operator_bundle["jacvec_conditional_buffers"]
            persistent.append({
                "kind": "matrix_free_operator",
                "name": v.name,
                "operator_id": v.id,
                "buffers": operator_bundle["operator_buffers_min"],
                "buffers_max": operator_bundle["operator_buffers_max"],
                **operator_bundle,
                "per_materialized_level": True,
                "exact": (operator_bundle["operator_buffers_min"]
                          == operator_bundle["operator_buffers_max"]),
                "note": (
                    "%d scalar apply scratch + 1 result accumulator + %d frozen coefficient + "
                    "%d fixed Jv fields%s"
                    % (operator_bundle["apply_scratch_buffers"],
                       operator_bundle["frozen_resource_buffers"],
                       operator_bundle["jacvec_fixed_buffers"],
                       " + 0..%d boundary-JVP fields" % conditional
                       if conditional else " (exact)")),
            })
        elif v.op == "condensed_coeffs":
            # The operator owns four frozen copies separately. Keeping this live source bundle as
            # its own owner stays exact when several operators share one coefficient assembly.
            persistent.append({
                "kind": "prepared_coefficient_bundle",
                "name": v.name,
                "buffers": 4,
                "buffers_max": 4,
                "per_materialized_level": True,
                "exact": True,
                "note": "four live condensed tensor-coefficient fields (exact)",
            })
        elif v.op in _KRYLOV_OPS:
            footprint = validated_krylov_footprint(v.attrs, operator=v.inputs[0])
            operator_bundle = operator_bundles.get(v.inputs[0].id)
            if operator_bundle is None:
                raise ValueError(
                    "solve_linear references an operator with no persistent allocation plan"
                )
            method = v.attrs["method_provider"]["provider_id"]
            preconditioner_allocation = (
                prepared_preconditioner_allocation_plan_from_identity(
                    v.attrs["preconditioner_provider"]
                )
            )
            preconditioner_buffers = preconditioner_allocation.prepared_buffers
            # One provider template is owned once by the operator record above. Every solve then
            # owns two genuinely private clones of that template: the problem's prepared A(0)
            # session and the bound workspace session used by the recurrence. A preconditioned
            # workspace additionally owns its persistent M(0) field beside the method's at-least-one
            # recurrence field.
            workspace_min = 1 + int(bool(footprint["preconditioned"]))
            operator_session_min = operator_bundle["operator_buffers_min"]
            operator_sessions_min = 2 * operator_session_min
            prepared_core_min = (
                workspace_min + 2 + preconditioner_buffers + operator_sessions_min
            )
            solve_buffers_min = 1 + prepared_core_min
            persistent.append({
                "kind": "krylov",
                "name": v.name,
                "buffers": solve_buffers_min,
                "buffers_max": None,
                "operator_id": v.inputs[0].id,
                "solution_buffers": 1,
                "workspace_buffers": None,
                "workspace_buffers_min": workspace_min,
                "prepared_problem_buffers": 2,
                "prepared_problem_operator_session_buffers_min": operator_session_min,
                "workspace_operator_session_buffers_min": operator_session_min,
                "operator_session_buffers_min": operator_sessions_min,
                "prepared_preconditioner_buffers": preconditioner_buffers,
                "prepared_core_buffers": None,
                "prepared_core_buffers_min": prepared_core_min,
                "workspace_scalar_values": None,
                "workspace_scaled_scalar_values": None,
                "collective_values": None,
                "reduction_value_capacity": None,
                "workspace_state_words": None,
                "native_workspace_authority": method,
                "per_materialized_level": True,
                "exact": False,
                "footprint": dict(footprint),
                "note": (
                    "%s solve owner: 1 solution + at least %d native workspace field(s) + "
                    "2 A(0)/zero + %d M(0)/zero fields + two private operator sessions "
                    "of at least %d fields each; exact method/session storage is computed "
                    "natively from the runtime vector distribution and metric before solve; "
                    "the separately reported provider template is owned once by operator id %d"
                    % (method, workspace_min, preconditioner_buffers, operator_session_min,
                       v.inputs[0].id)
                ),
            })
            for resource in preconditioner_allocation.scratch_resources:
                persistent.append({
                    "kind": resource.kind,
                    "name": v.name,
                    "buffers": resource.buffers,
                    "exact": resource.exact,
                    "note": resource.note,
                })
        elif v.op in _ELLIPTIC_OPS:
            persistent.append({"kind": "multigrid", "name": v.name, "buffers": 1,
                               "exact": False, "note": _MULTIGRID_BUFFER_NOTE})
    return persistent


def _vtype_of(program: Any, scratch_name: Any) -> Any:
    """The vtype of a scratch node by name (for the residual family bucketing). 'state' if absent."""
    for v in getattr(program, "_values", []):
        if v.name == scratch_name:
            return v.vtype
    return "state"


def _resolve_program(handle: Any) -> Any:
    """Return a ``pops.time.Program`` from a Program or a ``CompiledProblem`` carrying one, else None.

    Lazy ``pops.time`` import keeps the codegen layering acyclic and the module ``_pops``-free at
    module scope. A handle exposing a ``.program`` (a CompiledProblem) yields it; a Program (anything
    exposing ``scratch_liveness`` + ``buffer_reuse_report``) is returned as-is."""
    from pops.time import Program  # lazy: codegen may import time, but keep it off module scope
    if isinstance(handle, Program):
        return handle
    carried = getattr(handle, "program", None)
    if carried is not None:
        return carried
    # A duck-typed Program (the inspection helpers stay liberal): it must expose the liveness API.
    if hasattr(handle, "scratch_liveness") and hasattr(handle, "buffer_reuse_report"):
        return handle
    return None


__all__ = ["ScratchPlan", "build_scratch_plan"]
