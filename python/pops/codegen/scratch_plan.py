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
  - the PERSISTENT solver buffers -- the Krylov work vectors a ``solve_linear`` needs across its
    dynamic iteration and the multigrid hierarchy a ``solve_fields`` elliptic solve carries. These
    live for the whole solve (not a step-body scratch), so they are reported separately. Prepared
    Krylov counts are exact from the typed footprint; topology-dependent multigrid storage is marked
    conservative item by item.

Nothing here binds, dlopens, allocates or reads a runtime array: the builder reads the Program IR
(the same SSA value list ``_ir_hash`` digests) and the carried model's component counts only. It
imports ``pops.time`` lazily (in-function) so the codegen layering stays acyclic and the module is
``numpy`` / ``_pops``-free at module scope (cf. tests/python/architecture/test_import_graph.py).
"""
from __future__ import annotations

import json
from typing import Any

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
      persistent: Krylov / multigrid storage that lives for a whole solve, reported separately from
        step-body scratch. Krylov items expose exact field-buffer, scalar-value and collective-payload
        counts; each item says whether its count is exact.
      conservative: True iff any reported persistent figure is topology-dependent. Step-body scratch
        reuse and prepared Krylov footprints are exact; the ``notes`` identify any remaining estimate.
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
                lines.append("    %-13s %s x%d  (%s)"
                             % (p["name"], p["kind"], p["buffers"], p["note"]))
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
    live ranges are provably disjoint); the persistent Krylov / multigrid buffers are reported
    separately with CONSERVATIVE counts (the exact figures are solver-dependent bind inputs). It
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

    # --- persistent solver buffers: Krylov work vectors + the multigrid hierarchy. These live for a
    # whole solve, not a step-body scratch, so they are reported separately. Prepared Krylov counts
    # come exactly from authenticated typed IR; only topology-dependent hierarchy entries are
    # conservative. ---
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
            "prepared Krylov workspace and affine-problem field counts are EXACT from the typed IR "
            "footprint; only a geometric multigrid hierarchy remains topology-dependent (%s)."
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


def _persistent_solver_buffers(program: Any) -> list:
    """The exact prepared Krylov fields plus topology-dependent multigrid hierarchies.

    A ``solve_linear`` node carries an exact typed prepared footprint. A ``solve_fields`` elliptic
    solve carries a topology-dependent geometric multigrid hierarchy (~4/3 of the fine grid). Both
    persist for the solve, NOT the step body, so they are reported here (not in the step-body scratch
    reuse). Only the multigrid hierarchy remains conservative."""
    persistent = []
    for v in getattr(program, "_values", []):
        if v.op in _KRYLOV_OPS:
            method = v.attrs.get("method", "krylov")
            footprint = validated_krylov_footprint(v.attrs, operator=v.inputs[0])
            restart = footprint["restart"]
            preconditioned = footprint["preconditioned"]
            workspace = {
                "richardson": 2,
                "cg": 4,
                "bicgstab": 9 if preconditioned else 7,
                # Identity GMRES applies Arnoldi directly to the raw operator result, so it does
                # not reserve the extra transformed-vector field.  A real preconditioner still
                # owns that persistent vector across every restart cycle.
                "gmres": restart + (4 if preconditioned else 3),
            }[method]
            preconditioner_buffers = 2 if preconditioned else 0
            workspace_scalar_values = (
                restart * (restart + 1) + 4 * restart + 1
                if method == "gmres" else 0
            )
            collective_values = restart + 1 if method == "gmres" else 0
            persistent.append({
                "kind": "krylov",
                "name": v.name,
                "buffers": workspace + 2 + preconditioner_buffers,
                "workspace_buffers": workspace,
                "prepared_problem_buffers": 2,
                "prepared_preconditioner_buffers": preconditioner_buffers,
                "workspace_scalar_values": workspace_scalar_values,
                "collective_values": collective_values,
                "exact": True,
                "footprint": dict(footprint),
                "note": (
                    "%s: %d workspace + 2 A(0)/zero + %d M(0)/zero fields; "
                    "%d scalar + %d collective values (exact)"
                    % (method, workspace, preconditioner_buffers,
                       workspace_scalar_values, collective_values)
                ),
            })
            if v.attrs.get("preconditioner") == "geometric_mg":
                persistent.append({
                    "kind": "multigrid_preconditioner", "name": v.name, "buffers": 1,
                    "exact": False, "note": _MULTIGRID_BUFFER_NOTE})
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
