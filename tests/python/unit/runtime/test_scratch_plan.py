#!/usr/bin/env python3
"""Scratch-plan inspection: liveness -> reuse / rejected / persistent (Spec 5 sec.13.11.3, #38).

INERT inspection (acceptance criterion #38, epic ADC-479): build a REAL ``pops.time.Program`` (an
SSPRK3-style multi-stage step with field solves + a Krylov solve) lowered in memory -- NO Kokkos
compile, NO .so on disk -- and assert that

  - ``build_scratch_plan(program)`` / ``compiled.scratch_plan()`` return a ``ScratchPlan`` listing the
    per-category scratch counts (state / rhs / scalar-field), inspectable BEFORE any bind / run;
  - the REUSED buffers are SOUND: a scratch is only marked reusable when its SSA live range is
    PROVABLY disjoint from the buffer's earlier occupant (the earlier last-use precedes its def);
  - the REJECTED reuse names an inspectable REASON (a still-live occupant, an aux/field barrier);
  - persistent Krylov bundles report solution, apply, prepared-problem and workspace fields
    separately, while multigrid remains explicitly topology-dependent;
  - ``to_dict`` / ``to_json`` / ``str`` / ``repr`` work and round-trip through JSON.

Pure-Python: the Program lowers without _pops; the plan reuses ``Program.scratch_liveness`` /
``buffer_reuse_report`` (the same liveness ADC-465 ships). Pytest + __main__ guard (CI runs
``python3 <file>``)."""
from tests.python.support.requirements import require_native_or_skip
import json
import os
import sys
import tempfile

try:
    import pops  # noqa: F401
    from pops.numerics.terms import DefaultSource, Flux
    from pops.codegen.scratch_plan import ScratchPlan, build_scratch_plan
    from pops.codegen.loader import CompiledModel, CompiledProblem
except Exception as exc:  # noqa: BLE001 -- pops unavailable in this interpreter
    require_native_or_skip('test_scratch_plan (pops unavailable: %s)' % exc)

from tests.python.unit.runtime._typed_program import (
    solve_field,
    typed_compiled_artifact,
    typed_program_state,
)


def _ssprk3(name="ssprk3"):
    """A real in-memory SSPRK3 Program: 3 stages, each a field solve + rhs + linear_combine commit.

    Each stage solves the elliptic field from its own stage state, builds the rate, and combines into
    the next stage; the final stage commits. This is the canonical multi-stage step the plan
    analyzes: the per-stage rate scratch lifetimes do NOT overlap (each is consumed by the very
    next combine), so they collapse to ONE reused buffer -- a provable reuse."""
    P, _, _, _, _, temporal = typed_program_state(name, block_name="plasma")
    dt = P.dt
    U = temporal.n
    from pops.time import StagePoint, TimePoint
    stage1 = StagePoint("ssprk3_stage_1", {"main": TimePoint(P.clock, 1)})
    stage2 = StagePoint("ssprk3_stage_2", {"main": TimePoint(P.clock, 1)})
    f0 = solve_field(P, U)
    r0 = P.rhs(state=U, fields=f0, terms=[Flux(), DefaultSource()])
    u1 = P.value("U1", U + dt * r0, at=stage1)
    f1 = solve_field(P, u1)
    r1 = P.rhs(state=u1, fields=f1, terms=[Flux(), DefaultSource()])
    u2 = P.value("U2", 0.75 * U + 0.25 * u1 + 0.25 * dt * r1, at=stage2)
    f2 = solve_field(P, u2)
    r2 = P.rhs(state=u2, fields=f2, terms=[Flux(), DefaultSource()])
    un = P.value(
        "Un",
        (1.0 / 3.0) * U + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * r2,
        at=temporal.next.point,
    )
    P.commit(temporal.next, un)
    return P


def _krylov(name="krylov_demo", *, preconditioner=None):
    """A Program with a typed matrix-free linear solve -- exercises the persistent path."""
    P, _, _, _, _, temporal = typed_program_state(name, block_name="plasma")
    U = temporal.n
    f = solve_field(P, U, name="phi")
    r = P.rhs(state=U, fields=f, terms=[Flux(), DefaultSource()])
    buf = P.scalar_field("buf")
    A = P.matrix_free_operator("op")

    def _apply(p, out, x):
        lap = p.scalar_field("lap")
        p.laplacian(lap, x)
        return -1.0 * lap

    from pops.linalg import LinearProblem
    from pops.solvers.krylov import GMRES
    from pops.time import FailRun
    P.set_apply(A, _apply)
    P.solve(
        LinearProblem(A, buf, nullspace=None),
        solver=GMRES(max_iter=10, restart=3, preconditioner=preconditioner),
    ).consume(action=FailRun())
    P.commit(
        temporal.next,
        P.value("U1", U + P.dt * r, at=temporal.next.point),
    )
    return P


def _nested_subcycled_krylov(name="nested_subcycled_krylov", *, preconditioner=None):
    """One solve two logical-clock body blocks below the Program's top-level value list."""
    P, _, _, _, _, temporal = typed_program_state(name, block_name="plasma")
    U = temporal.n
    A = P.matrix_free_operator("op", domain="state", range_="state", ncomp=1)

    def _apply(p, _out, value):
        lap = p.scalar_field("lap")
        p.laplacian(lap, value)
        return value - p.dt * lap

    from pops.linalg import LinearProblem
    from pops.solvers.krylov import GMRES
    from pops.time import FailRun, SampleAndHold
    from pops.time.points import Clock, TimePoint
    P.set_apply(A, _apply)
    fast = Clock("fast", owner=P.owner_path)
    micro = Clock("micro", owner=P.owner_path)
    fast_state = P.synchronize(
        U, at=TimePoint(fast), relation=SampleAndHold(), name="to_fast")

    def fast_tick(builder, value):
        micro_state = builder.synchronize(
            value, at=TimePoint(micro), relation=SampleAndHold(), name="to_micro")

        def micro_tick(inner, micro_value):
            return inner.solve(
                LinearProblem(A, micro_value, nullspace=None),
                solver=GMRES(max_iter=10, restart=3, preconditioner=preconditioner),
            ).consume(action=FailRun())

        advanced = builder.subcycle(
            micro_state, clock=micro, within=fast, count=2,
            body_fn=micro_tick, name="micro_ticks")
        return builder.synchronize(
            advanced, at=TimePoint(fast), relation=SampleAndHold(), name="to_fast_tick")

    advanced = P.subcycle(
        fast_state, clock=fast, within=P.clock, count=2,
        body_fn=fast_tick, name="fast_ticks")
    returned = P.synchronize(
        advanced, at=temporal.next.point, relation=SampleAndHold(), name="to_macro")
    P.commit(temporal.next, returned)
    return P


def _structured_region_krylov(kind, name="structured_region_krylov"):
    """Author prepared solves under while/range/branch regions, never the top-level SSA list."""
    P, _, _, _, _, temporal = typed_program_state(
        "%s_%s" % (name, kind), block_name="plasma")
    U = temporal.n
    A = P.matrix_free_operator("op", domain="state", range_="state", ncomp=1)

    def _apply(builder, _out, value):
        lap = builder.scalar_field("lap")
        builder.laplacian(lap, value)
        return value - builder.dt * lap

    from pops.linalg import LinearProblem
    from pops.solvers.krylov import GMRES
    from pops.time import FailRun
    P.set_apply(A, _apply)

    def solve(builder, value):
        return builder.solve(
            LinearProblem(A, value, nullspace=None),
            solver=GMRES(max_iter=10, restart=3),
        ).consume(action=FailRun())

    if kind == "while_cond":
        advanced = P.while_(
            U,
            lambda builder, value: builder.norm2(solve(builder, value)) > 0,
            lambda _builder, value: value,
        )
    elif kind == "while_body":
        advanced = P.while_(
            U,
            lambda builder, value: builder.norm2(value) > 0,
            solve,
        )
    elif kind == "range":
        advanced = P.range(U, 2, solve)
    elif kind == "branch":
        advanced = P.branch(
            P.norm2(U) > 0,
            lambda builder: solve(builder, U),
            lambda builder: solve(builder, U),
        )
    else:
        raise ValueError("unknown structured region %r" % kind)
    P.commit(
        temporal.next,
        P.value("advanced", advanced, at=temporal.next.point),
    )
    return P


def _model(*, n_vars=3, n_aux=1, aux_names=("B_z",)):
    """A real CompiledModel metadata carrier (no .so) -- the engine class, carrying only metadata."""
    cons = ["rho", "mx", "my", "E"][:n_vars]
    roles = ["Density", "MomentumX", "MomentumY", "Energy"][:n_vars]
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production",
        cons_names=cons, cons_roles=roles, prim_names=cons, n_vars=n_vars, gamma=1.4,
        n_aux=n_aux, params={}, caps={"cpu": True, "mpi": True},
        abi_key="SIG|c++|c++23", model_hash="modelhash", cxx="c++", std="c++23",
        aux_extra_names=list(aux_names))


def _compiled(program):
    """A SYNTHETIC CompiledProblem: a real lowered Program + a real CompiledModel, no compile."""
    model = _model()
    compiled = CompiledProblem(
        "/tmp/pops-cache/problem.so", program, model, "SIG|c++|c++23",
        "c++", "c++23", problem_hash="deadbeefcafe", cache_key="0badc0de")
    return typed_compiled_artifact(compiled, model)


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


# ---------------------------------------------------------------------------
# scratch categories
# ---------------------------------------------------------------------------

def test_scratch_plan_categories():
    """scratch_plan() lists the state / rhs / scalar-field scratch counts of the IR."""
    print("== scratch_plan() lists the scratch categories ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    chk(isinstance(plan, ScratchPlan), "scratch_plan() returns a ScratchPlan")
    # SSPRK3: 3 linear_combine states (U1/U2/Un) + 3 rhs rates, no scalar field.
    chk(plan.categories["state"] == 3, "3 state scratches (U1/U2/Un linear_combine)")
    chk(plan.categories["rhs"] == 3, "3 rhs scratches (the per-stage rates)")
    chk(plan.categories.get("scalar_field", 0) == 0, "no scalar-field scratch in a plain SSPRK3")
    # The category counts sum to the raw scratch count (one buffer per scratch node, before reuse).
    chk(sum(plan.categories.values()) == plan.scratch_count,
        "the per-category counts sum to scratch_count")


def test_scratch_plan_available_before_run():
    """The plan is built from the IR -- no bind, no .so, available BEFORE install."""
    print("== scratch_plan() is inert (no bind / no .so read) ==")
    cp = _compiled(_ssprk3())
    chk(not os.path.exists(cp.so_path), "the synthetic .so path does not exist")
    plan = cp.scratch_plan()  # must not raise despite the absent .so
    chk(isinstance(plan, ScratchPlan), "compiled.scratch_plan() works with no .so (pure IR read)")
    # The free builder and both delegators agree (same IR -> same plan).
    chk(build_scratch_plan(cp.program).to_dict() == cp.scratch_plan().to_dict(),
        "build_scratch_plan and compiled.scratch_plan() agree")
    chk(build_scratch_plan(cp.program).to_dict() == cp.scratch_plan().to_dict(),
        "build_scratch_plan(program) and compiled.scratch_plan() agree")


# ---------------------------------------------------------------------------
# sound reuse + rejected reuse
# ---------------------------------------------------------------------------

def test_reuse_is_sound():
    """A scratch is marked reusable ONLY when its live range is PROVABLY disjoint from the buffer's
    earlier occupant -- verified directly against the liveness ranges."""
    print("== reused buffers have provably-disjoint live ranges ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    chk(plan.buffers_saved > 0, "SSPRK3 reuses at least one buffer")
    chk(plan.buffer_count < plan.scratch_count, "buffer_count < scratch_count (reuse happened)")
    # The 3 per-stage rates are each consumed by the next combine -> disjoint -> one buffer. Node
    # ids also count the intervening field calls, so assert this from their typed ``rhs`` operation
    # rather than freezing incidental generated names.
    rhs_reuse = [r for r in plan.reused if r["op"] == "rhs"]
    chk(len(rhs_reuse) == 2 and len({r["buffer"] for r in rhs_reuse}) == 1,
        "the later per-stage rates reuse the first rate's buffer")
    # SOUNDNESS: for every reused entry, the sharer's live range must NOT overlap the predecessor's.
    live = {r["name"]: r for r in P.scratch_liveness()}
    for entry in plan.reused:
        sharer = live[entry["scratch"]]
        for prior_name in entry["shares_with"]:
            prior = live[prior_name]
            disjoint = prior["last_use_index"] < sharer["def_index"] \
                or sharer["last_use_index"] < prior["def_index"]
            chk(disjoint, "reuse of %s over %s is sound (disjoint live ranges)"
                % (entry["scratch"], prior_name))


def test_rejected_reuse_has_reason():
    """A scratch that could NOT reuse a buffer is listed with an inspectable reason (overlap)."""
    print("== rejected reuse carries an inspectable reason ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    # U1 / U2 are still live when the next stages' states are defined (a later combine reads them), so
    # they cannot share a buffer with the still-live earlier states -> rejected.
    rejected_names = {r["scratch"] for r in plan.rejected}
    chk(rejected_names, "at least one reuse is rejected (the long-lived stage states)")
    for r in plan.rejected:
        chk(isinstance(r["reason"], str) and r["reason"], "rejected entry %r has a reason string"
            % r["scratch"])
    # SOUNDNESS of the rejection: a rejected scratch's range really DOES overlap a buffer occupant
    # live at its def -- it is not a spurious rejection. U2 (def 6) overlaps U1 (live to 6).
    live = {r["name"]: r for r in P.scratch_liveness()}
    if "U2" in rejected_names:
        u1, u2 = live["U1"], live["U2"]
        chk(u1["last_use_index"] >= u2["def_index"],
            "U2's rejection is real: U1 is still live at U2's def")


def test_no_fabricated_reuse():
    """No scratch is BOTH reused and rejected; reuse never claims a still-live buffer."""
    print("== reuse / rejected are consistent (no fabricated reuse) ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    reused = {r["scratch"] for r in plan.reused}
    rejected = {r["scratch"] for r in plan.rejected}
    chk(not (reused & rejected), "a scratch is never both reused and rejected")
    # Every reused entry names a real, earlier scratch on the same buffer.
    live = {r["name"]: r for r in P.scratch_liveness()}
    for entry in plan.reused:
        chk(all(s in live for s in entry["shares_with"]), "shares_with names real scratches")
        chk(all(live[s]["def_index"] < live[entry["scratch"]]["def_index"]
                for s in entry["shares_with"]), "the shared buffer's occupants are EARLIER")


# ---------------------------------------------------------------------------
# persistent Krylov / multigrid solver buffers
# ---------------------------------------------------------------------------

def test_persistent_multigrid_buffers():
    """An elliptic solve_fields contributes a persistent multigrid buffer (whole-solve)."""
    print("== persistent multigrid buffers for field solves ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    mg = [p for p in plan.persistent if p["kind"] == "multigrid"]
    chk(len(mg) == 3, "3 field solves -> 3 multigrid persistent buffers")
    chk(plan.conservative is True,
        "the plan is conservative while topology-dependent multigrid buffers remain")
    chk(any("geometric multigrid" in n.lower() and "conservative" in n.lower()
            for n in plan.notes),
        "a note scopes the remaining uncertainty to the multigrid hierarchy")


def test_persistent_krylov_buffers():
    """A solve_linear reports the complete per-level generated bundle transparently."""
    print("== persistent Krylov work vectors for a solve_linear ==")
    P = _krylov()
    plan = build_scratch_plan(P)
    krylov = [p for p in plan.persistent if p["kind"] == "krylov"]
    operators = [p for p in plan.persistent if p["kind"] == "matrix_free_operator"]
    chk(len(krylov) == 1, "one solve_linear -> one Krylov persistent entry")
    chk(len(operators) == 1, "one reusable matrix-free operator -> one persistent entry")
    chk(krylov[0]["workspace_buffers"] is None
        and krylov[0]["workspace_buffers_min"] == 1,
        "Python exposes only the universal native-workspace lower bound")
    chk(krylov[0]["prepared_problem_buffers"] == 2,
        "the prepared affine problem owns zero and A(0)")
    chk(krylov[0]["prepared_preconditioner_buffers"] == 0,
        "the identity preconditioner owns no affine-linearization fields")
    chk(krylov[0]["workspace_scalar_values"] is None
        and krylov[0]["collective_values"] is None,
        "provider-owned scalar and collective storage is not guessed in Python")
    chk(krylov[0]["solution_buffers"] == 1,
        "the published persistent solution is counted separately")
    chk(operators[0]["apply_scratch_buffers"] == 1
        and operators[0]["apply_accumulator_buffers"] == 1,
        "the authored Laplacian scratch and generated apply accumulator are counted")
    chk(krylov[0]["prepared_core_buffers"] is None
        and krylov[0]["prepared_core_buffers_min"] == 3,
        "the prepared core reports its structural lower bound")
    chk(krylov[0]["buffers"] == 4 and krylov[0]["buffers_max"] is None
        and operators[0]["buffers"] == operators[0]["buffers_max"] == 2
        and krylov[0]["buffers"] + operators[0]["buffers"] == 6,
        "the solve and operator owners expose a six-field structural lower bound")
    chk(krylov[0]["exact"] is False,
        "the native method provider remains the sole exact workspace authority")
    chk(krylov[0]["per_materialized_level"] is True,
        "AMR multiplicity is explicit: one complete bundle per materialized level")
    chk(plan.conservative is True,
        "the plan exposes both provider-owned Krylov and topology-dependent MG uncertainty")

    from pops.solvers import preconditioners
    prepared = build_scratch_plan(
        _krylov("preconditioned_krylov", preconditioner=preconditioners.GeometricMG()))
    prepared_krylov = [p for p in prepared.persistent if p["kind"] == "krylov"][0]
    prepared_resources = [
        p for p in prepared.persistent if p["kind"] == "multigrid_preconditioner"
    ]
    prepared_operator = [
        p for p in prepared.persistent if p["kind"] == "matrix_free_operator"][0]
    chk(prepared_krylov["workspace_buffers"] is None
        and prepared_krylov["workspace_buffers_min"] == 1,
        "preconditioning does not make Python a workspace authority")
    chk(prepared_krylov["prepared_preconditioner_buffers"] == 2,
        "an affine prepared preconditioner owns zero and M_raw(0)")
    chk(prepared_krylov["workspace_scalar_values"] is None
        and prepared_krylov["collective_values"] is None,
        "native scalar and collective storage remains provider-owned")
    chk(prepared_krylov["prepared_core_buffers"] is None
        and prepared_krylov["prepared_core_buffers_min"] == 5,
        "the preconditioned core exposes its five-field structural lower bound")
    chk(prepared_krylov["buffers"] == 6
        and prepared_krylov["buffers_max"] is None
        and prepared_operator["buffers"] == prepared_operator["buffers_max"] == 2
        and prepared_krylov["buffers"] + prepared_operator["buffers"] == 8,
        "solve and shared operator owners expose an eight-field lower bound")
    chk(prepared_krylov["exact"] is False,
        "the exact workspace is computed by the native provider at materialization")
    chk(
        len(prepared_resources) == 1
        and prepared_resources[0]["buffers"] == 1
        and prepared_resources[0]["exact"] is False,
        "the provider contract contributes its one topology-dependent MG resource",
    )


def test_persistent_krylov_buffers_descend_nested_subcycles_once():
    """A solve nested under recursive subcycle bodies remains one persistent allocation owner."""
    print("== persistent Krylov owner inside nested subcycles ==")
    P = _nested_subcycled_krylov()
    chk(not any(value.op == "solve_linear" for value in P._values),
        "the regression solve is absent from the top-level Program value list")
    plan = build_scratch_plan(P)
    krylov = [p for p in plan.persistent if p["kind"] == "krylov"]
    operators = [p for p in plan.persistent if p["kind"] == "matrix_free_operator"]
    chk(len(krylov) == 1,
        "recursive subcycle traversal reports the nested solve exactly once")
    chk(len(operators) == 1,
        "the top-level matrix-free operator remains one shared allocation owner")
    chk(krylov[0]["workspace_buffers"] is None
        and krylov[0]["workspace_buffers_min"] == 1
        and krylov[0]["buffers"] == 4,
        "the nested solve keeps the same provider-owned workspace lower bound")
    chk(krylov[0]["operator_id"] == operators[0]["operator_id"],
        "the nested solve references the separately counted shared operator owner")


def test_nested_preconditioner_provider_contributes_its_native_header_once():
    """Provider headers are planned from recursive IR, not a top-level scheme-name branch."""
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.solvers import preconditioners

    source = emit_cpp_program(
        _nested_subcycled_krylov(
            "nested_preconditioned_krylov",
            preconditioner=preconditioners.GeometricMG(),
        )
    )
    chk(
        source.count("#include <pops/runtime/program/coeff_elliptic_ops.hpp>") == 1,
        "the nested provider's contract contributes one deduplicated native include",
    )


def test_persistent_krylov_buffers_descend_every_structured_region_once():
    """Prepared owners under cond/body/true/false blocks are all visible before compilation."""
    print("== persistent Krylov owners inside every structured control-flow region ==")
    for kind, expected_solves in (
            ("while_cond", 1), ("while_body", 1), ("range", 1), ("branch", 2)):
        P = _structured_region_krylov(kind)
        chk(not any(value.op == "solve_linear" for value in P._values),
            "%s regression solves remain outside the top-level SSA list" % kind)
        plan = build_scratch_plan(P)
        krylov = [p for p in plan.persistent if p["kind"] == "krylov"]
        operators = [p for p in plan.persistent if p["kind"] == "matrix_free_operator"]
        chk(len(krylov) == expected_solves,
            "%s reports each distinct nested solve exactly once" % kind)
        chk(len(operators) == 1,
            "%s keeps the shared top-level operator as one allocation owner" % kind)
        chk(all(entry["operator_id"] == operators[0]["operator_id"] for entry in krylov),
            "%s nested solves retain exact shared-operator provenance" % kind)


def test_scratch_plan_rejects_tampered_krylov_footprint():
    """Inspection authenticates the duplicate footprint against its typed operator."""
    print("== scratch_plan() rejects a tampered Krylov footprint ==")
    P = _krylov()
    solve = next(value for value in P._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    footprint = dict(attrs["krylov_footprint"])
    footprint["preconditioned"] = "false"
    attrs["krylov_footprint"] = footprint
    # Model stale/corrupted serialized IR at the inert inspection trust boundary.
    object.__setattr__(solve, "attrs", attrs)
    try:
        build_scratch_plan(P)
    except ValueError as exc:
        chk("preconditioned" in str(exc) and "boolean" in str(exc),
            "the shared footprint validator rejects string-to-bool coercion")
    else:
        chk(False, "a tampered Krylov footprint must not produce an exact scratch plan")

    P = _krylov("tampered_ghost_depth")
    solve = next(value for value in P._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    footprint = dict(attrs["krylov_footprint"])
    footprint["input_ghosts"] = 0
    attrs["krylov_footprint"] = footprint
    object.__setattr__(solve, "attrs", attrs)
    try:
        build_scratch_plan(P)
    except ValueError as exc:
        chk("input_ghosts" in str(exc) and "operator" in str(exc),
            "the duplicate ghost depth is bound to the typed operator stencil")
    else:
        chk(False, "an undersized Krylov halo must be rejected before code generation")

    P = _krylov("tampered_component_count")
    solve = next(value for value in P._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    footprint = dict(attrs["krylov_footprint"])
    attrs["ncomp"] = 2
    footprint["components"] = 2
    attrs["krylov_footprint"] = footprint
    object.__setattr__(solve, "attrs", attrs)
    try:
        build_scratch_plan(P)
    except ValueError as exc:
        chk("component count" in str(exc) and "operator" in str(exc),
            "paired solve/footprint tampering cannot override the operator component count")
    else:
        chk(False, "a forged Krylov component count must be rejected before code generation")


def test_no_persistent_without_solve():
    """A pure transport step (no field / Krylov solve) has no persistent buffers and is EXACT."""
    print("== a solve-free step has no persistent buffers (exact plan) ==")
    P, _, _, _, _, temporal = typed_program_state("transport_only", block_name="plasma")
    U = temporal.n
    r = P.rhs(state=U, terms=[Flux()])
    P.commit(
        temporal.next,
        P.value("U1", U + P.dt * r, at=temporal.next.point),
    )
    plan = build_scratch_plan(P)
    chk(plan.persistent == [], "no solve -> no persistent solver buffers")
    chk(plan.conservative is False, "a solve-free plan is EXACT, not conservative")


# ---------------------------------------------------------------------------
# serialisation + printing
# ---------------------------------------------------------------------------

def test_to_dict_and_json_roundtrip():
    """to_dict round-trips through JSON; to_json writes a valid file and returns the string."""
    print("== scratch_plan() serialisation (to_dict / to_json) ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    d = plan.to_dict()
    chk(set(d) >= {"categories", "scratch_count", "buffer_count", "reused", "rejected",
                   "persistent", "conservative", "notes"}, "to_dict carries every field")
    chk(json.loads(json.dumps(d)) == d, "to_dict is JSON round-trippable")
    chk(d["buffers_saved"] == plan.scratch_count - plan.buffer_count, "buffers_saved is reported")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scratch_plan.json")
        plan.to_json(path)
        with open(path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        chk(on_disk["scratch_count"] == plan.scratch_count, "to_json(path) wrote a valid file")
    chk(json.loads(plan.to_json())["program"] == "ssprk3", "to_json() returns the JSON string")


def test_str_and_repr():
    """str(plan) is a readable report; repr is a short summary."""
    print("== scratch_plan() str / repr ==")
    P = _ssprk3()
    plan = build_scratch_plan(P)
    text = str(plan)
    chk("scratch plan for Program 'ssprk3'" in text, "str() names the program")
    chk("scratch categories" in text and "reused buffers" in text, "str() shows categories + reuse")
    chk("persistent solver buffers" in text, "str() shows the persistent buffers")
    chk("ScratchPlan(scratch=" in repr(plan), "repr() is a short ScratchPlan summary")


def test_build_rejects_no_program():
    """build_scratch_plan with no Program raises a clear error (never fakes a plan)."""
    print("== build_scratch_plan rejects a handle with no Program ==")
    class _Empty:
        program = None
    try:
        build_scratch_plan(_Empty())
        chk(False, "build_scratch_plan should reject a programless handle")
    except ValueError:
        chk(True, "build_scratch_plan raises ValueError with no Program")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d test functions passed" % (len(fns) - failed, len(fns)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
