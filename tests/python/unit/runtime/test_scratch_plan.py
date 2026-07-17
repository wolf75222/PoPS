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
  - the PERSISTENT Krylov / multigrid solver buffers appear for a solve and are labelled conservative;
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


def _krylov(name="krylov_demo"):
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
    from pops.solvers.krylov import CG
    from pops.time import FailRun
    P.set_apply(A, _apply)
    P.solve(
        LinearProblem(A, buf), solver=CG(max_iter=10),
    ).consume(action=FailRun())
    P.commit(
        temporal.next,
        P.value("U1", U + P.dt * r, at=temporal.next.point),
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
    chk(plan.conservative is True, "the plan declares itself conservative (persistent buffers)")
    chk(any("conservative" in n.lower() for n in plan.notes),
        "a note states the persistent counts are conservative")


def test_persistent_krylov_buffers():
    """A solve_linear (Krylov) node contributes persistent Krylov work vectors."""
    print("== persistent Krylov work vectors for a solve_linear ==")
    P = _krylov()
    plan = build_scratch_plan(P)
    krylov = [p for p in plan.persistent if p["kind"] == "krylov"]
    chk(len(krylov) == 1, "one solve_linear -> one Krylov persistent entry")
    chk(krylov[0]["buffers"] >= 1, "the Krylov entry reports >= 1 work vector")
    chk("conservative" in krylov[0]["note"].lower(), "the Krylov count is labelled conservative")


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
