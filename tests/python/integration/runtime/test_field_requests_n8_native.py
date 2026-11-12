"""N8 lot 5, native: field requests and Auxiliary method memory (spec 0.3 T6, T7, T27, T38).

The five profiles whose stages solve a field (M08, M09, M10, M11, M13) write their step as
``recorded_ssprk2``: each component requests its field with ``field.at(U)``, and the Candidate
returns the stage-1 state and observations as an Auxiliary memory, stored in History rings of depth
one after the assessment. Their b6 evaluator, kept verbatim, is selected by patching
``_traced_recorded_ssprk2`` (MG1). Both routes run through ``pops.run``; the b6 invariants of the
case pass on both, the step counts are equal, and the final state and the slot one of each
``stage1_*`` ring are equal bitwise. ``build_m09`` (the compatibility composition
``explicit-SSPRK2``) has no other native receipt.

Through ``Simulation.run``, M08's ``continuation.memory`` reads those rings; it is undefined
before the first accepted step (CN6) and superseded by a later run. With a lowered CG budget,
M08's stage-0 solve, whose dependency set excludes the proposal, is a fixed failure site: the run
raises ``BudgetExhausted`` tagged ``trial`` with the budget ``max_iterations`` after one attempt,
with no rejection counted (SV10; ``SolveFailed`` until N9 « La référence sous les solves de
relation » §3.6). As the budget grows, the failing site moves to the stage-1 solve, which reads the
proposal: a retryable site, whose rejections spend ``Retry.max_attempts`` and end in
``BudgetExhausted`` tagged ``trial``, caused by the ``BudgetExhausted`` (``max_iterations``) of the
last attempt (T38, design note O14). Both leave the accepted state and the macro step of the step.
"""
import numpy as np
import pytest

import pops
import examples.api_v02.models_M01_M14 as early
import examples.api_v02.models_M13 as late
from pops.runtime._simulation import BudgetExhausted, Simulation, SolveFailed
from pops.solvers import CG
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.memory import MemoryUndefined
from pops.time._program.traced_method import NotYetElaborated
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

# profile -> (module that calls _traced_recorded_ssprk2, builder, program name, block, rings)
PROFILES = {
    "M08": (early, early.build_m08, "M08_SSPRK2", "fluid", ("stage1_state", "stage1_potential")),
    "M09": (early, early.build_m09, "M09_SSPRK2", "fluid", ("stage1_state", "stage1_potential")),
    "M10": (early, early.build_m10, "M10_SSPRK2", "fluid", ("stage1_state", "stage1_potential")),
    "M11": (early, early.build_m11, "M11_SSPRK2", "fluid", ("stage1_state", "stage1_potential")),
    "M13": (late, late.build_m13, "M13_SSPRK2", "fluid",
            ("stage1_state", "stage1_potential", "stage1_photo")),
}
SITE = "shu_osher_ssprk2_recorded/flow#%d/live_Poisson/solve"


def _b6_route(case, block, state, flow, dt, *, name, memory, record, b6):
    return early._ssprk2(case, block, state, b6, dt, name=name)


def _array(world, read):
    value = collective_call(world, read)
    return None if value is None else np.asarray(value).copy()


def _run(build, block, rings, world, route):
    runtime, artifact, rows = execute_case(build, world, route)
    state = _array(world, lambda: runtime.state_global(block))
    memory = {ring: _array(world, lambda ring=ring: runtime.history_global(ring, 1))
              for ring in rings}
    steps = collective_call(world, runtime.macro_step)
    return state, memory, steps, artifact, rows


def _bitwise(left, right):
    if left is not None or right is not None:
        assert left is not None and right is not None
        assert left.dtype == right.dtype == np.float64
        # Bitwise, not within a tolerance: the same solves and stages in the same order.
        assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_field_profile_runs_bitwise_like_b6(profile, monkeypatch, isolated_native_cache,
                                            native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    module, build, name, block, rings = PROFILES[profile]
    _, world = native_profile(dimension=2, scope="N8 lot 5 field profile %s" % profile)
    traced = _run(build, block, rings, world, "N8 lot 5 %s traced method" % profile)
    monkeypatch.setattr(module, "_traced_recorded_ssprk2", _b6_route)
    b6 = _run(build, block, rings, world, "N8 lot 5 %s b6 builder" % profile)
    with collective_check(world):
        assert traced[3].program_name == b6[3].program_name == name
        assert all(row["passed"] for row in b6[4]) and all(row["passed"] for row in traced[4])
        assert traced[2] == b6[2] > 0
        _bitwise(b6[0], traced[0])
        for ring in rings:
            _bitwise(b6[1][ring], traced[1][ring])
    record_property("artifact_identities", (b6[3].artifact_identity.hexdigest,
                                            traced[3].artifact_identity.hexdigest))


# --- M08 through Simulation.run -----------------------------------------------------------------

def _route(attempts):
    def route(case, block, state, flow, dt, *, name, memory, record, b6):
        del b6
        return elaborate(Advance(early.recorded_ssprk2(memory, record)(flow=flow),
                                 retry=Retry(max_attempts=attempts)),
                         case=case, state=block[state], proposal=Proposal(dt), name=name)
    return route


def _simulation(world, monkeypatch, label, *, attempts=1, max_iter=None):
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_recorded_ssprk2", _route(attempts))
        if max_iter is not None:
            patch.setattr(early, "CG", lambda **options: CG(**dict(options, max_iter=max_iter)))
        spec = collective_call(world, early.build_m08)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="N8 lot 5 " + label,
                                         compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values={
        subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))
    return spec, runtime, Simulation(runtime)


def test_continuation_memory_reads_the_rings_of_the_accepted_step(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 lot 5 continuation memory")
    spec, runtime, simulation = _simulation(world, monkeypatch, "M08 continuation memory")
    empty = collective_call(world, lambda: simulation.run(0.0))
    memory = empty.continuation.memory
    assert sorted(memory) == ["stage1_potential", "stage1_state"]
    # Both refusals are local and come before any collective operation, on every rank.
    with collective_check(world):
        with pytest.raises(MemoryUndefined, match="before the first accepted step"):
            memory["stage1_state"]
        with pytest.raises(NotYetElaborated, match="G07"):
            runtime.checkpoint("unused")
    result = collective_call(world, lambda: simulation.run(spec.t_end))
    assert result.steps == 2 and result.rejections == 0
    for ring in ("stage1_state", "stage1_potential"):
        read = _array(world, lambda ring=ring: result.continuation.memory[ring])
        expected = _array(world, lambda ring=ring: runtime.history_global(ring, 1))
        with collective_check(world):
            _bitwise(expected, read)
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    with collective_check(world):
        assert metrics["passed"], metrics
    collective_call(world, lambda: simulation.run(spec.t_end + spec.dt))
    with collective_check(world):
        with pytest.raises(RuntimeError, match="superseded by a later run"):
            result.continuation.memory["stage1_state"]


def _failure(world, monkeypatch, max_iter, attempts):
    spec, runtime, simulation = _simulation(
        world, monkeypatch, "M08 CG(max_iter=%d) Retry(%d)" % (max_iter, attempts),
        attempts=attempts, max_iter=max_iter)
    start = _array(world, lambda: runtime.state_global("fluid"))

    def failed():
        try:
            simulation.run(spec.t_end)
        except (SolveFailed, BudgetExhausted) as error:
            return error
        raise AssertionError("the run did not fail")
    error = collective_call(world, failed)
    after = _array(world, lambda: runtime.state_global("fluid"))
    with collective_check(world):
        # The failure leaves the accepted state and the macro step of the step.
        _bitwise(start, after)
        assert runtime.macro_step() == 0
    return error


def test_failure_sites_are_fixed_at_stage_zero_and_retryable_at_stage_one(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 lot 5 trial failure sites")
    for attempts in (1, 3):
        error = _failure(world, monkeypatch, 1, attempts)
        failure = error.result.failure
        # SV10: a solve that reaches its iteration limit ends as BudgetExhausted, whose budget is
        # the solve's (N9 « La référence sous les solves de relation » §3.6, ruled Q3).
        assert type(error) is BudgetExhausted and failure.variant is BudgetExhausted
        assert failure.payload["budget"] == "max_iterations"
        assert failure.origin == "trial" and failure.payload["site"] == SITE % 0
        assert failure.payload["attempts"] == 1 and error.result.rejections == 0
        assert error.result.steps == 0
    # O14: no solve report is readable after a run, so the budget is swept until the stage-0
    # solve converges and the stage-1 solve, which reads the proposal, is the failing site.
    for max_iter in (2, 3):
        error = _failure(world, monkeypatch, max_iter, 3)
        if error.result.failure.payload["budget"] == "max_iterations":
            assert error.result.failure.payload["site"] == SITE % 0
            continue
        break
    else:
        pytest.fail("the stage-1 solve never failed alone for CG(max_iter) in 2, 3")
    for attempts, spent in ((3, error), (1, _failure(world, monkeypatch, max_iter, 1))):
        failure = spent.result.failure
        assert type(spent) is BudgetExhausted and failure.origin == "trial"
        assert failure.payload["attempts"] == attempts == spent.result.rejections
        cause = failure.payload["cause"]
        assert cause.variant is BudgetExhausted and cause.origin == "trial"
        assert cause.payload["budget"] == "max_iterations"
        assert cause.payload["site"] == SITE % 1
        assert SITE % 1 in failure.payload["rejection"] and cause.payload["reason"]
