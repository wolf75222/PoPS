"""N8 lot 6, native Dim2: M17's admissibility guards and T7 across attempts (spec 0.3 IR4, T7,
T38).

M17 writes its explicit Euler step between two pairs of admissibility guards,
``twice_admissible_euler`` of ``examples/api_v02/models_M15_orders.py``. Its b6 Program, kept
verbatim, is selected by patching ``traced_admissible_euler`` (MG1). Both routes run through
``pops.run``; the b6 invariants pass on both, the step counts are equal and the final states are
equal bitwise. M15, Dim1, and its failure sites are in ``test_candidate_guards_n8_dim1_native.py``.

T7 across the attempts of one step (design note, revision 2, section 5): under Retry(3), with a
CG budget at which M08's stage-1 solve fails alone, the stage-0 solve, a fixed site, runs at the
first attempt only. The run report counts the prepared solves that ran, every attempt included.
"""
import re

import numpy as np
import pytest

import pops
import pops.codegen.program_field_reuse as field_reuse
import examples.api_v02.models_M01_M14 as early
import examples.api_v02.models_M15_M18 as moments
import examples.api_v02.models_M15_orders as orders
from pops.runtime._simulation import BudgetExhausted, Simulation, SolveFailed
from pops.solvers import CG
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]



def _b6_route(case, block, state, dt, *, name, b6, **components):
    return b6()


def _array(world, read):
    value = collective_call(world, read)
    return None if value is None else np.asarray(value).copy()


def _run(build, world, route):
    runtime, artifact, rows = execute_case(build, world, route)
    state = _array(world, lambda: runtime.state_global("moments"))
    steps = collective_call(world, runtime.macro_step)
    return state, steps, artifact, rows


def _bitwise(left, right):
    if left is not None or right is not None:
        assert left is not None and right is not None
        assert left.dtype == right.dtype == np.float64
        # Bitwise, not within a tolerance: the same stages and guards in the same order.
        assert left.tobytes() == right.tobytes()


def _parity(build, name, dimension, profile, monkeypatch, record_property):
    _, world = native_profile(dimension=dimension, scope="N8 lot 6 guarded profile %s" % profile)
    traced = _run(build, world, "N8 lot 6 %s traced method" % profile)
    monkeypatch.setattr(orders, "traced_admissible_euler", _b6_route)
    b6 = _run(build, world, "N8 lot 6 %s b6 builder" % profile)
    with collective_check(world):
        assert traced[2].program_name == b6[2].program_name == name
        assert all(row["passed"] for row in b6[3]) and all(row["passed"] for row in traced[3])
        assert traced[1] == b6[1] > 0
        _bitwise(b6[0], traced[0])
    record_property("artifact_identities", (b6[2].artifact_identity.hexdigest,
                                            traced[2].artifact_identity.hexdigest))


def test_m17_runs_bitwise_like_b6(monkeypatch, isolated_native_cache, native_cxx, kokkos_root,
                                  record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _parity(moments.build_m17, "FanLi_raw_path_Euler", 2, "M17", monkeypatch, record_property)


# --- T7 across the attempts of one step (M08) ----------------------------------------------------

def _m08_route(attempts, substeps):
    def route(case, block, state, flow, dt, *, name, memory, record, b6):
        del b6
        program = elaborate(Advance(early.recorded_ssprk2(memory, record)(flow=flow),
                                    retry=Retry(max_attempts=attempts)),
                            case=case, state=block[state], proposal=Proposal(dt), name=name)
        if substeps != 1:
            program.program.cadence(substeps=substeps)
        return program
    return route


def _sabotaged_publish(publish):
    """A wrong reuse, for the sabotage proof of review A2: the attempt that keeps the solve of
    its fixed site zeroes the solution instead of reading it."""
    def sabotaged(value, var, lines, *, target, reused=None):
        if reused is not None:
            (solution,) = {match.group(1) for line in lines for match in [re.search(
                r"ctx\.solve_prepared_linear\(\*\w+, \*\w+, \*(\w+),", line)] if match}
            lines.append("if (%s) pops::PureFieldAlgebra::zero_valid(*%s);" % (reused, solution))
        return publish(value, var, lines, target=target, reused=reused)
    return sabotaged


def _m08(world, monkeypatch, attempts, max_iter, substeps=1, *, sabotage=False):
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_recorded_ssprk2", _m08_route(attempts, substeps))
        if max_iter is not None:
            patch.setattr(early, "CG", lambda **options: CG(**dict(options, max_iter=max_iter)))
        spec = collective_call(world, early.build_m08)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    with monkeypatch.context() as patch:
        if sabotage:
            patch.setattr(field_reuse, "publish_field_solve",
                          _sabotaged_publish(field_reuse.publish_field_solve))
        artifact = compile_profile_plan_once(
            world, plan, route="N8 lot 6 M08 CG(max_iter=%s) Retry(%d) substeps %d%s" % (
                max_iter, attempts, substeps, " sabotaged reuse" if sabotage else ""),
            compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values={
        subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))
    return spec, runtime, Simulation(runtime)


def _bits(native):
    return np.float64(native["last_prepared_solve_residual"]).tobytes()


def test_a_retry_keeps_the_solve_of_its_fixed_site(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    """M08 under Retry(3), its CG budget lowered until the stage-1 solve, which reads the
    proposal, fails alone: the stage-0 solve, a fixed site, runs at the first attempt only."""
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 lot 6 attempt reuse")

    def failed(spec, simulation):
        def run():
            try:
                simulation.run(spec.t_end)
            except (SolveFailed, BudgetExhausted) as error:
                return error
            raise AssertionError("the run did not fail")
        return collective_call(world, run)

    for max_iter in (2, 3):
        spec, _, simulation = _m08(world, monkeypatch, 3, max_iter)
        error = failed(spec, simulation)
        if type(error) is BudgetExhausted:
            break
    else:
        pytest.fail("the stage-1 solve never failed alone for CG(max_iter) in 2, 3")
    native = error.result.diagnostics["native"]
    with collective_check(world):
        assert error.result.rejections == 3
        # Attempt 1 solves both stages; attempts 2 and 3 solve stage 1 only.
        assert native["prepared_solves"] == 1 + 3
    # Review A2: no attempt of the step is accepted, so the reuse is checked by value. The
    # stage-1 solve reads the stage-0 solution through its right-hand side; the iteration count
    # and the final residual norm of its failed report are bitwise the same at the third attempt,
    # which reused that solution, as at the only attempt of Retry(1), which solved it.
    spec, _, simulation = _m08(world, monkeypatch, 1, max_iter)
    first = failed(spec, simulation)
    solved = first.result.diagnostics["native"]
    with collective_check(world):
        assert type(first) is BudgetExhausted and first.result.rejections == 1
        assert solved["prepared_solves"] == 2
        assert first.result.failure.payload["cause"].payload == \
            error.result.failure.payload["cause"].payload
        assert solved["last_prepared_solve_iterations"] == max_iter
        assert np.isfinite(solved["last_prepared_solve_residual"])
        assert solved["last_prepared_solve_residual"] > 0.0
        assert (native["last_prepared_solve_iterations"], _bits(native)) == (
            solved["last_prepared_solve_iterations"], _bits(solved))
    # The sabotage proof: a reuse that zeroes the kept solution changes that residual.
    spec, _, simulation = _m08(world, monkeypatch, 3, max_iter, sabotage=True)
    wrong = failed(spec, simulation)
    with collective_check(world):
        assert type(wrong) is BudgetExhausted and wrong.result.rejections == 3
        assert wrong.result.diagnostics["native"]["prepared_solves"] == 1 + 3
        assert _bits(wrong.result.diagnostics["native"]) != _bits(solved)
    # The same step accepted: every attempt is the first of its step, and solves both stages.
    spec, _, simulation = _m08(world, monkeypatch, 3, None)
    result = collective_call(world, lambda: simulation.run(spec.t_end))
    with collective_check(world):
        assert result.rejections == 0
        assert result.diagnostics["native"]["prepared_solves"] == 2 * result.steps


def test_a_cadence_with_sub_steps_reuses_no_solve(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    """Review A1: a cadence with sub-steps invokes the Program several times in one step, and the
    first invocation of a retry would find the slot of the last sub-step of the rejected attempt.
    Every invocation then opens its own epoch: Retry(3) solves three times what Retry(1) does."""
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 lot 6 attempt reuse sub-steps")

    def solves(attempts, max_iter):
        spec, _, simulation = _m08(world, monkeypatch, attempts, max_iter, substeps=2)

        def run():
            try:
                simulation.run(spec.t_end)
            except BudgetExhausted as error:
                return error.result
            return None
        result = collective_call(world, run)
        return None if result is None else (
            result.rejections, result.diagnostics["native"]["prepared_solves"])

    for max_iter in (2, 3):
        three = solves(3, max_iter)
        if three is not None:
            break
    else:
        pytest.fail("no attempt budget was spent for CG(max_iter) in 2, 3 with two sub-steps")
    one = solves(1, max_iter)
    with collective_check(world):
        assert three[0] == 3 and one[0] == 1
        assert three[1] == 3 * one[1]
