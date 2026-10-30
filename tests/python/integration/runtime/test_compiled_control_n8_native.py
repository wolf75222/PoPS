"""N8 lot 2, native: the compiled run of an elaborated Advance (spec 0.3 5.2, 14.1, CC14, PF1).

``Simulation(runtime).run(until, proposal=None, max_steps=None)`` is one call of
``System::run_advance``. On M01 the compiled run reaches the state of the b6 run loop bit for bit,
with its step count: on the final time of the case, on a frontier off the step grid, on the
frontier 0.25 that 80 steps of 0.05/16 reach only through the rounding that master's fixed step
absorbs (a cap alone takes 81), and over two runs whose fixed-step segment continues. No Python
runs during the native call (CF9). A budget ends a run with ``BudgetExhausted`` and the origin of
the budget, and a candidate that ``finite()`` rejects is attempted again with the same proposal,
the accepted state untouched, until ``Retry.max_attempts`` attempts are spent.
"""
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import pops
import examples.api_v02.models_M01_M14 as early
from pops.runtime._simulation import BudgetExhausted, RunResult, Simulation
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

STEP = 0.05/16  # the proposal of M01 at its 16 x 8 cells
REPO_ROOT = Path(__file__).resolve().parents[4]


def _b6(case, block, state, flow, dt, *, name):
    return early._ssprk2(case, block, state, lambda p, u, point, label: flow(u), dt, name=name)


def _retry(attempts):
    def route(case, block, state, flow, dt, *, name):
        return elaborate(Advance(early.shu_osher_ssprk2(flow=flow),
                                 retry=Retry(max_attempts=attempts)),
                         case=case, state=block[state], proposal=Proposal(dt), name=name)
    return route


def _artifact(world, monkeypatch, route, label):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(early, "_traced_ssprk2", route)
        spec = collective_call(world, early.build_m01)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="N8 lot 2 " + label,
                                         compile_artifact=pops.compile)
    return spec, artifact


def _bind(world, spec, artifact):
    return collective_call(world, lambda: pops.bind(artifact, initial_values={
        subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))


def _state(world, runtime):
    state = collective_call(world, lambda: runtime.state_global("fluid"))
    return None if state is None else np.asarray(state).copy()


def _same(world, left, right):
    with collective_check(world):
        if left is not None or right is not None:
            assert left is not None and right is not None
            assert left.dtype == right.dtype == np.float64
            # Bitwise, not within a tolerance: the same attempts of the same durations.
            assert left.tobytes() == right.tobytes()


def _b6_run(world, runtime, until):
    collective_call(world, lambda: pops.run(runtime, t_end=until, max_steps=10_000,
                                            console=False))
    return collective_call(world, runtime.time), collective_call(world, runtime.macro_step)


class _PythonCalls:
    """The native run of a Simulation, counting each Python frame that starts during it (CF9).

    A probe called inside the counted window proves that the counter records Python calls.
    """

    def __init__(self, run_native):
        self.run_native = run_native
        self.calls = []

    def __call__(self, *arguments):
        def profile(frame, event, _):
            if event == "call":
                self.calls.append(frame.f_code.co_name)
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            report = self.run_native(*arguments)
            _probe()
        finally:
            sys.setprofile(previous)
        return report


def _probe():
    return None


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, native_cxx, kokkos_root):
    """The traced and b6 artifacts of M01, compiled once for the module in its own cache."""
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        _, world = native_profile(dimension=2, scope="N8 compiled control")
        traced = _artifact(world, patch, None, "traced finite()")
        b6 = _artifact(world, patch, _b6, "b6 builder")
        yield world, traced, b6


@pytest.fixture(scope="module")
def retried(artifacts):
    """M01 elaborated with Retry(max_attempts=3), in the cache of the module."""
    with pytest.MonkeyPatch.context() as patch:
        return _artifact(artifacts[0], patch, _retry(3), "Retry(3)")


def test_the_compiled_run_lands_like_the_b6_run_loop(artifacts, record_property):
    world, (spec, traced), (b6_spec, b6) = artifacts
    assert spec.dt == b6_spec.dt == STEP
    manifest = traced.manifest().to_dict()["payload"]
    assert manifest["control"] == {
        "schema": "pops.advance-control@1", "accept": [{"assess": "finite", "origin": "default"}],
        "retry": {"max_attempts": 1, "max_transforms": 0}, "proposal": {"dt": STEP.hex()}}
    assert "control" not in b6.manifest().to_dict()["payload"]
    # The final time of the case, a frontier off the step grid, and 0.25, which a cap alone
    # reaches in 81 steps, the last one of about 1e-16.
    for until, steps in ((spec.t_end, 2), (0.0123, 4), (0.25, 80)):
        runtime = _bind(world, spec, traced)
        simulation = collective_call(world, lambda: Simulation(runtime))
        counter = _PythonCalls(simulation._run_native)
        simulation._run_native = counter
        result = collective_call(world, lambda: simulation.run(until))
        reference = _bind(world, b6_spec, b6)
        b6_time, b6_steps = _b6_run(world, reference, until)
        with collective_check(world):
            assert type(result) is RunResult and result.failure is None
            assert result.time == until == b6_time and runtime.time() == until
            assert result.steps == steps == b6_steps == runtime.macro_step()
            assert result.rejections == 0 and dict(result.outputs) == {}
            assert result.continuation.time == until
            assert result.continuation.macro_step == steps
            assert result.continuation.proposal.dt == STEP
            assert counter.calls == ["_probe"], counter.calls
            assert result.diagnostics["native"]["steps"] == steps
            assert result.diagnostics["run"]["until"] == until.hex()
        _same(world, _state(world, runtime), _state(world, reference))
        _same(world, collective_call(world, lambda: result.state["fluid"]),
              _state(world, runtime))
    record_property("artifact_identities", (traced.artifact_identity.hexdigest,
                                            b6.artifact_identity.hexdigest))


def test_a_second_run_continues_the_segment_of_the_first(artifacts):
    world, (spec, traced), (b6_spec, b6) = artifacts
    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))
    first = collective_call(world, lambda: simulation.run(0.1))
    second = collective_call(world, lambda: simulation.run(0.25))
    reference = _bind(world, b6_spec, b6)
    assert _b6_run(world, reference, 0.1) == (0.1, 32)
    assert _b6_run(world, reference, 0.25) == (0.25, 80)
    with collective_check(world):
        assert (first.time, first.steps, second.time, second.steps) == (0.1, 32, 0.25, 48)
        # The last step of the first run absorbed the rounding, so the segment restarts there.
        assert first.diagnostics["native"]["segment_origin"] == 0.1
        assert first.diagnostics["native"]["segment_steps"] == 0
        assert second.continuation.macro_step == 80
    _same(world, _state(world, runtime), _state(world, reference))
    # A later run supersedes the state of an earlier result.
    with collective_check(world):
        with pytest.raises(RuntimeError, match="superseded"):
            first.state["fluid"]
    # A run that does not move the frontier reaches it with no step.
    again = collective_call(world, lambda: simulation.run(0.25))
    with collective_check(world):
        assert (again.time, again.steps, again.failure) == (0.25, 0, None)


def test_max_steps_ends_the_run_with_the_budget_of_the_run(artifacts):
    world, (spec, traced), _ = artifacts
    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))

    def exhausted(**arguments):
        try:
            simulation.run(**arguments)
        except BudgetExhausted as error:
            return error
        raise AssertionError("the run did not spend its budget")
    error = collective_call(world, lambda: exhausted(until=0.25, max_steps=5))
    with collective_check(world):
        result = error.result
        assert type(result) is RunResult and result.failure.variant is BudgetExhausted
        assert result.failure.origin is None  # a budget of the run, not of an attempt (T39)
        assert result.failure.payload["budget"] == "max_steps"
        assert (result.steps, result.rejections) == (5, 0)
        assert result.time == runtime.time() < 0.25 and runtime.macro_step() == 5
        assert "max_steps=5" in str(error)
    zero = collective_call(world, lambda: exhausted(until=0.25, max_steps=0))
    with collective_check(world):
        assert zero.result.steps == 0 and zero.result.time == error.result.time
    # An exact proposal runs its nearest binary64 step, here the step of M01 (T5 of lot 1).
    finished = collective_call(world, lambda: simulation.run(
        0.25, proposal=Proposal(Fraction(1, 320))))
    with collective_check(world):
        assert finished.time == 0.25 and finished.steps == 75 and runtime.macro_step() == 80
        assert finished.diagnostics["run"]["proposal"] == STEP.hex()
    with collective_check(world):
        with pytest.raises(ValueError, match="max_steps"):
            simulation.run(0.5, max_steps=-1)
        with pytest.raises(TypeError, match="Proposal"):
            simulation.run(0.5, proposal=STEP)
        with pytest.raises(ValueError, match="finite"):
            simulation.run(math.inf)


@pytest.mark.parametrize("attempts", [1, 3])
def test_a_rejected_candidate_keeps_its_proposal_and_the_accepted_state(
        attempts, artifacts, request):
    world = artifacts[0]
    spec, artifact = artifacts[1] if attempts == 1 else request.getfixturevalue("retried")
    runtime = _bind(world, spec, artifact)
    simulation = collective_call(world, lambda: Simulation(runtime))
    assert simulation.control.max_attempts == attempts
    before = _state(world, runtime)
    # A step of 1e200 keeps both rates of SSPRK2 finite, which its physical core checks, and
    # overflows the candidate, U1 + dt*k1 of about 1e400: finite() rejects every attempt.
    huge = Proposal(1e200)

    def rejected():
        try:
            simulation.run(1e300, proposal=huge)
        except BudgetExhausted as error:
            return error
        raise AssertionError("finite() accepted an overflowing candidate")
    error = collective_call(world, rejected)
    with collective_check(world):
        result = error.result
        assert result.failure.origin == "assessment"
        assert result.failure.payload["budget"] == "max_attempts"
        assert result.failure.payload["attempts"] == attempts
        assert (result.steps, result.rejections) == (0, attempts)
        assert result.time == 0.0 and runtime.time() == 0.0 and runtime.macro_step() == 0
        assert "Advance assessment 'finite' rejected the candidate" in str(error)
        # T28, next=None: the continuation keeps the rejected proposal.
        assert result.continuation.proposal.dt == 1e200
    _same(world, before, _state(world, runtime))
    # A sane proposal then runs from the untouched state, as a fresh run does.
    resumed = collective_call(world, lambda: simulation.run(spec.t_end, proposal=Proposal(STEP)))
    fresh = _bind(world, spec, artifact)
    collective_call(world, lambda: Simulation(fresh).run(spec.t_end))
    with collective_check(world):
        assert resumed.steps == 2 and resumed.rejections == 0
    _same(world, _state(world, runtime), _state(world, fresh))


def test_a_failed_trial_leaves_the_run_with_its_native_error(artifacts):
    """A trial failure is not classified yet (T38, a later lot): it leaves the run as the native
    error that raised it, after its step restored the accepted state."""
    world, (spec, traced), _ = artifacts
    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))
    before = _state(world, runtime)
    # A step of 1e308 makes the second rate of SSPRK2 non-finite, which its physical core refuses.
    with collective_check(world):
        with pytest.raises(RuntimeError, match="non-finite all-component result"):
            simulation.run(1e308, proposal=Proposal(1e308))
        assert runtime.time() == 0.0 and runtime.macro_step() == 0
        assert simulation._continuation.proposal.dt == 1e308
    _same(world, before, _state(world, runtime))
    resumed = collective_call(world, lambda: simulation.run(spec.t_end, proposal=Proposal(STEP)))
    with collective_check(world):
        assert (resumed.time, resumed.steps, resumed.rejections) == (spec.t_end, 2, 0)


def test_the_compiled_run_refuses_what_it_cannot_run(artifacts):
    world, (spec, traced), (b6_spec, b6) = artifacts
    reference = _bind(world, b6_spec, b6)
    with collective_check(world):
        with pytest.raises(TypeError, match="no compiled control"):
            Simulation(reference)
    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))
    collective_call(world, lambda: simulation.run(spec.t_end))
    # The b6 run loop of pops.run does not continue a clock that a compiled run advanced.
    with collective_check(world):
        with pytest.raises(RuntimeError):
            pops.run(runtime, t_end=2*spec.t_end, max_steps=8, console=False)
        assert runtime.time() == spec.t_end and runtime.macro_step() == 2
    # Nor does a Simulation continue a clock that something else advanced.
    collective_call(world, lambda: runtime._executor._s.step(STEP))
    with collective_check(world):
        with pytest.raises(RuntimeError, match="advanced outside"):
            simulation.run(1.0)


def _master_fixed_dt(time, proposal, frontier, origin, steps):
    """The duration of master's FixedDtController (python/pops/runtime/_step_strategy.py@bb9)."""
    remaining = frontier - time
    if time + proposal == frontier:
        return proposal
    if origin + (steps + 1)*proposal == frontier:
        return remaining
    dt = min(proposal, remaining)
    if math.isclose(time + dt, frontier, rel_tol=0.0,
                    abs_tol=4.0*max(math.ulp(time + dt), math.ulp(frontier))):
        dt = remaining
    return dt


def test_the_attempt_duration_is_master_fixed_step_with_an_exact_landing(artifacts):
    world, (spec, traced), _ = artifacts
    runtime = _bind(world, spec, traced)
    duration = type(runtime._executor._s)._advance_attempt_duration
    # 79 accumulated steps of 0.05/16 stop short of 0.25 by a rounding: the segment absorbs it.
    time = 0.0
    for _ in range(79):
        time += STEP
    assert time + STEP < 0.25 and 80*STEP == 0.25
    assert duration(time, STEP, 0.25, 0.0, 79) == (0.25 - time, True, False)
    assert duration(0.0, 0.25, 1.0, 0.0, 0) == (0.25, False, True)
    assert duration(0.75, 0.25, 1.0, 0.0, 3) == (0.25, True, True)
    # No binary64 duration lands 2**-53 on 1 + 2**-52: the attempt stops at 1, the next lands.
    assert duration(2.**-53, 2.0, 1. + 2.**-52, 2.**-53, 0) == (1.0, False, False)
    assert duration(1.0, 2.0, 1. + 2.**-52, 1.0, 0) == (2.**-52, True, False)
    for arguments in [(0.0, 0.0, 1.0, 0.0, 0), (1.0, 0.1, 1.0, 0.0, 0), (0.0, 0.1, math.inf, 0., 0),
                      (0.0, 0.1, 1.0, 0.5, 0), (0.0, 0.1, 1.0, 0.0, -1)]:
        with pytest.raises(ValueError):
            duration(*arguments)
    # Random segments: the duration is master's wherever master publishes the frontier exactly,
    # and otherwise publishes it, or stops below it, by itself.
    generator = random.Random(20261030)
    for _ in range(4000):
        proposal = generator.choice((STEP, 0.1, 1/3, 2.**-generator.randint(1, 30)))
        origin = generator.choice((0.0, generator.uniform(0.0, 4.0)))
        steps = generator.randint(0, 400)
        time = origin
        for _ in range(steps):
            time += proposal
        frontier = generator.choice((
            origin + (steps + generator.randint(1, 3))*proposal,
            time + proposal*generator.choice((0.5, 1.0, 1.0 + 1e-15, 2.5)),
            math.nextafter(time + proposal, math.inf), math.nextafter(time + proposal, 0.0)))
        if not frontier > time:
            continue
        dt, lands, continuous = duration(time, proposal, frontier, origin, steps)
        master = _master_fixed_dt(time, proposal, frontier, origin, steps)
        assert dt > 0.0 and continuous == (dt == proposal)
        assert lands == (time + dt == frontier) and time + dt <= frontier
        keeps = master == proposal != frontier - time and time + proposal < frontier
        if keeps or time + master == frontier:
            assert (dt, lands) == (master, not keeps)
