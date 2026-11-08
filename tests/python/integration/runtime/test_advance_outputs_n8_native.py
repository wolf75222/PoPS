"""N8 lot 3, native: the Outputs of a compiled run (spec 0.3 5.2, CC14, PF1, CF9).

``Output(value, when=None, at=None)`` is evaluated by ``System::run_advance`` itself: once at the
start of each run, then after each publication that it is due on, with the achieved time of that
publication. Each time of ``at`` (a tuple, or ``every_dt(dt)``) ahead of the run is a frontier
that the run lands on exactly; ``when=every(k)`` is due on the publications whose macro step is a
multiple of k. On M01 each published array is, bit for bit, the state that the compiled run of
the same Advance without Outputs reaches when it runs to the same frontiers one run at a time, and
each ``cell_sum`` is the native sum of b6 over that state. No Python runs during the native call
(CF9).
"""
import ctypes
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pops
import examples.api_v02.models_M01_M14 as early
from pops.runtime._simulation import BudgetExhausted, Publication, Simulation
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.outputs import Output, Product, cell_max, cell_sum, every, every_dt
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

STEP = 0.05/16  # the proposal of M01 at its 16 x 8 cells
REPO_ROOT = Path(__file__).resolve().parents[4]
# 0.1 minus two units in the last place: within four of 0.1, so it is 0.1 itself (master merges a
# consumer deadline into the end of its run).
NEAR_TENTH = math.nextafter(math.nextafter(0.1, 0.0), 0.0)


def _snapshot(B, interval):
    return Product(u=B.state, mass=cell_sum(B.state), peak=cell_max(B.state) * 2.0)


def _mass(B, interval):
    return Product(mass=cell_sum(B.state))


OUTPUTS = (
    Output(_snapshot, at=(0.05, 0.0123, 0.05, 0.2)),  # unsorted, 0.05 twice, 0.2 after 0.1
    Output(_mass, at=every_dt(0.01)),
    Output(_mass, when=every(16)),
    Output(_snapshot, at=(NEAR_TENTH, 0.1)),
)


def _with_outputs(case, block, state, flow, dt, *, name):
    return elaborate(Advance(early.shu_osher_ssprk2(flow=flow)), case=case, state=block[state],
                     proposal=Proposal(dt), name=name, outputs=OUTPUTS)


def _artifact(world, monkeypatch, route, label):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(early, "_traced_ssprk2", route)
        spec = collective_call(world, early.build_m01)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="N8 lot 3 " + label,
                                         compile_artifact=pops.compile)
    return spec, artifact


def _bind(world, spec, artifact):
    return collective_call(world, lambda: pops.bind(artifact, initial_values={
        subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))


def _state(world, runtime):
    state = collective_call(world, lambda: runtime.state_global("fluid"))
    return None if state is None else np.asarray(state).copy()


def _bitwise(left, right):
    assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


class _PythonCalls:
    """The native run of a Simulation, counting each Python frame that starts during it (CF9)."""

    def __init__(self, run_native):
        self.run_native = run_native
        self.calls = []

    def __call__(self, *arguments):
        import sys

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
    """M01 elaborated with OUTPUTS, and without Outputs (the reference), compiled once."""
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        _, world = native_profile(dimension=2, scope="N8 advance outputs")
        outputs = _artifact(world, patch, _with_outputs, "outputs")
        plain = _artifact(world, patch, None, "no outputs")
        yield world, outputs, plain


def _reference(world, spec, artifact, frontiers):
    """The state, the macro step and the native sum of component 0 after compiled runs of the
    Advance without Outputs to each frontier in turn, from the initial state."""
    runtime = _bind(world, spec, artifact)
    simulation = collective_call(world, lambda: Simulation(runtime))
    native = runtime._executor._s
    states = {0.0: (_state(world, runtime), 0, native.reduce_component("fluid", "sum", 0))}
    for frontier in frontiers:
        collective_call(world, lambda: simulation.run(frontier))
        states[frontier] = (_state(world, runtime), runtime.macro_step(),
                            native.reduce_component("fluid", "sum", 0))
    return states


def test_outputs_land_on_their_times_and_publish_the_accepted_state(artifacts):
    world, (spec, traced), (plain_spec, plain) = artifacts
    manifest = traced.manifest().to_dict()["payload"]["control"]
    assert manifest["schema"] == "pops.advance-control@2"
    assert [row["at"] for row in manifest["outputs"]] == [
        {"times": [0.0123.hex(), 0.05.hex(), 0.2.hex()]}, {"every_dt": 0.01.hex()}, None,
        {"times": [NEAR_TENTH.hex(), 0.1.hex()]}]
    assert manifest["outputs"][2]["when"] == {"every": 16}
    assert manifest["outputs"][0]["record"] == [
        {"name": "u", "array": "fluid"}, {"name": "mass", "scalar": 0},
        {"name": "peak", "scalar": 1}]
    assert traced.artifact_identity.hexdigest != plain.artifact_identity.hexdigest

    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))
    counter = _PythonCalls(simulation._run_native)
    simulation._run_native = counter
    result = collective_call(world, lambda: simulation.run(0.1))
    lattice = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    frontiers = sorted({0.0123, 0.05, *lattice})
    reference = _reference(world, plain_spec, plain, frontiers)
    with collective_check(world):
        assert counter.calls == ["_probe"], counter.calls
        assert result.failure is None and result.time == 0.1 == runtime.time()
        assert result.steps == reference[0.1][1] == runtime.macro_step()
        snapshots, tenths, cadence, near = (result.outputs[output] for output in OUTPUTS)
        assert all(type(row) is Publication for row in snapshots + tenths + cadence + near)
        # Each Output once at the start of the run, then at each of its times, landed exactly;
        # 0.2 lies after until and stays ahead.
        assert [row.time for row in snapshots] == [0.0, 0.0123, 0.05]
        assert [row.time for row in tenths] == [0.0, *lattice]
        assert [row.time for row in near] == [0.0, 0.1]  # one evaluation for both times
        # every(16): the start, then the publications at macro steps 16 and 32.
        assert [row.macro_step for row in cadence] == [0, 16, 32]
        assert result.diagnostics["outputs"] == [3, 11, 3, 2]
        for row in snapshots + tenths + near:
            state, macro_step, total = reference[row.time]
            assert row.macro_step == macro_step
            assert row["mass"] == total  # the generated sum is b6's native sum, bit for bit
            if "u" in row.values:
                _bitwise(row["u"], state)
                assert row["peak"] == 2.0 * float(np.max(state[0]))
                assert math.isclose(row["mass"], float(np.sum(state[0])), rel_tol=1e-12)
        for row in cadence:
            # M01 is periodic advection: each sum is the initial mass up to roundoff.
            assert math.isclose(row["mass"], reference[0.0][2], rel_tol=1e-12)
    # The run with Outputs takes the same steps as the runs to each frontier: its state is theirs.
    _bitwise(_state(world, runtime), reference[0.1][0])


def test_every_k_counts_the_accepted_steps_across_runs(artifacts):
    world, (spec, traced), _ = artifacts
    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))
    first = collective_call(world, lambda: simulation.run(0.04))
    second = collective_call(world, lambda: simulation.run(0.1))
    with collective_check(world):
        cadence = OUTPUTS[2]
        assert [row.macro_step for row in first.outputs[cadence]] == [0, 16]
        # A run evaluates its Outputs at its start; every(16) continues from the macro step.
        start = first.continuation.macro_step
        assert [row.macro_step for row in second.outputs[cadence]] == [start, 16 * 2]
        assert [row.time for row in second.outputs[OUTPUTS[1]]] == [
            0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        assert [row.time for row in second.outputs[OUTPUTS[0]]] == [0.04, 0.05]
        assert second.time == 0.1


def test_a_time_after_until_stays_ahead_and_a_budget_keeps_the_publications(artifacts):
    world, (spec, traced), _ = artifacts
    runtime = _bind(world, spec, traced)
    simulation = collective_call(world, lambda: Simulation(runtime))

    def exhausted():
        try:
            simulation.run(0.25, max_steps=20)
        except BudgetExhausted as error:
            return error
        raise AssertionError("the run did not spend its budget")
    error = collective_call(world, exhausted)
    lattice = (0.01, 0.02, 0.03, 0.04, 0.05)
    with collective_check(world):
        result = error.result
        stopped = result.time
        assert result.steps == 20 and 0.0123 < stopped < 0.1
        assert [row.time for row in result.outputs[OUTPUTS[0]]] == [
            0.0, *(time for time in (0.0123, 0.05) if time <= stopped)]
        assert [row.time for row in result.outputs[OUTPUTS[1]]] == [
            0.0, *(time for time in lattice if time <= stopped)]
    finished = collective_call(world, lambda: simulation.run(0.25))
    with collective_check(world):
        # The next run starts where the budget stopped: its times ahead are those after it.
        assert [row.time for row in finished.outputs[OUTPUTS[0]]] == [
            stopped, *(time for time in (0.0123, 0.05, 0.2) if time > stopped)]
        # Before until=0.25, the time two units below 0.1 is a frontier of its own, as master
        # keeps a consumer deadline that is not within four units of the end of its run.
        assert [row.time for row in finished.outputs[OUTPUTS[3]]] == [stopped, NEAR_TENTH, 0.1]
        assert finished.time == 0.25 and finished.failure is None


def test_a_run_with_outputs_refuses_a_schedule_the_native_run_cannot_take(artifacts):
    world, (spec, traced), _ = artifacts
    runtime = _bind(world, spec, traced)
    native = runtime._executor._s
    with collective_check(world):
        # One schedule per Output of the installed Advance.
        with pytest.raises(ValueError, match="one schedule per Output"):
            native._run_advance(0.1, STEP, 10, [])
        with pytest.raises(ValueError, match="strictly increasing"):
            native._run_advance(0.1, STEP, 10, [(0, [0.05, 0.02], []), (0, [], []),
                                                (0, [], []), (0, [], [])])
        assert runtime.time() == 0.0 and runtime.macro_step() == 0


SCHEDULE_SHIM = r"""
#include <pops/runtime/program/advance_run.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace program = pops::runtime::program;

// One Output with its step cadence and its times; returns whether it is due after the
// publication at (time, macro_step) and moves *cursor as run_advance does.
extern "C" int due(double time, std::int64_t macro_step, std::int64_t every_steps,
                   const double* at, std::size_t count, std::size_t* cursor) {
  std::vector<program::AdvanceOutputSchedule> schedule(1);
  schedule[0].every_steps = every_steps;
  schedule[0].at.assign(at, at + count);
  std::vector<std::size_t> cursors{*cursor};
  const bool result = program::advance_output_due(0, time, macro_step, schedule, cursors);
  *cursor = cursors[0];
  return result ? 1 : 0;
}

extern "C" double next_frontier(double time, double until, const double* at, std::size_t count,
                                std::size_t cursor) {
  std::vector<program::AdvanceOutputSchedule> schedule(1);
  schedule[0].at.assign(at, at + count);
  return program::advance_next_frontier(time, until, schedule, {cursor});
}
"""


def test_a_publication_that_passes_several_times_evaluates_its_output_once(native_cxx, tmp_path):
    """CC14: the rule that ``System::run_advance`` applies after each publication, compiled from
    its header. Every method of this lot lands on its frontiers, so a run never passes a time of
    ``at``; a later method whose achieved time is not ``B.time + p.dt`` (T26) can, and its Output
    is then evaluated once for that publication, with every time it passed behind it."""
    source, library = tmp_path/"advance_output_schedule.cpp", tmp_path/"advance_output_schedule.so"
    source.write_text(SCHEDULE_SHIM)
    compiled = subprocess.run([native_cxx, "-std=c++20", "-O2", "-shared", "-fPIC",
                               "-I" + str(REPO_ROOT/"include"), str(source), "-o", str(library)],
                              capture_output=True, text=True, check=False)
    assert compiled.returncode == 0, compiled.stderr
    shim = ctypes.CDLL(str(library))
    shim.due.argtypes = [ctypes.c_double, ctypes.c_int64, ctypes.c_int64,
                         ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_size_t)]
    shim.due.restype = ctypes.c_int
    shim.next_frontier.argtypes = [ctypes.c_double, ctypes.c_double,
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                                   ctypes.c_size_t]
    shim.next_frontier.restype = ctypes.c_double
    times = (ctypes.c_double * 3)(0.1, 0.2, 0.3)
    cursor = ctypes.c_size_t(0)

    def publish(time, macro_step, every=0):
        return bool(shim.due(time, macro_step, every, times, 3, ctypes.byref(cursor))), cursor.value

    assert shim.next_frontier(0.0, 1.0, times, 3, 0) == 0.1
    assert publish(0.05, 1) == (False, 0)
    # One publication at 0.25 passes 0.1 and 0.2: one evaluation, and both times are behind it.
    assert publish(0.25, 2) == (True, 2)
    assert shim.next_frontier(0.25, 1.0, times, 3, cursor.value) == 0.3
    assert publish(0.3, 3) == (True, 3)
    assert shim.next_frontier(0.3, 1.0, times, 3, cursor.value) == 1.0
    # every(k) is due on the multiples of k whether or not a time is reached.
    assert publish(0.4, 4, every=2) == (True, 3) and publish(0.5, 5, every=2) == (False, 3)

