"""C22 native Scalar frontier: relaxed RK rotation with real continuation/restart."""

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.time import ComputedDt, Program, RejectAttempt
from tests.python.support.native_execution_context import artifact_execution_context

CELLS = 8


def rotation_case(*, retry=False, duration_scale=1.):
    frame = Rectangle("rotation_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("computed_rotation", frame=frame)
    state = model.state("U", components=("x", "y"))
    x, y = state
    source = model.source("rotation", on=state, value=(-y, x))
    rate = model.rate("rotation_balance", equation=ddt(state) == source)
    case = pops.Case("computed_rotation_case")
    block = case.block("rotation", model)
    discretization = DiscretizationPlan()
    discretization.rates.add(rate, StateStorage())
    case.numerics(discretization, block=block)
    program = Program("relaxed_heun")
    q = program.state(block[state])
    requested = program.requested_dt()
    rhs0 = program.source(model.module.operator_handle("rotation"), q.n)
    predictor = program.value("predictor", q.n + program.dt * rhs0, at=q.next.point)
    rhs1 = program.source(model.module.operator_handle("rotation"), predictor)
    unrelaxed = program.value("heun", q.n + .5 * program.dt * rhs0 + .5 * program.dt * rhs1,
                             at=q.next.point)
    increment = program.value("increment", unrelaxed - q.n, at=q.next.point)
    # Generic energy relaxation: gamma is evaluated by native collective dot products.
    gamma = -2. * program.dot_all(q.n, increment) / program.dot_all(increment, increment)
    candidate = program.value("relaxed", q.n + increment * gamma, at=q.next.point)
    if retry:
        candidate = program.guard("requested_duration_budget", candidate, requested < .75,
                                  action=RejectAttempt())
    program.reached_duration(gamma * requested * duration_scale)
    program.commit(q.next, candidate)
    program.step_strategy(ComputedDt(1., endpoint_ulps=4, max_rejections=1 if retry else 0))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, model, program


def reference(state, requested):
    rotation = np.array([[0., -1.], [1., 0.]])
    first = rotation @ state
    second = rotation @ (state + requested * first)
    increment = .5 * requested * (first + second)
    gamma = -2. * np.dot(state, increment) / np.dot(increment, increment)
    return state + gamma * increment, float(gamma * requested)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("retry", (False, True))
def test_native_scalar_frontier_rotation_restart_and_retry(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, retry):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _, _ = rotation_case(retry=retry)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    context = artifact_execution_context(artifact)
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    initial = np.zeros((2, CELLS, CELLS))
    initial[0] = 1.

    def bind():
        return pops.bind(artifact, initial_values={subject: initial.copy()},
                         resources={"execution_context": context})

    actual = bind()
    expected, duration = reference(np.array([1., 0.]), .5 if retry else 1.)
    report = pops.run(actual, t_end=duration, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == int(retry)
    assert actual.time() == duration and actual.macro_step() == 1
    fields = np.asarray(actual.state_global("rotation"))
    if fields.size:
        np.testing.assert_allclose(fields, np.broadcast_to(expected[:, None, None], initial.shape),
                                   rtol=0., atol=3.e-14)
    receipt = actual._executor._temporal_restart_state.controller_state["program_frontier"]
    assert receipt["requested_duration"] == (.5 if retry else 1.).hex()
    assert float.fromhex(receipt["duration"]) == actual.time()
    assert receipt["reached"] == actual.time().hex()
    checkpoint = actual.checkpoint(str(tmp_path / "computed_rotation"))
    restored = bind()
    restored.restart(checkpoint)
    second, second_duration = reference(expected, .5 if retry else 1.)
    end = duration + second_duration
    for runtime in (actual, restored):
        pops.run(runtime, t_end=end, max_steps=1, console=False)
    assert actual.time() == restored.time()
    lower = upper = end
    for _ in range(4):
        lower = float(np.nextafter(lower, -np.inf))
        upper = float(np.nextafter(upper, np.inf))
    assert lower <= actual.time() <= upper
    np.testing.assert_array_equal(actual.state_global("rotation"), restored.state_global("rotation"))
    if fields.size:
        np.testing.assert_allclose(actual.state_global("rotation"),
            np.broadcast_to(second[:, None, None], initial.shape), rtol=0., atol=5.e-14)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("scale", (0., -1., 2.))
def test_native_invalid_or_overshooting_scalar_frontier_rolls_back(
        isolated_native_cache, native_cxx, kokkos_root, scale):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _, _ = rotation_case(duration_scale=scale)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.zeros((2, CELLS, CELLS))
    initial[0] = 1.
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial},
                       resources={"execution_context": artifact_execution_context(artifact)})
    temporal = runtime._executor._temporal_restart_state.to_data()
    before = np.asarray(runtime.state_global("rotation")).copy()
    with pytest.raises((RuntimeError, ValueError), match="frontier"):
        pops.run(runtime, t_end=.8, max_steps=1, console=False)
    assert runtime.time() == 0. and runtime.macro_step() == 0
    np.testing.assert_array_equal(runtime.state_global("rotation"), before)
    assert runtime._executor._temporal_restart_state.time_hex == temporal["clock"]["time"]
    assert "program_frontier" not in runtime._executor._temporal_restart_state.controller_state


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_computed_frontier_rank_local_run_limit_is_collectively_refused(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _, _ = rotation_case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    context = artifact_execution_context(artifact)
    world = context.communicator.handle
    if world is None or int(world.size) < 2:
        pytest.skip("rank-local frontier refusal requires the installed MPI2 reception")
    initial = np.zeros((2, CELLS, CELLS))
    initial[0] = 1.
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial},
                       resources={"execution_context": context})
    before = np.asarray(runtime.state_global("rotation")).copy()
    with pytest.raises(RuntimeError, match="ComputedDt preparation differs"):
        pops.run(runtime, t_end=.8 if int(world.rank) == 0 else .7, max_steps=1, console=False)
    assert runtime.time() == 0. and runtime.macro_step() == 0
    np.testing.assert_array_equal(runtime.state_global("rotation"), before)
