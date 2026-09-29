"""Installed C17 witness: a small Newton step never publishes an unsolved state."""

from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, .125
ROOT = np.sqrt(2.)
TOLERANCE = 1.e-12


def _world():
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator

    return _pops.mpi_world() if native_mpi_communicator(_pops) == "MPI_COMM_WORLD" else None


def _all_failures(world, failure):
    if world is None:
        return (failure,)
    from pops._native_collectives import allgather_value

    return allgather_value(world, failure)


def _root_check(world, operation):
    failure = ""
    if world is None or world.rank == 0:
        try:
            operation()
        except Exception as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
    if world is not None:
        from pops._native_collectives import broadcast_value

        failure = broadcast_value(world, failure, root=0)
    assert not failure, failure


def _case(step_tolerance):
    frame = Rectangle("c17_original_residual", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("c17_square_root", frame=frame)
    state = model.state("U", components=("z",))
    # Explicit storage route avoids any default spatial/Poisson discretization.
    zero = model.source("zero", on=state, value=(0 * state[0],))
    rate = model.rate("stored", equation=ddt(state) == zero)
    case = pops.Case("c17_stagnation")
    block = case.block("material", model=model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, StateStorage())
    case.numerics(numerics, block=block)

    program = pops.Program("c17_original_equation")
    q = program.state(block[state])
    seed = program.value("algorithmic_seed", 1 * q.n, at=q.next.point)
    program.store_history("provisional_seed", seed, depth=2)

    def original_residual(_builder, unknown, fixed):
        return (unknown[0] * unknown[0] - fixed[0],)

    candidate = program.solve(
        LocalResidual(original_residual, seed, captures={"fixed": q.n}),
        solver=LocalNewton(tolerance=TOLERANCE, step_tolerance=step_tolerance,
                           max_iterations=20),
    ).consume(action=FailRun())
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout


def _compile(case, layout, route):
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    world = _world()
    if world is None:
        artifact = pops.compile(resolved)
    else:
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

        artifact = compile_resolved_plan_once(
            world, resolved, route=route, compile_artifact=pops.compile)
    assert artifact.resolved_dimension == 2
    return artifact, world


def _state(runtime, world):
    values = np.asarray(runtime.state_global("material"))
    return values.reshape(1, N, N).copy() if world is None or world.rank == 0 else None


@pytest.mark.parametrize("step_tolerance,accepted", ((1., False), (0., True)))
def test_original_residual_is_required_before_publication_and_history_rotation(
        isolated_native_cache, native_cxx, kokkos_root, step_tolerance, accepted):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = _case(step_tolerance)
    artifact, world = _compile(case, layout, "c17-stagnation-%s" % int(accepted))
    initial = np.full((1, N, N), 2.)
    runtime = pops.bind(artifact, initial_state={"material": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    before = _state(runtime, world)
    before_history = runtime._executor.history_fill_count("provisional_seed")
    before_diagnostics = dict(runtime._executor.program_diagnostics())
    before_clock = (runtime.time(), runtime.macro_step())
    if accepted:
        result = pops.run(runtime, t_end=DT, max_steps=1, console=False)
        assert result.accepted_steps == 1
        assert runtime._executor.history_fill_count("provisional_seed") == before_history + 1
        assert runtime.time() == DT and runtime.macro_step() == 1
    else:
        failure = ""
        try:
            pops.run(runtime, t_end=DT, max_steps=1, console=False)
        except Exception as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
        failures = _all_failures(world, failure)
        assert all(item.startswith("RuntimeError:") and
                   "safeguard_failure" in item and "action=fail_run" in item
                   for item in failures), failures
        assert runtime._executor.history_fill_count("provisional_seed") == before_history
        assert dict(runtime._executor.program_diagnostics()) == before_diagnostics
        assert (runtime.time(), runtime.macro_step()) == before_clock

    after = _state(runtime, world)

    def check():
        np.testing.assert_array_equal(before, initial)
        if accepted:
            np.testing.assert_allclose(after, ROOT, rtol=0., atol=TOLERANCE)
        else:
            np.testing.assert_array_equal(after, initial)

    _root_check(world, check)
