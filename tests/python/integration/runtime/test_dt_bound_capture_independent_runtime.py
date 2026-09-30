"""Public native reception of readonly dt-bound captures; executed by integration.

No private executor/ABI query: AdaptiveCFL/run evaluates the native bound.
"""
import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import AdaptiveCFL
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check, state_snapshots,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
CELLS = 4
CFL = .25


def authored_capture(capture, order=(0, 1, 2), *, install_bound=True):
    frame = Rectangle("capture_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("capture_model", frame=frame)
    state = model.state("U", components=tuple(("a", "b", "c")[i] for i in order))
    flux = model.flux("zero", frame=frame, state=state,
                      components={axis: tuple(0 * q for q in state) for axis in frame.axes},
                      waves={axis: tuple(1 + 0 * q for q in state) for axis in frame.axes})
    rate = model.rate("storage", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                        reconstruction=reconstruction.FirstOrder(),
                                        riemann=riemann.Rusanov()))
    case = pops.Case("capture_case")
    # Authored block order deliberately differs from the Program's update order.
    query = case.block("query_only", model) if capture == "query_only" else None
    fluid = case.block("fluid", model)
    for block in (fluid,) if query is None else (query, fluid):
        case.numerics(numerics, block=block)
        case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                          projection=ConservativeCellAverage()))
    program = pops.Program("capture_step")
    q = program.state(fluid[state])
    old = q.n  # This exact main-region State is later captured by the bound.
    factors = tuple((2., 3., 5.)[i] for i in order)
    candidate = program.value("changed", tuple(old[i] * factors[i] for i in range(3)),
                              at=q.next.point)
    program.commit(q.next, candidate)
    bound = old if query is None else program.state(query[state])

    def query_bound(P, cfl):
        current = bound if query is None else bound.n
        return cfl / (1 + P.dot_all(current, current))

    if install_bound:
        program.set_dt_bound(query_bound)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, q, candidate


def arrays(order):
    fluid = np.broadcast_to(np.array([2., 5., 7.])[list(order), None, None],
                             (3, CELLS, CELLS)).copy()
    query = np.broadcast_to(np.array([11., 13., 17.])[list(order), None, None],
                             fluid.shape).copy()
    return fluid, query


def bound_duration(values):
    # Independent dense oracle across every cell and every component.
    return CFL / (1. + float(np.sum(values * values)))


def initial_subject_values(artifact, initial, query=None):
    supplied = {"fluid": initial}
    if query is not None:
        supplied["query_only"] = query
    bindings = artifact.plan.initial_condition_plan.bindings
    assert {b.subject.block_ref.local_id for b in bindings} == set(supplied)
    return {b.subject: np.ascontiguousarray(supplied[b.subject.block_ref.local_id])
            for b in bindings}


def compiled_capture(world, capture, order=(0, 1, 2)):
    def resolve():
        case, layout, _, _, _ = authored_capture(capture, order)
        return pops.resolve(pops.validate(case), layout=layout)
    resolved = collective_call(world, resolve)
    return compile_resolved_plan_once(world, resolved, route="independent dt bound " + capture,
                                      compile_artifact=pops.compile)


def initialized_runtime(world, artifact, initial, query=None):
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    supplied = collective_call(world, lambda: initial_subject_values(artifact, initial, query))
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=supplied,
                                                       resources={"execution_context": context}))
    names = ("fluid",) if query is None else ("fluid", "query_only")
    before = state_snapshots(runtime, world, names)
    with collective_check(world):
        assert runtime.time() == 0. and runtime.macro_step() == 0
        if world.rank == 0:
            np.testing.assert_array_equal(before[0].reshape(initial.shape), initial)
            if query is not None:
                np.testing.assert_array_equal(before[1].reshape(query.shape), query)
    return runtime, before


@pytest.mark.parametrize("capture", ("main_current", "query_only"))
def test_native_public_bound_capture_route_and_permutation(
        isolated_native_cache, native_cxx, kokkos_root, capture):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    world = select_native_dimension(2).mpi_world()
    for order in ((0, 1, 2), (2, 0, 1)):
        artifact = compiled_capture(world, capture, order)
        initial, query = collective_call(world, lambda order=order: arrays(order))
        runtime, _ = initialized_runtime(world, artifact, initial,
                                         query if capture == "query_only" else None)
        with collective_check(world):
            factors = np.array([2., 3., 5.])[list(order), None, None]
            first = initial * factors
            bound_values = query if capture == "query_only" else initial
            first_dt = bound_duration(bound_values)
            second_dt = bound_duration(query if capture == "query_only" else first)
            endpoint = first_dt + second_dt
        # Two accepted updates detect a bound silently omitted, routed to block/component 0,
        # or reusing the old snapshot after the first accepted update.
        report = collective_call(world, lambda runtime=runtime, endpoint=endpoint:
                                 pops.run(runtime, t_end=endpoint, max_steps=2, console=False))
        names = ("fluid", "query_only") if capture == "query_only" else ("fluid",)
        after = state_snapshots(runtime, world, names)
        with collective_check(world):
            assert report.accepted_steps == 2 and report.rejected_steps == 0
            assert runtime.time() == endpoint and runtime.macro_step() == 2
            if world.rank == 0:
                np.testing.assert_array_equal(after[0].reshape(initial.shape),
                                               initial * factors * factors)
                if capture == "query_only":
                    np.testing.assert_array_equal(after[1].reshape(query.shape), query)


def test_native_dot_all_invalid_bound_input_does_not_publish(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    world = select_native_dimension(2).mpi_world()
    artifact = compiled_capture(world, "query_only")
    initial, query = collective_call(world, lambda: arrays((0, 1, 2)))
    with collective_check(world):
        invalid = query.copy()
        invalid[2, 0, 0] = 1.e308  # Finite bind payload; its square overflows in dot_all.
        assert np.isfinite(invalid).all()
    runtime, before = initialized_runtime(world, artifact, initial, invalid)
    with collective_check(world):
        cursors = runtime.consumer_cursors.to_data()
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=.01,
                                                              max_steps=1, console=False))
    after = state_snapshots(runtime, world, ("fluid", "query_only"))
    with collective_check(world):
        assert all(failures), "a rank accepted an overflowing dt-bound contraction"
        # Native std::overflow_error maps to OverflowError in serial; an MPI
        # collective rejection may wrap it in RuntimeError on every participant.
        assert all((failure[0] == "OverflowError" or failure[2])
                   and ("dot_all" in failure[1] or "finite_local" in failure[1])
                   for failure in failures), failures
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert runtime.consumer_cursors.to_data() == cursors
        for current, prior in zip(after, before, strict=True):
            np.testing.assert_array_equal(current, prior)
