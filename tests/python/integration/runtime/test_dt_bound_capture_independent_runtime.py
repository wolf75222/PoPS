"""Public native reception of readonly dt-bound captures; executed by integration.

No private executor/ABI query: AdaptiveCFL/run evaluates the native bound.
"""
import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import AdaptiveCFL
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


@pytest.mark.parametrize("capture", ("main_current", "query_only"))
def test_native_public_bound_capture_route_and_permutation(
        isolated_native_cache, native_cxx, kokkos_root, capture):
    del isolated_native_cache, native_cxx, kokkos_root
    for order in ((0, 1, 2), (2, 0, 1)):
        case, layout, _, _, _ = authored_capture(capture, order)
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        initial, query = arrays(order)
        values = {"fluid": initial.copy()}
        if capture == "query_only":
            values["query_only"] = query.copy()
        runtime = pops.bind(artifact, initial_state=values,
                            resources={"execution_context": artifact_execution_context(artifact)})
        factors = np.array([2., 3., 5.])[list(order), None, None]
        first = initial * factors
        bound_values = query if capture == "query_only" else initial
        first_dt = bound_duration(bound_values)
        second_dt = bound_duration(query if capture == "query_only" else first)
        endpoint = first_dt + second_dt
        # Two accepted updates detect a bound silently omitted, routed to block/component 0,
        # or reusing the old snapshot after the first accepted update.
        report = pops.run(runtime, t_end=endpoint, max_steps=2, console=False)
        assert report.accepted_steps == 2 and report.rejected_steps == 0
        assert runtime.time() == endpoint and runtime.macro_step() == 2
        actual = np.asarray(runtime.state_global("fluid"))
        if actual.size:
            np.testing.assert_array_equal(actual.reshape(initial.shape), initial * factors * factors)
        if capture == "query_only":
            readonly = np.asarray(runtime.state_global("query_only"))
            if readonly.size:
                np.testing.assert_array_equal(readonly.reshape(query.shape), query)


def test_native_dot_all_invalid_bound_input_does_not_publish(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _, _, _ = authored_capture("query_only")
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial, query = arrays((0, 1, 2))
    for poison in (np.nan, 1.e308):
        invalid = query.copy()
        invalid[2, 0, 0] = poison  # The extra component must participate in the query.
        runtime = pops.bind(artifact, initial_state={"fluid": initial.copy(), "query_only": invalid},
                            resources={"execution_context": artifact_execution_context(artifact)})
        before = {name: np.asarray(runtime.state_global(name)).copy()
                  for name in ("fluid", "query_only")}
        cursors = runtime.consumer_cursors.to_data()
        with pytest.raises(RuntimeError, match="dot_all|non-finite|finite_local"):
            pops.run(runtime, t_end=.01, max_steps=1, console=False)
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert runtime.consumer_cursors.to_data() == cursors
        for name, prior in before.items():
            np.testing.assert_array_equal(runtime.state_global(name), prior)
