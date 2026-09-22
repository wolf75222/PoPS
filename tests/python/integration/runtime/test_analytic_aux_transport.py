"""Native geometry auxiliary and mixed-topology transport against an exact FV update."""
import math

import numpy as np
import pops
import pytest

from pops.analytic import coordinate, sin
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import NoFlux
from pops.domain import Rectangle
from pops.fields import AnalyticAux, AuxiliaryBoundary
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, Var, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


@pytest.mark.parametrize("layout_kind", ("uniform", "one_level_amr"))
def test_native_analytic_metric_and_periodic_faces_match_conservative_update(
        isolated_native_cache, native_cxx, kokkos_root, layout_kind):
    del isolated_native_cache, native_cxx, kokkos_root
    nr, nt, dt = 16, 32, .001
    frame = Rectangle("mixed geometry", lower=(0., 0.), upper=(16., 2 * math.pi)).frame(Cartesian2D())
    model = pops.Model("analytic metric", frame=frame)
    state = model.state("U", components=("q",))
    metric = Var("radius", "aux")
    flux = model.flux("transport", frame=frame, state=state,
        components={frame.x: (0 * state[0],), frame.y: (metric * state[0],)},
        waves={frame.x: (0 * state[0],), frame.y: (metric,)})
    model.wave_speeds(flux, frame=frame, values={
        frame.x: (0 * state[0], 0 * state[0]), frame.y: (metric, metric)})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    module = model.module
    radius = module.aux_handle(module.aux_field("radius", frame=frame.canonical_id))
    module.aux_provider(AnalyticAux(radius, coordinate(frame, frame.x), frame=frame,
                                   boundary=AuxiliaryBoundary(width=1, kind="foextrap")))
    module.operator_registry().get(flux.reg_name).requirements["aux"] = ("radius",)
    case = pops.Case("metric transport")
    block = case.block("tracer", model=model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.HLL(waves=riemann.waves.ExplicitPair())))
    plan.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: NoFlux(state=block[state]),
        frame.boundaries.x_max: NoFlux(state=block[state]),
    }, periodic=PeriodicAxes((frame.y,))))
    case.numerics(plan, block=block)
    program = pops.Program("forward Euler metric transport")
    q = program.state(block[state])
    program.commit(q.next, program.value("accepted", q.n + program.dt * rate(q.n), at=q.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    r, theta = coordinate(frame, frame.x), coordinate(frame, frame.y)
    case.initials.add(InitialCondition(state=block[state],
        value=Analytic(frame=frame, components=(r * (2 + .1 * sin(theta)),)),
        projection=ConservativeCellAverage()))
    grid = CartesianGrid(frame=frame, cells=(nr, nt), periodic=PeriodicAxes((frame.y,)))
    if layout_kind == "uniform":
        layout = Uniform(grid)
    else:
        transfer = AMRTransfer()
        transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam("unused_refinement_threshold", default=1000.))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=1, ratios=()),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["q"] > case.value(threshold)),
                                    Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.synchronous())
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator
    from tests.python.support.native_execution_context import artifact_execution_context
    resolved = pops.resolve(pops.validate(case), layout=layout)
    if native_mpi_communicator(_pops) == "MPI_COMM_WORLD":
        # MPI ranks own distinct pytest caches. Publish and authenticate one binary.
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
        artifact = compile_resolved_plan_once(
            _pops.mpi_world(), resolved,
            route="analytic-metric-" + layout_kind, compile_artifact=pops.compile)
    else:
        artifact = pops.compile(resolved)
    runtime = pops.bind(artifact,
        resources={"execution_context": artifact_execution_context(artifact)})
    initial = np.asarray(runtime.block_level_state_global("tracer", 0)
        if layout_kind == "one_level_amr" else runtime.state_global("tracer")).reshape(nt, nr)
    radial = np.arange(nr) + .5
    angular = (np.arange(nt) + .5) * (2 * math.pi / nt)
    expected_initial = radial[None, :] * (
        2 + .1 * (math.sin(math.pi / nt) / (math.pi / nt)) * np.sin(angular[:, None]))
    np.testing.assert_allclose(initial, expected_initial, rtol=2e-14, atol=1e-14)
    expected = initial - dt * radial[None, :] / (2 * math.pi / nt) * (
        initial - np.roll(initial, 1, axis=0))
    report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == 0
    actual = np.asarray(runtime.block_level_state_global("tracer", 0)
        if layout_kind == "one_level_amr" else runtime.state_global("tracer")).reshape(nt, nr)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-14)
    assert abs(np.sum(actual) - np.sum(initial)) <= 5e-12
