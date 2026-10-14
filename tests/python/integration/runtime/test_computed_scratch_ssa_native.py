"""The real emitted native composition must compute 3*Lap(q), never 4*Lap(q)."""
import numpy as np
import pops
import pytest

from pops.fields import CellMidpoint
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context


def build_scratch_ssa_case(*, width=1, iterations=1, cells=(16, 4), dt=1e-4, steps=4):
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.numerics import DiscretizationPlan, StateStorage
    from pops.time import FixedDt
    frame = Rectangle("scratch_ssa", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("scratch_ssa", frame=frame)
    u = model.state("q", components=tuple(f"q{i}" for i in range(width)), sampling="cell_average")
    case = pops.Case("scratch_ssa")
    block = case.block("q", model)
    plan = DiscretizationPlan()
    plan.states.add(u, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    subject = block[u]
    case.initials.add(InitialCondition(state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    p = pops.Program("scratch_ssa")
    state = p.state(subject)

    def step(builder, current, _index=None):
        field = builder.cell_field(current, quadrature=CellMidpoint())
        prototype = builder.scalar_field(like=field)
        first = builder.laplacian(prototype, field)
        twice = builder.value("twice", 2*current)
        second_field = builder.cell_field(twice, quadrature=CellMidpoint())
        second = builder.laplacian(prototype, second_field)
        # Both reads deliberately occur after the second write. Their results
        # must retain separate SSA storage even though the prototype is shared.
        a = builder.cell_mean(first, template=current, quadrature=CellMidpoint())
        b = builder.cell_mean(second, template=current, quadrature=CellMidpoint())
        return builder.value("substep", current+(p.dt/iterations)*(a+b), at=current.point)

    candidate = step(p, state.n) if iterations == 1 else p.fold(state.n, iterations, step)
    p.commit(state.next, p.value("accepted", candidate, at=state.next.point))
    p.step_strategy(FixedDt(dt))
    case.program(p)
    layout = Uniform(CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)))
    x = (np.arange(cells[0])+.5)/cells[0]
    mean = (1+np.arange(width))[:, None, None]
    mode = np.broadcast_to(.01*(1+np.arange(width))[:, None, None]*np.sin(2*np.pi*x),
                           (width, cells[1], cells[0]))
    eigenvalue = -4*cells[0]**2*np.sin(np.pi/cells[0])**2
    expected = mean+(1+3*dt/iterations*eigenvalue)**(steps*iterations)*mode
    return case, layout, {subject: mean+mode}, expected


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("width,iterations", ((1, 1), (3, 1), (1, 2), (3, 2)))
def test_native_scratch_reuse_preserves_three_laplacians(isolated_native_cache, width, iterations, record_property):
    del isolated_native_cache
    _, world = native_profile(dimension=2, scope="computed scratch SSA lifetime")
    case, layout, initial, expected = collective_call(world, lambda: build_scratch_ssa_case(width=width, iterations=iterations))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="scratch_ssa", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=initial, resources={"execution_context": context}))
    collective_call(world, lambda: pops.run(runtime, t_end=4e-4, max_steps=4, console=False))
    values = collective_call(world, lambda: runtime.state_global("q"))
    with collective_check(world):
        if values is not None:
            error = float(np.max(abs(np.asarray(values).reshape(expected.shape)-expected)))
            assert error < 2e-12, (width, iterations, error)
            record_property("three_laplacian_error", error)
