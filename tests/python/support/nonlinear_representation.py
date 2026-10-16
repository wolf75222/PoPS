"""Public source-only coordinate equation and its independent physical oracle."""
import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import DerivativeStrategy, FailRun, FixedDt, ImplicitStage, SolveUnknown


def build(*, workspace_byte_capacity=64 << 20, cells=4):
    frame = Rectangle("representation-domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("representation-equations", frame=frame)
    state = model.state("U", components=("u", "v"))
    u, v = state
    source = model.source("rate", on=state, value=(-u+v, u-2*v))
    rate = model.rate("evolution", equation=ddt(state) == source)
    mapping = model.local_transform("physical", (u+u*u, v+v*v))
    case = pops.Case("closed-source-representation")
    block = case.block("material", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, StateStorage())
    case.numerics(numerics, block=block)
    p = pops.Program("closed-source-representation",
                     representation_workspace_byte_capacity=workspace_byte_capacity)
    temporal = p.state(block[state])
    seed = p.value("seed", temporal.n, at=temporal.next.point)
    request = ImplicitStage(rate, temporal.n, p.dt, accumulation=mapping).request(
        unknown=SolveUnknown("coordinates", seed), seed=seed,
        derivative=DerivativeStrategy("finite_difference"))
    raw = p.solve(request, solver=Newton(tolerance=1.e-12, max_iterations=20,
        linear_tolerance=1.e-11, linear_max_iterations=80, restart=20)).consume(action=FailRun())[0]
    physical = p.transform(raw, transform=mapping)
    p.commit(temporal.next, p.value("accepted", physical, at=temporal.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
        projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)))
    x, y = np.meshgrid((np.arange(cells)+.5)/cells, (np.arange(cells)+.5)/cells)
    initial = np.stack((.8+.1*x, .4+.05*y))
    return case, layout, initial, .01


def physical_step(values, dt):
    # Independent inversion of I-dt*[[-1,1],[1,-2]]. No coordinate or Newton code.
    u, v = values
    determinant = 1+3*dt+dt*dt
    return np.stack((((1+2*dt)*u+dt*v)/determinant,
                     (dt*u+(1+dt)*v)/determinant))
