"""N3 lot 2 (ROM-87), native: the advection of master's tutorial 02 written with the surface and
``muscl``.

The advection of tutorial 02 of ``scalar_advection`` (the case of S1-c of the vertical slice) on
the box [0, 1] x [0, 1] with no periodic axis, ``ddt(u) == -div(vector(1, 1/4) * u)``, inflow on
x_min and y_min (``u == g``) and outflow on x_max and y_max, realized by ``ghosts()``, and the
SSPRK2 of ``pops.lib.time.methods``, on 16 x 8 cells for 50 steps of dt = 1e-3. The face is
``upwind(reconstruction=muscl(limiter=X))`` with each limiter X of ``pops.lib.numerics``.
``pops.compile``, ``pops.bind`` and ``simulation.run`` give, bit for bit, the state that the same
case written by hand for b6 and N8 reaches with master's native
``reconstruction.MUSCL(limiters.X())`` and ``riemann.Rusanov()``, the form of ``upwind()`` (D1):
E2 of the lot note, ahead of S1-c.
"""
from pathlib import Path

import numpy as np
import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics import mc, minmod, muscl, superbee, upwind, vanleer
from pops.lib.numerics.boundaries import ghosts
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.surface.math import ddt, div, outflow, vector
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 8)
G = 0.05                        # the value of the inflow
UNTIL = 0.05                    # 50 steps of dt = 1e-3
LIMITERS = {"minmod": (minmod, limiters.Minmod), "vanleer": (vanleer, limiters.VanLeer),
            "mc": (mc, limiters.MC), "superbee": (superbee, limiters.Superbee)}


def _surface_case(limiter):
    domain = surface.Box(x=(0.0, 1.0), y=(0.0, 1.0))
    sides = domain.boundary
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    g = model.constant("g", value=G)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    model.on([sides.x_min, sides.y_min], u == g)
    model.on([sides.x_max, sides.y_max], outflow(u))
    numerics = surface.Numerics(
        represent={u: surface.CellAverage()},
        schemes={advection: upwind(reconstruction=muscl(limiter=limiter))}, boundary=ghosts())
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, u0


@method
def _n8_ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    first = u + h * rate(u, t)
    return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)


def _hand_case(native):
    """The same case written for b6 and N8, with master's MUSCL and Rusanov."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.25)))
    g = model.value(model.param(ConstParam("g", G)))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (b * u[0],)}, waves={x: (a,), y: (b,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, FiniteVolume(
        flux=flux_u, variables=variables.Conservative(u),
        reconstruction=reconstruction.MUSCL(native()), riemann=riemann.Rusanov()))
    case = pops.Case("advection")
    block = case.block("advection", model)
    boundaries = frame.boundaries
    plan.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=block[u], value=g),
        boundaries.y_min: Inflow(state=block[u], value=g),
        boundaries.x_max: Outflow(state=block[u]),
        boundaries.y_max: Outflow(state=block[u])}))
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=CELLS))


def _initial():
    """u0[i, j] at the centre of the cell (x_i, y_j): no symmetry between the axes."""
    x = (np.arange(CELLS[0]) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1]) + 0.5) / CELLS[1]
    return 0.5 * np.exp(-30.0 * ((x[:, None] - 0.5) ** 2 + (y[None, :] - 0.5) ** 2))


def _bitwise(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


@pytest.fixture(scope="module", params=tuple(LIMITERS))
def compiled(request, tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    word, native = LIMITERS[request.param]
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        case, u, u0 = _surface_case(word)
        artifact = surface.compile(case, target=surface.OpenMP())
        hand_case, layout = _hand_case(native)
        hand = pops.compile(pops.resolve(pops.validate(hand_case), layout=layout))
        yield {"u": u, "u0": u0, "artifact": artifact, "hand": hand}


def _hand_state(compiled):
    hand = compiled["hand"]
    (binding,) = tuple(hand.plan.initial_condition_plan.bindings)
    initial = np.ascontiguousarray(np.transpose(_initial())[np.newaxis, ...])
    runtime = pops.bind(hand, initial_values={binding.subject: initial})
    result = Simulation(runtime).run(UNTIL)
    assert result.failure is None
    return np.transpose(np.asarray(result.state["advection"])[0])


def test_muscl_runs_as_master_s_native_muscl(compiled, tmp_path):
    u = compiled["u"]
    simulation = surface.bind(compiled["artifact"], data={compiled["u0"]: _initial()},
                              directory=tmp_path)
    result = simulation.run(until=UNTIL)
    assert result.failure is None and result.time == UNTIL and result.steps == 50
    final = result.state[u]
    assert final.shape == CELLS
    _bitwise(final, _hand_state(compiled))
    # the inflow brings g into the corner cell (x_0, y_0), where u0 is about 1e-7
    assert 0.3 * G < final[0, 0] < G
