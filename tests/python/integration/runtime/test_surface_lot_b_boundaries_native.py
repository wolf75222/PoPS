"""Lot B of N15 (ROM-98, OD31), native: a case of the surface with boundary relations, run.

An advection on the box [0, 1] x [0, 1] with no periodic axis, ``ddt(u) == -div(vector(1, 1/4)
* u)``, ``rusanov()`` and the SSPRK2 of ``pops.lib.time.methods``, at level 1 on 16 x 8 cells.
The inflow pieces x_min and y_min take ``u == g`` with the runtime parameter g, the outflow
pieces x_max and y_max take ``outflow(u)``, and ``pops.Numerics(boundary=ghosts())`` realizes
them (2.6.2, 2.6.9, D9). ``pops.compile``, ``pops.bind`` and ``simulation.run`` give, bit for bit,
the state that the same case written by hand for b6 and N8, with the ``TransportBoundarySet`` of
``Inflow`` and ``Outflow``, reaches through ``pops.bind`` and ``Simulation.run``, with g bound
and with its default; the inflow brings g into the box.
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
from pops.lib.numerics import ghosts
from pops.lib.numerics.schemes import rusanov
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.surface.math import ddt, div, outflow, vector
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 8)
G = 0.2                         # the bound value of the runtime parameter g (default 0.05)
UNTIL = 0.05                    # 50 steps of dt = 1e-3


def _surface_case():
    domain = surface.Box(x=(0.0, 1.0), y=(0.0, 1.0))
    sides = domain.boundary
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    g = model.parameter("g", default=0.05)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    model.on([sides.x_min, sides.y_min], u == g)
    model.on([sides.x_max, sides.y_max], outflow(u))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: rusanov()}, boundary=ghosts())
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, g, u0


@method
def _n8_ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    first = u + h * rate(u, t)
    return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)


def _hand_case():
    """The same case written for b6 and N8."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.25)))
    g_handle = model.param(RuntimeParam("g", default=0.05))
    g = model.value(g_handle)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (b * u[0],)}, waves={x: (a,), y: (b,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, FiniteVolume(
        flux=flux_u, variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
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
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS))
    return case, layout, block[g_handle]


def _initial():
    """u0[i, j] at the centre of the cell (x_i, y_j): no symmetry between the axes."""
    x = (np.arange(CELLS[0]) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1]) + 0.5) / CELLS[1]
    return 0.5 * np.exp(-30.0 * ((x[:, None] - 0.5) ** 2 + (y[None, :] - 0.5) ** 2))


def _bitwise(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        case, u, g, u0 = _surface_case()
        artifact = surface.compile(case, target=surface.OpenMP())
        hand_case, layout, g_handle = _hand_case()
        hand = pops.compile(pops.resolve(pops.validate(hand_case), layout=layout))
        yield {"u": u, "g": g, "u0": u0, "artifact": artifact, "hand": hand,
               "g_handle": g_handle}


def _hand_state(compiled, params):
    hand = compiled["hand"]
    (binding,) = tuple(hand.plan.initial_condition_plan.bindings)
    initial = np.ascontiguousarray(np.transpose(_initial())[np.newaxis, ...])
    runtime = pops.bind(hand, initial_values={binding.subject: initial},
                        params={compiled["g_handle"]: params[0]} if params else {})
    result = Simulation(runtime).run(UNTIL)
    assert result.failure is None
    return np.transpose(np.asarray(result.state["advection"])[0])


@pytest.mark.parametrize("bound", [(G,), ()], ids=["g bound", "g default"])
def test_inflow_and_outflow_run_as_the_hand_written_case(compiled, tmp_path, bound):
    u = compiled["u"]
    simulation = surface.bind(compiled["artifact"],
                              params={compiled["g"]: bound[0]} if bound else {},
                              data={compiled["u0"]: _initial()}, directory=tmp_path)
    result = simulation.run(until=UNTIL)
    assert result.failure is None and result.time == UNTIL and result.steps == 50
    final = result.state[u]
    assert final.shape == CELLS
    _bitwise(final, _hand_state(compiled, bound))
    # the inflow brings g into the corner cell (x_0, y_0), where u0 is about 1e-7
    g = bound[0] if bound else 0.05
    assert 0.3 * g < final[0, 0] < g
