"""Lot D-E of N15 (ROM-98, OD31), part E, native: the projection of an analytic initial value.

A periodic advection on a box of rank two, ``rusanov()`` and the SSPRK2 of
``pops.lib.time.methods`` on 16 x 8 cells, whose initial value is ``pops.project(e, ...)``:

1. the bump of master's tutorial 02 with its amplitude a runtime parameter, by
   ``quadrature=GaussLegendre(1)``: the compiled initial state and the state after 20 steps are,
   bit for bit, those of the same case written by hand for b6 (``Analytic`` with the cell
   integral of the rule), and the initial state is the bump at the centres of the cells to
   round-off, with the amplitude bound at bind;
2. a Gaussian by ``integral=I``, its exact cell integral written with ``erf`` of the bounds of
   the cell: the initial state is the exact cell average to round-off;
3. the same Gaussian by ``quadrature=GaussLegendre(2)``: the tensor rule of two points per axis,
   computed in NumPy, to round-off.
"""
import math
from pathlib import Path

import numpy as np
import pytest

import pops
import pops.analytic as an
import pops.surface as surface
from pops import math as pm
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Analytic
from pops.lib.numerics.schemes import rusanov
from pops.lib.quadrature import GaussLegendre
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.surface.math import ddt, div, erf, exp, sqrt, vector
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 8)
AMPLITUDE = 0.75                # the bound value of the runtime parameter (default 0.95)
BETA = 30.0                     # the Gaussian exp(-BETA ((x - 0.5)^2 + (y - 0.5)^2))


def _surface_case(initial):
    """``initial(model, x, y)`` gives the initial value of u."""
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    value, handles = initial(model, domain.x, domain.y)
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: value})
    start = surface.Output(surface.record(u=u), when=[0.0])
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)}, outputs=[start])
    return case, u, start, handles


def _bump(model, x, y):
    amplitude = model.parameter("amplitude", default=0.95)
    e = 0.05 + amplitude * exp(-120.0 * ((x - 0.3) ** 2 + (y - 0.35) ** 2))
    return surface.project(e, quadrature=GaussLegendre(1)), {"amplitude": amplitude}


def _gaussian(x, y):
    return exp(-BETA * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def _exact(model, x, y):
    def integral(cell):
        scale = sqrt(BETA)
        factor = math.pi / BETA / 4
        return factor * (erf(scale * (cell.upper(x) - 0.5)) - erf(scale * (cell.lower(x) - 0.5))) \
            * (erf(scale * (cell.upper(y) - 0.5)) - erf(scale * (cell.lower(y) - 0.5)))

    return surface.project(_gaussian(x, y), integral=integral), {}


def _two_points(model, x, y):
    return surface.project(_gaussian(x, y), quadrature=GaussLegendre(2)), {}


@method
def _n8_ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    first = u + h * rate(u, t)
    return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)


def _hand_case():
    """Case 1 written for b6 and N8."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.25)))
    handle = model.param(RuntimeParam("amplitude", default=0.95))
    bounds = an.CellBounds(frame)

    def bump(px, py):
        return 0.05 + an.param(handle) * an.exp(-120.0 * ((px - 0.3) ** 2 + (py - 0.35) ** 2))

    def centre(axis):
        lower, upper = bounds.lower(axis), bounds.upper(axis)
        return lower + (upper - lower) * 0.5

    provider = Analytic(frame=frame, components=(bump(an.x(frame), an.y(frame)),),
                        cell_integrals=(bounds.measure * bump(centre(x), centre(y)),))
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
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=provider,
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=Proposal(0.001))
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes((x, y))))
    return case, layout, block[handle]


def _bitwise(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def _faces(n):
    return np.arange(n + 1) / n


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        artifacts = {}
        for name, initial in (("bump", _bump), ("exact", _exact), ("two", _two_points)):
            case, u, start, handles = _surface_case(initial)
            artifacts[name] = (surface.compile(case, target=surface.OpenMP()), u, start, handles)
        hand_case, layout, handle = _hand_case()
        hand = pops.compile(pops.resolve(pops.validate(hand_case), layout=layout))
        yield {"artifacts": artifacts, "hand": hand, "handle": handle}


def _initial_state(compiled, name, tmp_path, params=None):
    artifact, u, start, handles = compiled["artifacts"][name]
    bound = {handles[key]: value for key, value in (params or {}).items()}
    simulation = surface.bind(artifact, params=bound, directory=tmp_path)
    result = simulation.run(until=0.02)
    assert result.failure is None and result.steps == 20
    (row,) = result.outputs[start]
    assert row.time == 0.0
    return row["u"], result.state[u]


def test_a_one_point_rule_is_the_hand_written_case_and_the_bump_at_the_centres(compiled,
                                                                                tmp_path):
    initial, final = _initial_state(compiled, "bump", tmp_path, {"amplitude": AMPLITUDE})
    runtime = pops.bind(compiled["hand"], params={compiled["handle"]: AMPLITUDE})
    simulation = Simulation(runtime)
    hand_initial = np.transpose(np.asarray(runtime.state_global("advection"))[0]).copy()
    result = simulation.run(0.02)
    assert result.failure is None
    _bitwise(initial, hand_initial)
    _bitwise(final, np.transpose(np.asarray(result.state["advection"])[0]))
    fx, fy = _faces(CELLS[0]), _faces(CELLS[1])
    cx = fx[:-1] + (fx[1:] - fx[:-1]) * 0.5
    cy = fy[:-1] + (fy[1:] - fy[:-1]) * 0.5
    bump = 0.05 + AMPLITUDE * np.exp(-120.0 * ((cx[:, None] - 0.3) ** 2
                                               + (cy[None, :] - 0.35) ** 2))
    np.testing.assert_allclose(initial, bump, rtol=1e-14, atol=0.0)


def test_an_exact_integral_gives_the_exact_cell_average(compiled, tmp_path):
    initial, _ = _initial_state(compiled, "exact", tmp_path)
    fx, fy = _faces(CELLS[0]), _faces(CELLS[1])
    scale = math.sqrt(BETA)

    def primitive(faces):
        return np.array([math.erf(scale * (face - 0.5)) for face in faces])

    px, py = np.diff(primitive(fx)), np.diff(primitive(fy))
    exact = math.pi / BETA / 4 * px[:, None] * py[None, :] / (np.diff(fx)[:, None]
                                                              * np.diff(fy)[None, :])
    np.testing.assert_allclose(initial, exact, rtol=1e-13, atol=0.0)


def test_a_two_point_rule_is_its_tensor_average(compiled, tmp_path):
    initial, _ = _initial_state(compiled, "two", tmp_path)
    nodes, weights = np.polynomial.legendre.leggauss(2)
    fx, fy = _faces(CELLS[0]), _faces(CELLS[1])
    px = fx[:-1, None] + (fx[1:] - fx[:-1])[:, None] * ((nodes + 1) / 2)[None, :]
    py = fy[:-1, None] + (fy[1:] - fy[:-1])[:, None] * ((nodes + 1) / 2)[None, :]
    samples = np.exp(-BETA * ((px[:, None, :, None] - 0.5) ** 2
                              + (py[None, :, None, :] - 0.5) ** 2))
    average = np.einsum("ijab,a,b->ij", samples, weights / 2, weights / 2)
    np.testing.assert_allclose(initial, average, rtol=1e-14, atol=0.0)
