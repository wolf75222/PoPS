"""Lot D-E of N15 (ROM-98, OD31), part D, native: ``dt=cfl_limit(flux, cfl=, max_dt=)``.

The physics of the slice at first order: ``ddt(u) == -div(vector(1, 1/4) * u)`` on the box
[0, 1] x [0, 1] with 64 x 64 cells, ``rusanov()``, the SSPRK2 of ``pops.lib.time.methods``,
inflow at the far field on x_min and y_min and outflow on x_max and y_max, and
``dt=cfl_limit(advection, cfl=0.45, max_dt=1e-2)``. The compiled run of the surface chooses each
step by N8's compiled step rule and lands on its frontiers:

1. a run to T_END gives, bit for bit, the steps and the state of the same case written by hand
   for b6 and N8 (``StepRule(0.45, 1e-2, strategy=AdaptiveCFL(0.45, 1e-2))``) under
   ``Simulation.run``, and of master's ``AdaptiveCFL`` route under ``pops.run`` on that
   artifact and on the surface's own artifact (review R1 of #782);
2. runs to 0.05, 0.1 and T_END in turn land on each frontier with the hand-written case's steps
   and states.
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
from pops.lib.numerics.boundaries import ghosts
from pops.lib.numerics import rusanov
from pops.lib.time import cfl_limit
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.surface.math import ddt, div, outflow, vector
from pops.time._program.elaboration import Advance, StepRule, elaborate
from pops.time._step.strategy import AdaptiveCFL
from pops.time._program.traced_method import Candidate, method

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (64, 64)
AX, AY = 1.0, 0.25
FAR_FIELD = 0.05
CFL, MAX_DT, T_END = 0.45, 1.0e-2, 0.20
FRONTIERS = (0.05, 0.1, T_END)


def _surface_case():
    domain = surface.Box(x=(0.0, 1.0), y=(0.0, 1.0))
    sides = domain.boundary
    model = surface.Model("scalar_advection")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(AX, AY) * u)
    model.balance(ddt(u) == -div(advection))
    model.on([sides.x_min, sides.y_min], u == FAR_FIELD)
    model.on([sides.x_max, sides.y_max], outflow(u))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: rusanov()}, boundary=ghosts())
    program = surface.Program(
        advance=surface.Advance(ssprk2(rate=model),
                                dt=cfl_limit(advection, cfl=CFL, max_dt=MAX_DT)),
        initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, u0


@method
def _n8_ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    first = u + h * rate(u, t)
    return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)


def _rule(fraction, max_dt=None):
    """N8's rule as cfl_limit realizes it: the rule, and master's AdaptiveCFL for pops.run."""
    return StepRule(fraction, max_dt, strategy=AdaptiveCFL(fraction, max_dt))


def _hand_case():
    """The same case written for b6 and N8."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("scalar_advection", frame=frame)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (AX * u[0],), y: (AY * u[0],)}, waves={x: (AX,), y: (AY,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, FiniteVolume(
        flux=flux_u, variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("scalar_advection")
    block = case.block("scalar_advection", model)
    boundaries = frame.boundaries
    plan.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=block[u], value=FAR_FIELD),
        boundaries.y_min: Inflow(state=block[u], value=FAR_FIELD),
        boundaries.x_max: Outflow(state=block[u]),
        boundaries.y_max: Outflow(state=block[u])}))
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=_rule(CFL, MAX_DT))
    return case, Uniform(CartesianGrid(frame=frame, cells=CELLS))


def _initial():
    """The bump of master's tutorial 02 at the cell centres, u0[i, j] at (x_i, y_j)."""
    x = (np.arange(CELLS[0]) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1]) + 0.5) / CELLS[1]
    return FAR_FIELD + 0.95 * np.exp(-120.0 * ((x[:, None] - 0.30) ** 2
                                               + (y[None, :] - 0.35) ** 2))


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
        case, u, u0 = _surface_case()
        artifact = surface.compile(case, target=surface.OpenMP())
        hand_case, layout = _hand_case()
        hand = pops.compile(pops.resolve(pops.validate(hand_case), layout=layout))
        yield {"u": u, "u0": u0, "artifact": artifact, "hand": hand}


def _hand_runtime(compiled, artifact=None):
    hand = compiled["hand"] if artifact is None else artifact
    (binding,) = tuple(hand.plan.initial_condition_plan.bindings)
    initial = np.ascontiguousarray(np.transpose(_initial())[np.newaxis, ...])
    return pops.bind(hand, initial_values={binding.subject: initial})


def _hand_state(runtime):
    return np.transpose(np.asarray(runtime.state_global("scalar_advection"))[0]).copy()


def _surface(compiled, tmp_path):
    return surface.bind(compiled["artifact"], data={compiled["u0"]: _initial()},
                        directory=tmp_path)


def test_a_run_to_t_end_is_the_hand_written_rule_and_masters_adaptive_cfl(compiled, tmp_path):
    result = _surface(compiled, tmp_path).run(until=T_END)
    assert result.failure is None and result.time == T_END
    final = result.state[compiled["u"]]

    runtime = _hand_runtime(compiled)
    hand = Simulation(runtime).run(T_END)
    assert hand.failure is None and hand.time == T_END
    assert result.steps == hand.steps
    _bitwise(final, np.transpose(np.asarray(hand.state["scalar_advection"])[0]))

    master = _hand_runtime(compiled)
    pops.run(master, t_end=T_END, max_steps=10_000, console=False)
    assert master.time() == T_END and master.macro_step() == result.steps
    _bitwise(final, _hand_state(master))
    # the CFL step, 0.45 / 64 / 1.0 ~ 0.0070, is below max_dt: 29 steps, the last one cut
    assert result.steps == 29

    # review R1 of #782: pops.run on the surface's own artifact takes the strategy that
    # cfl_limit pairs with its rule, master's AdaptiveCFL, and reaches the same steps and state
    own = _hand_runtime(compiled, compiled["artifact"].compiled)
    pops.run(own, t_end=T_END, max_steps=10_000, console=False)
    assert own.time() == T_END and own.macro_step() == result.steps
    _bitwise(final, _hand_state(own))


def test_runs_in_turn_land_on_each_frontier(compiled, tmp_path):
    simulation = _surface(compiled, tmp_path)
    runtime = _hand_runtime(compiled)
    hand = Simulation(runtime)
    steps = 0
    for frontier in FRONTIERS:
        result = simulation.run(until=frontier)
        expected = hand.run(frontier)
        assert result.failure is None and expected.failure is None
        assert result.time == frontier == expected.time
        steps += result.steps
        assert steps == int(runtime.macro_step())
        _bitwise(result.state[compiled["u"]], _hand_state(runtime))
