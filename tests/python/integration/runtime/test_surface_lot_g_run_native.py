"""Lot G of N15 (ROM-98, OD31), native: a case of the surface compiled, bound and run.

A periodic advection on a box of rank two, ``ddt(u) == -div(vector(a, b) * u)`` with the runtime
parameter ``b``, ``rusanov()`` and the SSPRK2 of ``pops.lib.time.methods``, at level 1 on 16 x 8
cells (an unequal count per axis, so that the order of the axes shows). ``pops.compile``,
``pops.bind`` and ``simulation.run`` give, bit for bit, the state that the same case written by
hand for b6 and N8 reaches through ``pops.bind`` and ``Simulation.run``, with the parameter
bound to the same value; the arrays of the surface are in the order of the support's axes (Q3).
Its Outputs publish the accepted state at their times and every k steps, and ``npz`` writes the
series of one of them after each run, with its metadata; a second simulation that would replace
that file is refused before it runs (2.6.9).

The parity check of A2 (review of the N15 lot « solves », P2): on one random periodic field of
16 x 8 cells, one forward Euler step of the generated two_point diffusion of ``-kappa grad(u)``
(``_TwoPoint``, b6's ``Diffusion``) and one step of ``u + dt kappa P.laplacian(u)``, b6's shared
Laplacian in a Program written by hand, agree to round-off, and both are kappa (u_{i+1} - 2 u_i +
u_{i-1}) / h^2 on each axis: the spacing and the sign of b6's convention, with unequal spacings.
"""
from pathlib import Path

import numpy as np
import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics.schemes import rusanov, two_point
from pops.lib.output import OutputExists, npz
from pops.lib.time.methods import forward_euler, ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.surface.math import ddt, div, grad, vector
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.outputs import EveryDt
from pops.time._program.traced_method import Candidate, method

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 8)
B = 0.25                        # the bound value of the runtime parameter b (default 0.5)
TIMES = [0.02, 0.05]
H = 0.004                       # pops.every(time=H)


def _surface_case():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.parameter("b", default=0.5, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    at_times = surface.Output(surface.record(u=u), when=TIMES,
                              write=npz("series/u.npz",
                                        metadata=surface.record(nx=CELLS[0], ny=CELLS[1])))
    by_steps = surface.Output(surface.record(u=u), when=surface.every(steps=10))
    by_time = surface.Output(surface.record(u=u), when=surface.every(time=H))
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)},
                        outputs=[at_times, by_steps, by_time])
    return case, u, b, u0, at_times, by_steps, by_time


@method
def _n8_ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    first = u + h * rate(u, t)
    return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)


def _hand_case():
    """The same case written for b6 and N8, without Outputs."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b_handle = model.param(RuntimeParam("b", default=0.5))
    b = model.value(b_handle)
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
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=Proposal(0.001))
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes((x, y))))
    return case, layout, block[b_handle]


def _initial():
    """u0[i, j] at the centre of the cell (x_i, y_j): no symmetry between the axes."""
    x = (np.arange(CELLS[0]) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1]) + 0.5) / CELLS[1]
    return (np.sin(2 * np.pi * x)[:, None] + 0.5 * np.cos(4 * np.pi * y)[None, :]
            + 0.01 * np.arange(CELLS[0])[:, None] * np.arange(CELLS[1])[None, :])


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
        case, u, b, u0, at_times, by_steps, by_time = _surface_case()
        artifact = surface.compile(case, target=surface.OpenMP())
        hand_case, layout, b_handle = _hand_case()
        hand = pops.compile(pops.resolve(pops.validate(hand_case), layout=layout))
        yield {"case": case, "u": u, "b": b, "u0": u0, "at_times": at_times,
               "by_steps": by_steps, "by_time": by_time, "artifact": artifact, "hand": hand,
               "b_handle": b_handle}


def _hand_states(compiled, frontiers):
    """The state of the hand-written case after runs to each frontier in turn, in b6's layout."""
    hand = compiled["hand"]
    (binding,) = tuple(hand.plan.initial_condition_plan.bindings)
    initial = np.ascontiguousarray(np.transpose(_initial())[np.newaxis, ...])
    runtime = pops.bind(hand, initial_values={binding.subject: initial},
                        params={compiled["b_handle"]: B})
    simulation = Simulation(runtime)
    states = {}
    # The run of the surface lands on the times of its Outputs, those of every(time=H) included:
    # the hand-written case runs to the same frontiers in turn.
    lattice = [EveryDt(H).time(k) for k in range(1, int(max(frontiers) / H) + 1)]
    frontiers = sorted(set(frontiers) | {t for t in lattice if t <= max(frontiers)})
    for frontier in frontiers:
        result = simulation.run(frontier)
        assert result.failure is None
        states[frontier] = (np.asarray(result.state["advection"]).copy(), runtime.macro_step())
    return states


def test_the_surface_run_is_the_run_of_the_hand_written_case(compiled, tmp_path):
    u, at_times, by_steps = compiled["u"], compiled["at_times"], compiled["by_steps"]
    simulation = surface.bind(compiled["artifact"], params={compiled["b"]: B},
                              data={compiled["u0"]: _initial()}, directory=tmp_path)
    result = simulation.run(until=0.05)
    reference = _hand_states(compiled, TIMES)
    state, steps = reference[0.05]
    assert result.failure is None and result.time == 0.05 and result.steps == steps == 50
    final = result.state[u]
    assert final.shape == CELLS
    _bitwise(final, np.transpose(state[0]))
    assert list(result.state) == [u]
    # the Output at a list of times publishes at those times only, landed exactly
    rows = result.outputs[at_times]
    assert [row.time for row in rows] == TIMES
    for row in rows:
        reached, macro_step = reference[row.time]
        assert row.step == macro_step
        _bitwise(row["u"], np.transpose(reached[0]))
    # every(steps=10): the start of the run, then every tenth accepted step
    cadence = result.outputs[by_steps]
    assert [row.step for row in cadence] == [0, 10, 20, 30, 40, 50]
    _bitwise(cadence[0]["u"], _initial())
    _bitwise(cadence[-1]["u"], final)
    assert result.diagnostics["threads"]["asked"] == 1
    # every(time=H): the start t0 = 0 and t0 + k H up to until, each landed on
    assert [row.time for row in result.outputs[compiled["by_time"]]] == [
        EveryDt(H).time(k) for k in range(13)]
    # npz: the series of the Output after the run, its times, steps and metadata
    with np.load(tmp_path / "series" / "u.npz") as archive:
        assert sorted(archive.files) == ["nx", "ny", "step", "time", "u"]
        assert archive["time"].tolist() == TIMES
        assert archive["step"].tolist() == [row.step for row in rows]
        assert archive["u"].shape == (2,) + CELLS
        _bitwise(archive["u"][1], final)
        assert int(archive["nx"]) == CELLS[0] and int(archive["ny"]) == CELLS[1]


def test_a_second_run_continues_and_rewrites_its_series(compiled, tmp_path):
    u, at_times = compiled["u"], compiled["at_times"]
    simulation = surface.bind(compiled["artifact"], params={compiled["b"]: B},
                              data={compiled["u0"]: _initial()}, directory=tmp_path)
    first = simulation.run(until=0.02)
    second = simulation.run(until=0.05)
    assert [row.time for row in first.outputs[at_times]] == [0.02]
    assert [row.time for row in second.outputs[at_times]] == [0.05]
    # the start of the second run is the end of the first: published once, by the first run
    by_steps = compiled["by_steps"]
    assert [row.step for row in first.outputs[by_steps]] == [0, 10, 20]
    assert [row.step for row in second.outputs[by_steps]] == [30, 40, 50]
    state, steps = _hand_states(compiled, TIMES)[0.05]
    assert (first.steps, second.steps) == (20, 30) and first.steps + second.steps == steps
    _bitwise(second.state[u], np.transpose(state[0]))
    with np.load(tmp_path / "series" / "u.npz") as archive:
        assert archive["time"].tolist() == TIMES
    # every(time=H) on the second run, whose start 0.02 = 5 H is on the lattice: t0 + k H
    assert [row.time for row in second.outputs[compiled["by_time"]]] == [
        EveryDt(H).time(k) for k in range(6, 13)]
    # a new simulation in the same directory would replace a file that it did not write
    other = surface.bind(compiled["artifact"], data={compiled["u0"]: _initial()},
                         directory=tmp_path)
    with pytest.raises(OutputExists, match="overwrite=True"):
        other.run(until=0.01)



def test_a_later_run_that_starts_off_the_lattice_of_every_is_refused(compiled, tmp_path):
    """F1 of the review of lot G: the run would publish N8's k H, where the spec gives t0 + k H;
    it is refused before it runs, and the simulation is left as it was."""
    simulation = surface.bind(compiled["artifact"], params={compiled["b"]: B},
                              data={compiled["u0"]: _initial()}, directory=tmp_path)
    first = simulation.run(until=0.01)          # 10 steps, on the lattice of every(steps=10)
    with pytest.raises(surface.NotYetElaborated,
                       match="starts at t0 = 0.01, off the lattice k h of pops.every\\(time=0.004\\)"
                             ".*the N8 lot 'run-anchored schedules' \\(ROM-89\\)"):
        simulation.run(until=0.015)
    assert first.time == 0.01 and first.steps == 10
    assert simulation._simulation._runtime.time() == 0.01


# --- A2: the two_point diffusion is kappa P.laplacian ---------------------------------------------

KAPPA = 0.125
DT = 1e-4


def _diffusion_case():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    kappa = model.constant("kappa", value=KAPPA)
    u0 = model.data("u0", surface.Scalar(domain))
    diffusion = model.diffusive_flux(of=u, value=-kappa * grad(u))
    model.balance(ddt(u) == -div(diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(forward_euler(rate=model), dt=DT),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, u0


def _laplacian_case():
    """One step of u + dt kappa P.laplacian(u), b6's shared Laplacian, written by hand."""
    from pops.fields import CellMidpoint
    from pops.numerics import StateStorage
    from pops.time import FixedDt

    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("laplacian", frame=frame)
    u = model.state("u", components=("u",), sampling="cell_average")
    case = pops.Case("laplacian")
    block = case.block("laplacian", model)
    plan = DiscretizationPlan()
    plan.states.add(u, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    subject = block[u]
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                       projection=ConservativeCellAverage()))
    program = pops.Program("laplacian")
    state = program.state(subject)
    current = state.n
    field = program.cell_field(current, quadrature=CellMidpoint())
    laplacian = program.laplacian(program.scalar_field(like=field), field)
    mean = program.cell_mean(laplacian, template=current, quadrature=CellMidpoint())
    candidate = program.value("step", current + (program.dt * KAPPA) * mean, at=current.point)
    program.commit(state.next, program.value("accepted", candidate, at=state.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes(frame.axes)))
    return case, layout, subject


def _stencil(value):
    """kappa (u_{i+1} - 2 u_i + u_{i-1}) / h^2 on each axis, periodic, in the order of the axes."""
    total = np.zeros_like(value)
    for axis, count in enumerate(CELLS):
        total += (np.roll(value, -1, axis) - 2 * value + np.roll(value, 1, axis)) * count ** 2
    return KAPPA * total


@pytest.fixture(scope="module")
def diffusion(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        case, u, u0 = _diffusion_case()
        artifact = surface.compile(case, target=surface.OpenMP())
        hand_case, layout, subject = _laplacian_case()
        hand = pops.compile(pops.resolve(pops.validate(hand_case), layout=layout))
        yield {"artifact": artifact, "u": u, "u0": u0, "hand": hand, "subject": subject}


def test_the_two_point_diffusion_is_kappa_times_the_shared_laplacian(diffusion, tmp_path):
    field = np.random.default_rng(20261111).standard_normal(CELLS)
    simulation = surface.bind(diffusion["artifact"], data={diffusion["u0"]: field},
                              directory=tmp_path)
    result = simulation.run(until=DT)
    assert result.steps == 1 and result.time == DT
    two_point_rate = (result.state[diffusion["u"]] - field) / DT
    runtime = pops.bind(diffusion["hand"], initial_values={
        diffusion["subject"]: np.ascontiguousarray(np.transpose(field)[np.newaxis, ...])})
    pops.run(runtime, t_end=DT, max_steps=1, console=False)
    reached = np.asarray(runtime.state_global("laplacian")).reshape((1,) + CELLS[::-1])
    laplacian_rate = (np.transpose(reached[0]) - field) / DT
    scale = float(np.max(np.abs(_stencil(field))))
    # two steps of the same size from the same field: equal to round-off of the update
    assert np.max(np.abs(two_point_rate - laplacian_rate)) <= 1e-10 * scale
    assert np.max(np.abs(two_point_rate - _stencil(field))) <= 1e-10 * scale
    assert np.max(np.abs(laplacian_rate - _stencil(field))) <= 1e-10 * scale
