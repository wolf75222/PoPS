"""Lot B of N15 (ROM-98, OD31), native: a case of the surface with boundary relations, run.

An advection on the box [0, 1] x [0, 1] with no periodic axis, ``ddt(u) == -div(vector(1, 1/4)
* u)``, ``rusanov()`` and the SSPRK2 of ``pops.lib.time.methods``, at level 1 on 16 x 8 cells.
The inflow pieces x_min and y_min take ``u == g`` with the runtime parameter g, the outflow
pieces x_max and y_max take ``outflow(u)``, and ``pops.Numerics(boundary=ghosts())`` realizes
them (2.6.2, 2.6.9, D9). ``pops.compile``, ``pops.bind`` and ``simulation.run`` give, bit for bit,
the state that the same case written by hand for b6 and N8, with the ``TransportBoundarySet`` of
``Inflow`` and ``Outflow``, reaches through ``pops.bind`` and ``Simulation.run``, with g bound
and with its default; the inflow brings g into the box.

G04-a (E3): the surface fills its ghost cells from the body of the policy (``SourceGhost``, the
``UserGhostPolicy`` of the model package), the hand case with the native laws ``FixedState`` and
``Extrapolate``. The equality is the proof that the generated ghosts are the native ones: at the
depth 1, at the depths 1 and 2 under ``muscl(limiter=vanleer)`` and with a periodic axis. A
policy outside the laws of b6 runs as the hand case with ``SourceGhost``, a native face is
refused beside a source ghost face (C2), and a ghost that is not finite is refused by the face
that reads it, with the cause of that face (D5, R3).
"""
from pathlib import Path

import numpy as np
import pytest

import pops
import pops.surface as surface
from examples.api_v02 import b6_numerics, b6_reconstructions
from examples.api_v02.common import state_array
from examples.api_v02.models_M01_M14 import _domain, _euler, _initial as bind_initial_array, _ssprk2
from pops import math as pm
from pops.analytic import time
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow, SourceGhost
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics.boundaries import ghosts
from pops.lib.numerics import muscl, rusanov, vanleer
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.surface.discrete import Ghost, forms
from pops.surface.math import ddt, div, outflow, sqrt, vector
from pops.time import Clock, FixedDt, SampleAndHold, TimePoint
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 8)
G = 0.2                         # the bound value of the runtime parameter g (default 0.05)
UNTIL = 0.05                    # 50 steps of dt = 1e-3


def _surface_case(scheme=None, policy=None, periodic_x=False, *, dt=1e-3, temporal=False):
    x = surface.Periodic(0, 1) if periodic_x else (0.0, 1.0)
    domain = surface.Box(x=x, y=(0.0, 1.0))
    sides = domain.boundary
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    g = model.parameter("g", default=0.05)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    inlet = g + model.time if temporal else g
    if periodic_x:
        model.on(sides.y_min, u == inlet)
        model.on(sides.y_max, outflow(u))
    else:
        model.on([sides.x_min, sides.y_min], u == inlet)
        model.on([sides.x_max, sides.y_max], outflow(u))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: scheme or rusanov()},
                                boundary=policy or ghosts())
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=dt),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, g, u0


@method
def _n8_ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    first = u + h * rate(u, t)
    return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)


def _native_laws(state, g, depth):
    del depth
    return Inflow(state=state, value=g), Outflow(state=state)


def _native_face(flux_u, u):
    return FiniteVolume(flux=flux_u, variables=variables.Conservative(u),
                        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())


def _user(body, name="muscl"):
    """A reconstruction of the cell written for b6 over the samples -1, 0 and 1, as the surface
    realizes a ``Reconstruct``."""
    def recon(u):
        def sampled(sample):
            minus, centre, plus = sample(-1)[0], sample(0)[0], sample(1)[0]
            return (body(minus, centre, plus),)

        return reconstruction.User(sampled, state=u, formal_order=3, name=name)

    return recon


_hand_vanleer = _user(lambda minus, centre, plus: centre + .5 * b6_reconstructions.vanleer(
    centre - minus, plus - centre))


def _vanleer_face(flux_u, u):
    """The face of ``rusanov(reconstruction=muscl(limiter=vanleer))``, written for b6."""
    return b6_numerics.rusanov(flux_u, u, reconstruction=_hand_vanleer(u))


def _hand_case(face=_native_face, laws=_native_laws, depth=1, periodic_x=False, *, compose=None):
    """The same case written for b6 and N8: ``face(flux_u, u)`` is the method of its rate and
    ``laws(state, g, depth)`` the conditions of a value and of no data."""
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
    plan.rates.add(selection, face(flux_u, u))
    case = pops.Case("advection")
    block = case.block("advection", model)
    boundaries = frame.boundaries
    value, no_data = laws(block[u], g, depth)
    periodic = PeriodicAxes((x,)) if periodic_x else None
    sides = {boundaries.y_min: value, boundaries.y_max: no_data}
    if not periodic_x:
        sides.update({boundaries.x_min: value, boundaries.x_max: no_data})
    plan.boundaries.add(TransportBoundarySet(sides, periodic=periodic))
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    if compose is None:
        elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
                  state=block[u], proposal=Proposal(0.001))
    else:
        compose(case, block[u], selection)
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=periodic))
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


def _mirror(state, g, depth):
    """The ghosts of ghosts() written for b6 with ``SourceGhost``: 2 g - u_k and u_1."""
    return (SourceGhost(state=state, body=lambda inner, k: (2 * g - inner(k)[0],), depth=depth),
            SourceGhost(state=state, body=lambda inner, k: (inner(1)[0],), depth=depth))


def _extrapolated(state, g, depth):
    """2 g - u_k for a value and the linear extrapolation 2 u_1 - u_2 for no data."""
    value, _ = _mirror(state, g, depth)
    return value, SourceGhost(state=state, body=lambda inner, k: (2 * inner(1)[0] - inner(2)[0],),
                              depth=depth)


@surface.scheme
def _linear(face, term, depth):
    q = term.quantity
    if term.form is forms.Value:
        return Ghost(2 * term.data - face.inner(depth).value(q))
    return Ghost(2 * face.inner(1).value(q) - face.inner(2).value(q))


@surface.scheme
def _not_finite(face, term, depth):
    """A ghost that is not finite where u < 1, everywhere in this run."""
    q = term.quantity
    if term.form is forms.Value:
        return Ghost(2 * term.data - face.inner(depth).value(q))
    return Ghost(sqrt(face.inner(1).value(q) - 1))


_VARIANTS = {
    "first order": (lambda: _surface_case(), lambda: _hand_case()),
    "muscl": (lambda: _surface_case(scheme=rusanov(reconstruction=muscl(limiter=vanleer))),
              lambda: _hand_case(face=_vanleer_face, depth=2)),
    "periodic x": (lambda: _surface_case(periodic_x=True), lambda: _hand_case(periodic_x=True)),
    "linear extrapolation": (
        lambda: _surface_case(scheme=rusanov(reconstruction=muscl(limiter=vanleer)),
                              policy=_linear()),
        lambda: _hand_case(face=_vanleer_face, laws=_extrapolated, depth=2)),
}


@pytest.fixture(scope="module")
def native_cache(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        yield {}


def _compiled(native_cache, variant):
    if variant not in native_cache:
        surface_case, hand_case = _VARIANTS[variant]
        case, u, g, u0 = surface_case()
        artifact = surface.compile(case, target=surface.OpenMP())
        hand, layout, g_handle = hand_case()
        compiled = pops.compile(pops.resolve(pops.validate(hand), layout=layout))
        native_cache[variant] = {"u": u, "g": g, "u0": u0, "artifact": artifact,
                                 "hand": compiled, "g_handle": g_handle}
    return native_cache[variant]


def _hand_state(compiled, params):
    hand = compiled["hand"]
    (binding,) = tuple(hand.plan.initial_condition_plan.bindings)
    initial = np.ascontiguousarray(np.transpose(_initial())[np.newaxis, ...])
    runtime = pops.bind(hand, initial_values={binding.subject: initial},
                        params={compiled["g_handle"]: params[0]} if params else {})
    result = Simulation(runtime).run(UNTIL)
    assert result.failure is None
    return np.transpose(np.asarray(result.state["advection"])[0])


def _surface_run(compiled, tmp_path, bound):
    simulation = surface.bind(compiled["artifact"],
                              params={compiled["g"]: bound[0]} if bound else {},
                              data={compiled["u0"]: _initial()}, directory=tmp_path)
    return simulation.run(until=UNTIL)


@pytest.mark.parametrize("bound", [(G,), ()], ids=["g bound", "g default"])
def test_inflow_and_outflow_run_as_the_hand_written_case(native_cache, tmp_path, bound):
    compiled = _compiled(native_cache, "first order")
    u = compiled["u"]
    result = _surface_run(compiled, tmp_path, bound)
    assert result.failure is None and result.time == UNTIL and result.steps == 50
    final = result.state[u]
    assert final.shape == CELLS
    _bitwise(final, _hand_state(compiled, bound))
    # the inflow brings g into the corner cell (x_0, y_0), where u0 is about 1e-7
    g = bound[0] if bound else 0.05
    assert 0.3 * g < final[0, 0] < g


@pytest.mark.parametrize("variant", ["muscl", "periodic x", "linear extrapolation"])
def test_the_generated_ghosts_run_as_the_hand_written_case(native_cache, tmp_path, variant):
    """muscl reads the ghosts at the depths 1 and 2 (native FixedState and Extrapolate); the
    periodic axis leaves x to the halo exchange; the linear extrapolation is no law of b6, and
    its oracle is the hand case with the same ``SourceGhost``."""
    compiled = _compiled(native_cache, variant)
    result = _surface_run(compiled, tmp_path, (G,))
    assert result.failure is None and result.time == UNTIL and result.steps == 50
    _bitwise(result.state[compiled["u"]], _hand_state(compiled, (G,)))


def test_a_native_face_beside_a_source_ghost_face_is_refused(native_cache):
    """C2: a source ghost face is filled by the generated System block, whose face is the
    generated face of the package; the native riemann.Rusanov() is refused by name."""
    del native_cache
    hand, layout, _ = _hand_case(laws=_mirror)
    with pytest.raises(ValueError, match=r"needs the generated face of its package"):
        pops.compile(pops.resolve(pops.validate(hand), layout=layout))


def test_a_ghost_that_is_not_finite_is_refused_by_the_face_that_reads_it(native_cache, tmp_path):
    """D5, R3: the fill writes the ghost as the native laws do, without a test; the User face
    that reads it observes its intermediates and refuses the step (its cause kUserNonFiniteFlux:
    no other stage of this block tests a value), not the fill."""
    case, u, g, u0 = _surface_case(policy=_not_finite())
    del u, g
    artifact = surface.compile(case, target=surface.OpenMP())
    simulation = surface.bind(artifact, data={u0: _initial()}, directory=tmp_path)
    try:
        result = simulation.run(until=UNTIL)
    except Exception as exc:  # noqa: BLE001 - the refusal may be raised or returned
        message = str(exc)
    else:
        assert result.failure is not None
        message = str(result.failure)
    assert "hyperbolic face evaluation refused publication" in message


def _advance_line(initial, dt, rhs, sample=lambda begin, stage, dt: begin + stage * dt, *, steps=2):
    """Independent first-order spatial line and two SSPRK2 evaluations per step."""
    state = initial
    for tick in range(steps):
        begin = tick * dt
        first = state + dt * rhs(state, sample(begin, 0, dt))
        state = 0.5 * state + 0.5 * (first + dt * rhs(first, sample(begin, 1, dt)))
    return state


def _check_time_oracle(initial, dt, rhs, *, subcycles=1):
    expected = _advance_line(initial, dt, rhs, steps=2 * subcycles)
    for wrong_sample in (lambda begin, stage, dt: 0.,
                         lambda begin, stage, dt: begin,
                         lambda begin, stage, dt: float(stage)):
        wrong = _advance_line(initial, dt, rhs, wrong_sample, steps=2 * subcycles)
        assert np.max(np.abs(expected - wrong)) > 1e-5
    if subcycles > 1:
        def macro_time(begin, stage, duration):
            interval = duration * subcycles
            return (begin // interval) * interval + stage * interval

        wrong = _advance_line(initial, dt, rhs, macro_time, steps=2 * subcycles)
        assert np.max(np.abs(expected - wrong)) > 1e-5
    return expected


def test_generated_scalar_ghost_reads_each_physical_stage_time(native_cache, tmp_path):
    del native_cache
    native_profile(scope="G04 generated scalar stage clock")
    dt = 1 / 128
    case, u, g, u0 = _surface_case(periodic_x=True, dt=dt, temporal=True)
    initial = np.ones(CELLS)
    artifact = surface.compile(case, target=surface.OpenMP())
    simulation = surface.bind(artifact, params={g: 1.}, data={u0: initial}, directory=tmp_path)
    result = simulation.run(until=2 * dt)
    assert result.failure is None and result.time == 2 * dt and result.steps == 2

    def rhs(state, physical_time):
        # Constant positive speed makes first-order Rusanov upwind. The y-min ghost is
        # the reflection across the prescribed face value, g + t. x is periodic and uniform.
        upstream = np.r_[2 * (1 + physical_time) - state[0], state[:-1]]
        return 0.25 * CELLS[1] * (upstream - state)

    expected = _check_time_oracle(initial[0], dt, rhs)
    _bitwise(result.state[u], np.broadcast_to(expected, CELLS))


def _ssprk2_on_child(case, subject, rate, program, child, dt):
    parent = program.state(subject)
    local = program.state(subject, clock=child)
    first = program.synchronize(parent.n, at=TimePoint(child), relation=SampleAndHold())

    def advance(builder, state):
        predictor = builder.value("child_predictor", state + builder.dt * rate(state),
                                  at=TimePoint(child, 1))
        return builder.value("child_accepted", .5 * state + .5 * (predictor + builder.dt * rate(predictor)),
                             at=local.next.point)

    last = program.subcycle(first, clock=child, within=program.clock, count=2, body_fn=advance)
    accepted = program.synchronize(last, at=parent.next.point, relation=SampleAndHold())
    program.commit(parent.next, accepted)
    program.step_strategy(FixedDt(dt))
    case.program(program)


def test_subcycled_scalar_ghost_reads_its_child_stage_clock(native_cache):
    del native_cache
    native_profile(scope="G04 subcycled scalar stage clock")
    dt = 1 / 128
    program = pops.Program("advection_clocked_child")
    child = Clock("child", owner=program.owner_path)

    def laws(state, g, depth):
        del g
        return (SourceGhost(state=state, depth=depth,
                            body=lambda inner, k: (1 + time(child),)),
                SourceGhost(state=state, depth=depth, body=lambda inner, k: (inner(1)[0],)))

    case, layout, _ = _hand_case(
        face=b6_numerics.rusanov, laws=laws, periodic_x=True,
        compose=lambda case, subject, rate: _ssprk2_on_child(case, subject, rate, program, child, dt),
    )
    line = 1 + np.arange(CELLS[1]) / 256
    initial = np.broadcast_to(line[:, None], CELLS[::-1])[np.newaxis, ...].copy()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    artifact.verify()
    (binding,) = tuple(artifact.plan.initial_condition_plan.bindings)
    runtime = pops.bind(artifact, initial_values={binding.subject: initial})
    report = pops.run(runtime, t_end=2 * dt, max_steps=2, console=False)
    assert report.accepted_steps == 2 and runtime.time() == 2 * dt and runtime.macro_step() == 2

    def rhs(state, physical_time):
        return .25 * CELLS[1] * (np.r_[1 + physical_time, state[:-1]] - state)

    expected = _check_time_oracle(line, dt / 2, rhs, subcycles=2)
    actual = state_array(runtime, "advection", initial.shape)
    _bitwise(actual, np.broadcast_to(expected[:, None], initial.shape))


def test_generated_euler_ghost_reads_each_declared_stage_time(native_cache):
    del native_cache
    native_profile(scope="G04 generated Euler stage clock")
    _euler_ghost_trajectory(subcycled=False)


def test_subcycled_euler_ghost_reads_its_child_stage_clock(native_cache):
    del native_cache
    native_profile(scope="G04 subcycled Euler stage clock")
    _euler_ghost_trajectory(subcycled=True)


def _euler_ghost_trajectory(*, subcycled):
    dt = 1 / 128
    frame, layout = _domain("M03_clocked_boundary", CELLS, periodic=(1,))
    model = pops.Model("M03_clocked_boundary", frame=frame)
    gamma, pressure_offset, vx, vy = 1.4, 0.2, 0.3, 0.1
    state, flux, rate, _ = _euler(model, frame, gamma=gamma, pressure_offset=pressure_offset)
    case = pops.Case("M03_clocked_boundary")
    block = case.block("fluid", model)
    subject = block[state]
    if subcycled:
        program = pops.Program("M03_clocked_boundary_SSPRK2_child")
        clock = Clock("child", owner=program.owner_path)
        _ssprk2_on_child(case, subject, rate, program, clock, dt)
    else:
        program, _ = _ssprk2(case, block, state, lambda p, u, point, label: rate(u), dt,
                              name="M03_clocked_boundary_SSPRK2")
        clock = program.clock
    plan = DiscretizationPlan()
    plan.rates.add(rate, b6_numerics.rusanov(flux, state))
    energy_offset = (1 + gamma * pressure_offset) / (gamma - 1)
    density = 1 + time(clock)
    inlet = SourceGhost(
        state=subject, depth=1,
        body=lambda inner, k: (density, vx * density, vy * density, energy_offset + 0.05 * density))
    outlet = SourceGhost(
        state=subject, depth=1, body=lambda inner, k: tuple(inner(1)[j] for j in range(4)))
    plan.boundaries.add(TransportBoundarySet(
        {frame.boundaries.x_min: inlet, frame.boundaries.x_max: outlet},
        periodic=layout.mesh.periodic))
    case.numerics(plan, block=block)
    initial = np.stack([np.full(CELLS[::-1], value) for value in (1., vx, vy, energy_offset + 0.05)])
    if subcycled:
        initial = _euler_entropy_wave(1 + np.arange(CELLS[0]) / 512, vx, vy, energy_offset)
    initial_values = bind_initial_array(case, subject, initial)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    artifact.verify()
    runtime = pops.bind(artifact, initial_values=initial_values)
    report = pops.run(runtime, t_end=2 * dt, max_steps=2, console=False)
    assert runtime.time() == 2 * dt and runtime.macro_step() == 2, report

    def rhs(rho, physical_time):
        # The full Euler flux on the constant-pressure/velocity entropy manifold reduces
        # to the Rusanov mass flux. The four components then follow from this density.
        states = np.r_[1 + physical_time, rho, rho[-1]]
        left, right = states[:-1], states[1:]
        speeds = vx + np.sqrt(gamma * (1 + pressure_offset) / np.minimum(left, right))
        mass_flux = 0.5 * vx * (left + right) - 0.5 * speeds * (right - left)
        return -np.diff(mass_flux) * CELLS[0]

    children = 2 if subcycled else 1
    rho = _check_time_oracle(initial[0, 0], dt / children, rhs, subcycles=children)
    expected = _euler_entropy_wave(rho, vx, vy, energy_offset)
    actual = state_array(runtime, "fluid", initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    pressure = ((gamma - 1) * (actual[3] - (actual[1] ** 2 + actual[2] ** 2) / (2 * actual[0]))
                - gamma * pressure_offset)
    np.testing.assert_allclose(pressure, 1., rtol=0, atol=2e-10)


def _euler_entropy_wave(density, vx, vy, energy_offset):
    rho = np.broadcast_to(density, CELLS[::-1])
    return np.stack((rho, vx * rho, vy * rho, energy_offset + 0.05 * rho))
