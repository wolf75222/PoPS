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
from pops import math as pm
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
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 8)
G = 0.2                         # the bound value of the runtime parameter g (default 0.05)
UNTIL = 0.05                    # 50 steps of dt = 1e-3


def _surface_case(scheme=None, policy=None, periodic_x=False):
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
    if periodic_x:
        model.on(sides.y_min, u == g)
        model.on(sides.y_max, outflow(u))
    else:
        model.on([sides.x_min, sides.y_min], u == g)
        model.on([sides.x_max, sides.y_max], outflow(u))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: scheme or rusanov()},
                                boundary=policy or ghosts())
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


def _hand_case(face=_native_face, laws=_native_laws, depth=1, periodic_x=False):
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
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=Proposal(0.001))
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
