"""N8 « dt= as a function of cell » (ROM-89), native: the compiled step bound of a cells rule
(design note ``n8/cell_step_rule/design_note.md`` rev 3, D2, D4, D6; review rev 3, P4).

The slice's physics, ``ddt(u) == -div(vector(1, 1/4) * u)`` on [0, 1] x [0, 1] with inflow and
outflow, SSPRK2 and ``rusanov()`` at first order, under ``dt=`` a function of ``cell``:

1. a function written by hand that writes ``cfl_limit``'s bound runs the same steps and state, bit
   for bit, as ``cfl_limit`` (the bound and its identity are the same; only the declared budget
   differs, which the face guards read and these steps do not reach);
2. a function twice too large, and ``cfl_limit`` beside a face that records twice the wave speed
   (``upwind(speed=2 a.n)``, ROM-108): the compiled run stops by name at the first refused attempt
   (``FaceFrequencyStop``, ``kFaceFrequency``) with the faces' frequency F; ``pops.run`` on the same
   artifact retries once at fraction / F, the step that master's ``AdaptiveCFL`` retries at;
3. L1's program A (spec 17.1: advection-diffusion, ``upwind()`` and ``two_point()``,
   ``dt=combined_limit``) on the square, as N3 lot 1 elaborates L1 in two dimensions
   (``test_stencil_adaptive_cfl_native.py``): 0 attempts lost on the compiled route, and the
   margin of the step against the guard's frontier, ``dt F / phi - 1`` in ulp, under its 32 eps
   tolerance;
4. a cell at rest in a moving state (Burgers along x) bounds nothing: no failure, with and
   without ``max_dt``; a state at rest everywhere without ``max_dt`` steps to the frontier
   (2.6.9). A NaN speed fails by name at its cell (C4).

The module runs in the Dim=2 process; L1 on the interval is a Dim=1 case of its own.

The slice to ``T_END`` and to three frontiers, bit for bit against #788's compiled stability rule
and master's ``AdaptiveCFL``, is ``test_surface_lot_d_step_rule_native.py``.
"""
from fractions import Fraction
import math

import numpy as np
import pytest

import pops
import pops.surface as surface
from pops.lib.numerics import rusanov, two_point, upwind
from pops.lib.numerics.boundaries import ghosts
from pops.lib.time import cfl_limit
from pops.lib.time.methods import ssprk2
from pops.runtime._simulation import FaceFrequencyStop
from pops.surface.math import ddt, div, dot, grad, largest, outflow, smallest, vector, where
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CELLS = (32, 32)
AX, AY = 1.0, 0.25
FAR_FIELD = 0.05
CFL, MAX_DT, T_END = 0.45, 1.0e-2, 0.10
EPS = 2.0 ** -52


def _slice(dt, scheme=None):
    """The slice's case at first order with ``dt``; ``scheme`` takes the flux."""
    domain = surface.Box(x=(0.0, 1.0), y=(0.0, 1.0))
    sides = domain.boundary
    model = surface.Model("scalar_advection")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(AX, AY) * u)
    model.balance(ddt(u) == -div(advection))
    model.on([sides.x_min, sides.y_min], u == FAR_FIELD)
    model.on([sides.x_max, sides.y_max], outflow(u))
    numerics = surface.Numerics(
        represent={u: surface.CellAverage()},
        schemes={advection: rusanov() if scheme is None else scheme(advection)},
        boundary=ghosts())
    rule = dt(domain, advection)
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=rule),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return surface.compile(case, target=surface.OpenMP()), u, u0


def _cfl_limit(domain, advection):
    return cfl_limit(advection, cfl=CFL, max_dt=MAX_DT)


def _by_hand(domain, advection):
    """cfl_limit's bound written by hand: the same expression, the budget of a fixed step."""
    def limit(cell):
        s = largest([cell.speed(advection, domain.x), cell.speed(advection, domain.y)])
        w = smallest([cell.width(domain.x), cell.width(domain.y)])
        return smallest([where(s > 0, (CFL * w) / s, MAX_DT), MAX_DT])
    return limit


def _twice(domain, advection):
    """A bound twice the stable step of the faces, under the budget of a fixed step."""
    def twice(cell):
        s = largest([cell.speed(advection, domain.x), cell.speed(advection, domain.y)])
        return (2 * smallest([cell.width(domain.x), cell.width(domain.y)])) / s
    return twice


def _guarded(advection):
    """ROM-108's face: upwind() whose stated speed is twice the wave speed a.n."""
    return upwind(speed=lambda face: 2 * dot(vector(AX, AY), face.normal))


def _initial():
    x = (np.arange(CELLS[0]) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1]) + 0.5) / CELLS[1]
    return FAR_FIELD + 0.95 * np.exp(-120.0 * ((x[:, None] - 0.30) ** 2
                                               + (y[None, :] - 0.35) ** 2))


def _bitwise(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


@pytest.fixture(scope="module")
def slices(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        yield {"cfl_limit": _slice(_cfl_limit), "by_hand": _slice(_by_hand),
               "twice": _slice(_twice), "guarded": _slice(_cfl_limit, _guarded)}


def _simulation(compiled, tmp_path):
    artifact, _, u0 = compiled
    return surface.bind(artifact, data={u0: _initial()}, directory=tmp_path)


def _runtime(compiled):
    """The surface's artifact bound for ``pops.run``, with the slice's initial state."""
    hand = compiled[0].compiled
    (binding,) = tuple(hand.plan.initial_condition_plan.bindings)
    initial = np.ascontiguousarray(np.transpose(_initial())[np.newaxis, ...])
    return pops.bind(hand, initial_values={binding.subject: initial},
                     resources={"execution_context": artifact_execution_context(hand)})


def test_a_function_written_by_hand_runs_as_cfl_limit(slices, tmp_path):
    ruled = _simulation(slices["cfl_limit"], tmp_path / "ruled").run(until=T_END)
    hand = _simulation(slices["by_hand"], tmp_path / "hand").run(until=T_END)
    assert ruled.failure is None and hand.failure is None
    assert (ruled.time, hand.time) == (T_END, T_END)
    assert ruled.steps == hand.steps and (ruled.rejections, hand.rejections) == (0, 0)
    # 0.45 / 32 / 1.0 = 0.0140625 is above max_dt: ten steps of 1e-2
    assert ruled.steps == 10
    _bitwise(ruled.state[slices["cfl_limit"][1]], hand.state[slices["by_hand"][1]])


@pytest.mark.parametrize("name, fraction", [("twice", 1.0), ("guarded", CFL)])
def test_a_refused_face_stops_the_compiled_run_and_pops_run_retries(slices, tmp_path, name,
                                                                   fraction):
    """K1-c and C1: the compiled run stops by name at the first refused attempt, with the faces'
    frequency F; pops.run retries once at fraction / F, as master's AdaptiveCFL retries at
    cfl / F (ROM-108), and lands on its frontier."""
    with pytest.raises(FaceFrequencyStop) as caught:
        _simulation(slices[name], tmp_path).run(until=T_END)
    stop = caught.value
    assert stop.result.steps == 0 and stop.result.time == 0.0
    assert math.isfinite(stop.frequency) and stop.frequency > 0.0
    assert "pops.face-step-frequency@1" in str(stop)

    runtime = _runtime(slices[name])
    report = pops.step(runtime, t_bound=T_END, console=False)
    assert (report.accepted_steps, report.rejected_steps) == (1, 1)
    assert float(runtime.time()) == fraction / stop.frequency

    runtime = _runtime(slices[name])
    pops.run(runtime, t_end=T_END, max_steps=10_000, console=False)
    assert float(runtime.time()) == T_END


# --- L1, program A (spec 17.1): the exit case under the guard's 32 eps -----------------------------

L1_CELLS, NU, L1_END = 64, 1e-3, 0.05


@pytest.fixture(scope="module")
def l1(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
        model = surface.Model("advection_diffusion")
        u = model.quantity("u", surface.Scalar(domain))
        a = model.constant("a", value=1.0)
        nu = model.parameter("nu", default=NU)
        advection = model.flux(of=u, value=vector(a, a) * u)
        diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
        model.balance(ddt(u) == -div(advection + diffusion))
        numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                    schemes={advection: upwind(), diffusion: two_point()})

        def combined_limit(cell):
            # c_a + 2 d r_d <= 1 on the square: upwind's |a| / h on the largest axis and the
            # four faces of two_point(), 4 nu / h^2
            h = cell.width(domain.x)
            return 1 / (abs(a) / h + 4 * nu / h**2)

        program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=combined_limit),
                                  initial={u: 0.5})
        case = surface.Case(model=model, numerics=numerics, program=program,
                            layout={domain: surface.Uniform(cells=(L1_CELLS, L1_CELLS))})
        yield surface.compile(case, target=surface.OpenMP()), u, nu


def test_l1_program_a_loses_no_attempt_under_the_guard(l1, tmp_path, record_property):
    artifact, u, nu = l1
    result = surface.bind(artifact, params={nu: NU}, directory=tmp_path).run(until=L1_END)
    assert result.failure is None and result.time == L1_END and result.rejections == 0
    # every step but the last is the bound E = 1 / (|a| / h + 4 nu / h^2), in binary64 as the
    # kernel writes it; F = |a| / h + 4 nu / h^2 is the guard's frontier (upwind's |a| / h on the
    # largest axis and the four faces of two_point()), and phi = 1 (a function by hand)
    h = 1.0 / L1_CELLS
    dt = 1.0 / (abs(1.0) / h + 4.0 * NU / h ** 2)
    exact = Fraction(dt) * (1 / Fraction(h) + 4 * Fraction(NU) / Fraction(h) ** 2)
    margin = float(exact - 1) / EPS
    record_property("l1_margin_ulp", margin)
    assert abs(margin) < 32.0
    assert result.steps == math.ceil(L1_END / dt) or result.steps == math.ceil(L1_END / dt) + 1


# --- a cell at rest (2.6.9, C4) ----------------------------------------------------------------------

REST_CELLS, REST_END = (64, 4), 0.05


def _burgers(max_dt):
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("burgers")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    flux = model.flux(of=u, value=vector(u * u / 2, 0 * u))
    model.balance(ddt(u) == -div(flux))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: rusanov()})
    program = surface.Program(
        advance=surface.Advance(ssprk2(rate=model), dt=cfl_limit(flux, cfl=CFL, max_dt=max_dt)),
        initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=REST_CELLS)})
    return surface.compile(case, target=surface.OpenMP()), u0


@pytest.fixture(scope="module")
def burgers(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        yield {max_dt: _burgers(max_dt) for max_dt in (None, MAX_DT)}


def _step_profile():
    x = (np.arange(REST_CELLS[0]) + 0.5) / REST_CELLS[0]
    return np.repeat(np.where(x < 0.5, 0.0, 1.0)[:, None], REST_CELLS[1], axis=1)


@pytest.mark.parametrize("max_dt", [None, MAX_DT])
def test_a_cell_at_rest_bounds_nothing(burgers, tmp_path, max_dt):
    artifact, u0 = burgers[max_dt]
    moving = surface.bind(artifact, data={u0: _step_profile()},
                          directory=tmp_path / "moving").run(until=REST_END)
    assert moving.failure is None and moving.time == REST_END and moving.rejections == 0
    # the first step is the moving cells' cfl h / 1, below max_dt
    assert moving.steps >= math.ceil(REST_END / (CFL / REST_CELLS[0]))
    rest = surface.bind(artifact, data={u0: np.zeros(REST_CELLS)},
                        directory=tmp_path / "rest").run(until=REST_END)
    assert rest.failure is None and rest.time == REST_END
    # at rest everywhere: a step to the frontier, or max_dt's steps (2.6.9)
    assert rest.steps == (1 if max_dt is None else math.ceil(REST_END / max_dt))


def test_a_speed_that_is_not_a_number_fails_by_name(burgers, tmp_path):
    """C4: the predicate's speed is an observed operation, so a NaN cell fails the step bound by
    name at its cell, though s > 0 is false there and the where takes the finite cap."""
    artifact, u0 = burgers[MAX_DT]
    initial = _step_profile()
    initial[7, 2] = np.nan
    with pytest.raises(Exception, match="step_bound_nonfinite") as caught:
        surface.bind(artifact, data={u0: initial}, directory=tmp_path).run(until=REST_END)
    assert "at the cell (7, 2)" in str(caught.value)


def test_a_negative_zero_divisor_fails_by_name(tmp_path, native_cxx, kokkos_root, monkeypatch):
    """Ruling 13/11 (b): E = w / -0 is -infinity, a typed failure (step_bound_nonfinite), never a
    step; only w / +0 = +infinity is admitted, at the root."""
    del native_cxx, kokkos_root
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("POPS_NATIVE_CACHE_DIR", str(tmp_path / "cache"))

    def negative_zero(domain, advection):
        def limit(cell):
            w = cell.width(domain.x)
            return w / -(0 * w)
        return limit

    compiled = _slice(negative_zero)
    with pytest.raises(Exception, match="step_bound_nonfinite") as caught:
        _simulation(compiled, tmp_path).run(until=T_END)
    assert "at the cell (0, 0)" in str(caught.value)


def test_a_poisson_source_is_refused_by_name(monkeypatch, isolated_native_cache, native_cxx,
                                             kokkos_root):
    """Ruling 13/11, 2 (C3): a block whose elliptic brick is a real Poisson source (M08, a fluid
    with its charge density) is refused by name by the run of a cells rule; the zero brick of the
    slice and of L1 is admitted (the tests above)."""
    del isolated_native_cache, native_cxx, kokkos_root
    import examples.api_v02.models_M01_M14 as early
    from pops.mesh._amr.refinement_criterion import CellWidth
    from pops.runtime._simulation import Simulation
    from pops.time._program.elaboration import Advance, Retry, elaborate
    from pops.time._step.bound import StepBound

    def route(case, block, state, flow, dt, *, name, memory, record, b6):
        del b6, dt
        return elaborate(Advance(early.recorded_ssprk2(memory, record)(flow=flow),
                                 retry=Retry(max_attempts=1)),
                         case=case, state=block[state], proposal=StepBound(CellWidth(0), 2),
                         name=name)

    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_recorded_ssprk2", route)
        spec = early.build_m08()
    artifact = pops.compile(pops.resolve(pops.validate(spec.case), layout=spec.layout))
    runtime = pops.bind(artifact, initial_values={
        subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)})
    with pytest.raises(RuntimeError, match=r"refuses a default Poisson solve \(the Poisson source "
                                           r"of an evolving block\)"):
        Simulation(runtime).run(0.01)
