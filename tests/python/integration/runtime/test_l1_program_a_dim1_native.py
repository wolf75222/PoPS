"""L1 programme A on the periodic interval (spec 0.3 §17.1, ROM-112)."""

from fractions import Fraction
import math

import numpy as np
import pytest

import pops.surface as surface
from pops._native_selector import select_native_dimension
from pops.lib.numerics import two_point, upwind
from pops.lib.time.methods import ssprk2
from pops.runtime._simulation import FaceFrequencyStop
from pops.surface.math import ddt, div, grad

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CELLS = 256
SPEED, NU, END = 1.0, 1e-2, 1.0
EPS = 2.0 ** -52
WIDTH = 1.0 / CELLS
FREQUENCY = (Fraction(abs(SPEED)) / Fraction(WIDTH)
             + 2 * Fraction(NU) / Fraction(WIDTH) ** 2)
DT = float(1 / FREQUENCY)
ONE_STEP_ATOL = 8 * EPS
FINAL_ATOL = 4e-13


def _compile(factor):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=SPEED)
    nu = model.parameter("nu", default=NU, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})

    def combined_limit(cell):
        h = cell.width(domain.x)
        return factor * (1 / (abs(a) / h + 2 * nu / h**2))

    program = surface.Program(
        advance=surface.Advance(ssprk2(rate=model), dt=combined_limit, attempts=2),
        initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    artifact = surface.compile(case, target=surface.OpenMP())
    return artifact, u, nu, u0


@pytest.fixture(scope="module")
def variants(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    native = select_native_dimension(1)
    assert native.__native_dimension__ == 1
    assert native.__has_kokkos__
    cache = tmp_path_factory.mktemp("pops-native-cache-l1")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        yield tuple(_compile(1 + margin * EPS) for margin in (0, 16, 64))


def _initial():
    centres = (np.arange(CELLS, dtype=np.float64) + 0.5) * WIDTH
    return np.where(np.abs(centres - 0.5) < 0.25, 1.0, 0.0)


def _bind(compiled, path, initial):
    artifact, _, nu, u0 = compiled
    return surface.bind(artifact, params={nu: NU}, data={u0: initial}, directory=path)


def _attempt_duration(simulation):
    # Read the generated proposal from the bound native System before the attempt.
    # The typed face refusal reports its frequency, but not the attempted duration.
    system = simulation._simulation._runtime._executor._s
    return system._step_bound(system._step_bound_identity())


def _margin(dt):
    return float((Fraction(dt) * FREQUENCY - 1) / Fraction(EPS))


def _rhs(values):
    left = values - np.roll(values, 1)
    right = np.roll(values, -1) - values
    return -(SPEED / WIDTH) * left + (NU / WIDTH**2) * (right - left)


def _ssprk2(values, dt):
    predictor = values + dt * _rhs(values)
    return (values + predictor + dt * _rhs(predictor)) / 2


def _check_state(state, expected, atol):
    values = np.asarray(state)
    assert values.dtype == np.float64 and values.shape == (CELLS,)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, expected, rtol=0, atol=atol)
    return float(np.max(np.abs(values - expected)))


def _one_accepted_step(simulation, u, dt, initial):
    with pytest.raises(surface.BudgetExhausted) as caught:
        simulation.run(until=END, max_steps=1)
    result = caught.value.result
    assert result.failure.payload["budget"] == "max_steps"
    assert result.steps == 1 and result.rejections == 0 and result.time == dt
    return _check_state(result.state[u], _ssprk2(initial, dt), atol=ONE_STEP_ATOL)


def test_l1_program_a_loses_no_attempt_on_the_interval(variants, tmp_path, record_property):
    nominal, _, _ = variants
    _, u, _, _ = nominal
    initial = _initial()

    one = _bind(nominal, tmp_path / "one-step", initial)
    dt = _attempt_duration(one)
    assert abs(dt - DT) <= math.ulp(DT)
    assert abs(_margin(dt)) < 32
    record_property("first_step_error", _one_accepted_step(one, u, dt, initial))

    result = _bind(nominal, tmp_path / "full-run", initial).run(until=END, max_steps=10_000)
    steps = math.ceil(END / DT)
    assert result.failure is None and result.time == END and result.rejections == 0
    assert result.steps == steps == 1567

    expected = initial
    for _ in range(steps - 1):
        expected = _ssprk2(expected, DT)
    # The constant-step segment anchors time at the product step * DT.
    final_dt = END - (steps - 1) * DT
    expected = _ssprk2(expected, final_dt)
    error = _check_state(result.state[u], expected, atol=FINAL_ATOL)
    native = select_native_dimension(1).runtime_environment_report()
    assert native["dimension"] == 1 and native["real_bytes"] == 8
    record_property("native_backend", native["kokkos_backend"])
    record_property("native_device", native["kokkos_device"])
    record_property("dt", dt.hex())
    record_property("dt_frequency_margin_eps", _margin(dt))
    record_property("oracle_final_dt", final_dt.hex())
    record_property("final_error", error)


def test_l1_guard_admits_the_trial_inside_its_roundoff_margin(variants, tmp_path, record_property):
    _, admitted, _ = variants
    _, u, _, _ = admitted
    initial = _initial()
    simulation = _bind(admitted, tmp_path, initial)
    dt = _attempt_duration(simulation)
    margin = _margin(dt)
    assert 0 < margin < 32
    record_property("dt_frequency_margin_eps", margin)
    record_property("first_step_error", _one_accepted_step(simulation, u, dt, initial))


def test_l1_guard_refuses_the_trial_beyond_its_roundoff_margin(variants, tmp_path, record_property):
    _, _, refused = variants
    initial = _initial()
    simulation = _bind(refused, tmp_path, initial)
    dt = _attempt_duration(simulation)
    margin = _margin(dt)
    assert 32 < margin < 96

    with pytest.raises(FaceFrequencyStop, match="pops.face-step-frequency@1") as caught:
        simulation.run(until=END)
    failure = caught.value
    result = failure.result
    assert result.steps == 0 and result.time == 0.0 and result.rejections == 1
    assert result.continuation.macro_step == 0
    assert result.diagnostics["native"]["attempts"] == 1
    assert abs(failure.frequency - float(FREQUENCY)) <= 8 * EPS * float(FREQUENCY)
    block, = result.state
    accepted = np.asarray(result.state[block])
    assert accepted.dtype == np.float64 and accepted.size == CELLS
    assert accepted.tobytes() == initial.tobytes()
    record_property("refused_dt", dt.hex())
    record_property("dt_frequency_margin_eps", margin)
    record_property("guard_frequency", failure.frequency.hex())
