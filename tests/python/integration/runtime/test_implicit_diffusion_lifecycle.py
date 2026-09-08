"""Declared M5.4 Uniform scalar matrix through the complete public lifecycle."""
from __future__ import annotations

import math

import numpy as np
import pops
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.math import ddt, div, grad, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import (
    DerivativeStrategy, FailRun, FixedDt, ImplicitDiffusionStage, SolveUnknown,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def make_case(n, *, method="backward_euler", dt=0.0001, nonlinear=False, invalid=False,
              tau_scale=1, derivative_route="finite_difference", commit_mode="solved"):
    from pops.numerics import Diffusion

    frame = Rectangle("implicit_heat_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("implicit_heat", frame=frame)
    state = model.state("U", components=("energy",))
    (energy,) = state
    temperature = model.primitive("temperature", (sqrt(1 + 4 * energy) - 1) / 2) \
        if nonlinear else energy
    flux = model.diffusive_flux("thermal_flux", state=state, value=0.1 * grad(temperature))
    if nonlinear:
        power = model.source("power", on=state, value=(1.0 + 0.0 * energy,))
        rate = model.rate("heat", equation=ddt(state) == div(flux) + power)
        accumulation = model.local_transform(
            "coordinate_to_energy", (energy + energy * energy,),
            valid_if=(energy > -0.5) & (energy < 1.5))
        seed_op = model.local_transform(
            "initial_temperature", (energy if invalid else 0.2 + 0.0 * energy,), valid_if=energy >= 0.0)
    else:
        rate = model.rate("heat", equation=ddt(state) == div(flux))
        accumulation, seed_op = None, None
    case = pops.Case("implicit_heat_case")
    block = case.block("material", model=model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = pops.Program("implicit_heat_" + method)
    temporal = program.state(block[state])
    if invalid:
        # This write occurs inside the attempted step before the failing residual. Every
        # ring slot and fill flag must return to the accepted envelope after FailRun.
        program.store_history("prior_energy", temporal.n, depth=1)
    if method == "forward_euler":
        candidate = program.value("updated", temporal.n + program.dt * rate(temporal.n),
                                  at=temporal.next.point)
    else:
        seed = temporal.n if seed_op is None else program.transform(temporal.n, transform=seed_op)
        coordinates = program.value("stage_coordinates", seed, at=temporal.next.point)
        stage = ImplicitDiffusionStage(rate, temporal.n, tau_scale * program.dt, accumulation=accumulation)
        request = stage.request(unknown=SolveUnknown("temperature" if nonlinear else "state", coordinates),
                                seed=seed, derivative=DerivativeStrategy(derivative_route))
        solved = program.solve(request, solver=Newton(
            tolerance=1e-12, max_iterations=20, linear_tolerance=1e-8,
            linear_max_iterations=100, restart=30)).consume(action=FailRun())[0]
        conserved = solved if accumulation is None else program.transform(solved, transform=accumulation)
        candidate = program.value("updated", conserved, at=temporal.next.point)
    if commit_mode == "history_only":
        program.store_history("uncommitted_solve", candidate, depth=1)
    if commit_mode in ("unused", "history_only"):
        candidate = program.value("unchanged", temporal.n, at=temporal.next.point)
    elif commit_mode == "weighted":
        candidate = program.value("blended", 0.5 * temporal.n + 0.5 * candidate,
                                  at=temporal.next.point)
    elif commit_mode == "raw_coordinate":
        candidate = solved
    elif commit_mode == "wrong_accumulation":
        candidate = program.transform(solved, transform=seed_op)
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes)))
    return case, layout


def run_case(n, initial, *, steps=1, **kwargs):
    case, layout = make_case(n, **kwargs)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    runtime = pops.bind(artifact, initial_state={"material": np.asarray(initial).reshape(1, n, n).copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    dt = kwargs.get("dt", 0.0001)
    report = pops.run(runtime, t_end=steps * dt, max_steps=steps)
    assert report.accepted_steps == steps
    return np.asarray(runtime.state_global("material")).reshape(n, n), runtime


def mode(n):
    x = (np.arange(n) + 0.5) / n
    return np.sin(2 * np.pi * x)[:, None] * np.sin(2 * np.pi * x)[None, :]


def test_same_physical_flux_explicit_implicit_fourier_parity(isolated_native_cache, native_cxx, kokkos_root):
    n, dt = 32, 0.0001
    wave = mode(n)
    eigenvalue = 8 * 0.1 * math.sin(math.pi / n) ** 2 * n ** 2
    for method, amplitude in (("forward_euler", 1 - dt * eigenvalue),
                              ("backward_euler", 1 / (1 + dt * eigenvalue))):
        actual, _ = run_case(n, 1 + wave, method=method, dt=dt)
        np.testing.assert_allclose(actual, 1 + amplitude * wave, rtol=0, atol=1e-9)


@pytest.mark.parametrize("method", ["forward_euler", "backward_euler"])
def test_scalar_diffusion_manufactured_convergence(method, isolated_native_cache, native_cxx, kokkos_root):
    errors = []
    for n, steps in ((16, 4), (32, 16), (64, 64)):
        wave = mode(n)
        actual, _ = run_case(n, 1 + wave, method=method, dt=0.005 * (16 / n) ** 2, steps=steps)
        exact = 1 + math.exp(-8 * math.pi ** 2 * 0.1 * 0.02) * wave
        errors.append(float(np.sqrt(np.mean((actual - exact) ** 2))))
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert math.log2(coarse / fine) >= 1.7, errors


def test_shifted_operator_retains_nonzero_constant(isolated_native_cache, native_cxx, kokkos_root):
    actual, _ = run_case(16, np.full((16, 16), 2.5), dt=0.125)
    np.testing.assert_allclose(actual, 2.5, rtol=0, atol=1e-11)


def test_nonlinear_accumulation_preserves_energy(isolated_native_cache, native_cxx, kokkos_root):
    energy, _ = run_case(16, np.zeros((16, 16)), dt=1.0, nonlinear=True)
    temperature = (np.sqrt(1 + 4 * energy) - 1) / 2
    np.testing.assert_allclose(energy, 1.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(temperature, (-1 + math.sqrt(5)) / 2, rtol=0, atol=1e-10)
    assert float(np.min(temperature)) > 0.6


def test_invalid_nonlinear_domain_leaves_accepted_state_unchanged(isolated_native_cache, native_cxx, kokkos_root):
    n = 16
    case, layout = make_case(n, dt=1.0, nonlinear=True, invalid=True)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.zeros((1, n, n))
    initial[0, 0, 0] = 2.0
    runtime = pops.bind(artifact, initial_state={"material": initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    native = runtime._executor

    def envelope():
        return (runtime.time(), runtime.macro_step(), native._program_exchange_records(), tuple(
            (name, native.history_initialized(name), native.history_fill_count(name), tuple(
                np.asarray(native.history_global(name, slot)).tobytes()
                for slot in range(native.history_depth(name))))
            for name in native.history_names()))

    assert tuple(native.history_names()) == ("prior_energy",)
    before = envelope()
    with pytest.raises(RuntimeError, match="(?i)(invalid|domain|transform|spatial)") as failed:
        pops.run(runtime, t_end=1.0, max_steps=1)
    assert "coordinate_to_energy" in str(failed.value)
    np.testing.assert_array_equal(np.asarray(runtime.state_global("material")).reshape(initial.shape), initial)
    assert envelope() == before
    assert native.history_fill_count("prior_energy") == 0
