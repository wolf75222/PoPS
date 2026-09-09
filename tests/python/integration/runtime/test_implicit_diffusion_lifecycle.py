"""Declared M5.4 Uniform scalar matrix through the complete public lifecycle."""
from __future__ import annotations

import math
import json

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
              tau_scale=1, derivative_route="finite_difference", commit_mode="solved",
              failure_action=None, solver=None):
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
        solved = program.solve(request, solver=solver if solver is not None else Newton(
            tolerance=1e-12, max_iterations=20, linear_tolerance=1e-8,
            linear_max_iterations=100, restart=30)).consume(
                action=FailRun() if failure_action is None else failure_action)[0]
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
    elif commit_mode == "mutated_alias":
        program.project(candidate)
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
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: np.asarray(initial).reshape(1, n, n).copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    dt = kwargs.get("dt", 0.0001)
    report = pops.run(runtime, t_end=steps * dt, max_steps=steps)
    assert report.accepted_steps == steps
    return np.asarray(runtime.state_global("material")).reshape(n, n), runtime


def mode(n):
    x = (np.arange(n) + 0.5) / n
    return np.sin(2 * np.pi * x)[:, None] * np.sin(2 * np.pi * x)[None, :]


def test_same_physical_flux_explicit_implicit_fourier_parity(isolated_native_cache, native_cxx, kokkos_root,
                                                          record_property):
    n, dt = 32, 0.0001
    wave = mode(n)
    eigenvalue = 8 * 0.1 * math.sin(math.pi / n) ** 2 * n ** 2
    for method, amplitude in (("forward_euler", 1 - dt * eigenvalue),
                              ("backward_euler", 1 / (1 + dt * eigenvalue))):
        actual, _ = run_case(n, 1 + wave, method=method, dt=dt)
        np.testing.assert_allclose(actual, 1 + amplitude * wave, rtol=0, atol=1e-9)
        record_property(method + "_max_error", float(np.max(np.abs(actual - (1 + amplitude * wave)))))


@pytest.mark.parametrize("method", ["forward_euler", "backward_euler"])
def test_scalar_diffusion_manufactured_convergence(method, isolated_native_cache, native_cxx, kokkos_root,
                                                 record_property):
    errors = []
    for n, steps in ((16, 4), (32, 16), (64, 64)):
        wave = mode(n)
        actual, _ = run_case(n, 1 + wave, method=method, dt=0.005 * (16 / n) ** 2, steps=steps)
        exact = 1 + math.exp(-8 * math.pi ** 2 * 0.1 * 0.02) * wave
        errors.append(float(np.sqrt(np.mean((actual - exact) ** 2))))
    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors[:-1], errors[1:], strict=True)]
    record_property("errors_N16_N32_N64", json.dumps(errors))
    record_property("orders", json.dumps(orders))
    assert all(order >= 1.7 for order in orders), errors


def test_shifted_operator_retains_nonzero_constant(isolated_native_cache, native_cxx, kokkos_root,
                                                 record_property):
    actual, _ = run_case(16, np.full((16, 16), 2.5), dt=0.125)
    np.testing.assert_allclose(actual, 2.5, rtol=0, atol=1e-11)
    record_property("constant_max_error", float(np.max(np.abs(actual - 2.5))))


def test_nonlinear_accumulation_preserves_energy(isolated_native_cache, native_cxx, kokkos_root,
                                               record_property):
    energy, runtime = run_case(16, np.zeros((16, 16)), dt=1.0, nonlinear=True,
                               solver=m5_qualification_solver())
    temperature = (np.sqrt(1 + 4 * energy) - 1) / 2
    np.testing.assert_allclose(energy, 1.0, rtol=0, atol=2e-11)
    np.testing.assert_allclose(temperature, (-1 + math.sqrt(5)) / 2, rtol=0, atol=2e-11)
    assert float(np.min(temperature)) > 0.6
    diagnostics = runtime._executor.program_diagnostics()
    evaluations = [value for key, value in diagnostics.items() if key.endswith(".full_residual_evaluations")]
    derivatives = [value for key, value in diagnostics.items() if key.endswith(".finite_difference_jvps")]
    assert len(evaluations) == len(derivatives) == 1
    assert evaluations[0] > 2 * derivatives[0] > 0
    record_property("temperature_min_max", json.dumps([float(np.min(temperature)), float(np.max(temperature))]))
    record_property("energy_max_error", float(np.max(np.abs(energy - 1))))
    accumulation_residual = float(np.max(np.abs(temperature + temperature**2 - 1)))
    assert accumulation_residual <= 2e-11
    record_property("accumulation_residual_linf", accumulation_residual)
    record_property("solver", json.dumps(M5_SOLVER_CONTROLS))
    record_property("full_residual_evaluations", evaluations[0])
    record_property("finite_difference_jvps", derivatives[0])


def test_invalid_nonlinear_domain_leaves_accepted_state_unchanged(isolated_native_cache, native_cxx, kokkos_root,
                                                                record_property):
    n = 16
    case, layout = make_case(n, dt=1.0, nonlinear=True, invalid=True)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.zeros((1, n, n))
    initial[0, 0, 0] = 2.0
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
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
    assert "action=fail_run" in str(failed.value)
    record_property("failure_reason", str(failed.value))
    np.testing.assert_array_equal(np.asarray(runtime.state_global("material")).reshape(initial.shape), initial)
    assert envelope() == before
    assert native.history_fill_count("prior_energy") == 0


# Newton exposes a hybrid outer tolerance and a relative inner tolerance, not
# separate absolute/relative pairs. These original M5 profiles use a stricter
# outer relative target than the plan; existing M6 helper defaults stay intact.
M5_SOLVER_CONTROLS = {
    "outer_stop": "1e-13 * max(1, initial_residual_euclidean_norm)",
    "inner_stop": "1e-12 * current_residual_euclidean_norm",
    "outer_max_iterations": 20, "inner_max_iterations": 100, "restart": 30,
    "derivative": "finite_difference",
    "native_final_norm": "not exposed by the public spatial solve diagnostics",
}


def m5_qualification_solver(*, max_iterations=20):
    return Newton(tolerance=1e-13, max_iterations=max_iterations,
                  linear_tolerance=1e-12, linear_max_iterations=100, restart=30)


def cell_average_mode(n):
    """Exact unit-cell averages of sin(2*pi*x)*sin(2*pi*y)."""
    return np.sinc(1 / n)**2 * mode(n)


def test_original_m5_backward_euler_spatial_matrix(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    final_time, nu = 0.05, 0.1
    rows = []
    for n in (16, 32, 64):
        steps = 13 * (n // 16)**2
        dt = final_time / steps
        assert dt <= 0.1 / (nu * n*n)
        wave = cell_average_mode(n)
        initial = 2 + wave
        actual, runtime = run_case(n, initial, dt=dt, steps=steps,
                                   solver=m5_qualification_solver())
        exact = 2 + math.exp(-8 * math.pi**2 * nu * final_time) * wave
        error = np.abs(actual - exact)
        assert np.isfinite(actual).all()
        assert abs(runtime.time() - final_time) <= 2e-14
        mass_error = abs(float(np.mean(actual) - np.mean(initial))) / abs(float(np.mean(initial)))
        assert mass_error <= 2e-11
        row = {"n": n, "steps": steps, "dt": dt, "final_time": final_time,
               "l1": float(np.mean(error)), "l2": float(np.sqrt(np.mean(error**2))),
               "linf": float(np.max(error)), "relative_mass_error": mass_error,
               "solver": M5_SOLVER_CONTROLS}
        rows.append(row)
        record_property(f"original_m5_be_spatial_N{n}", json.dumps(row))
    orders = {}
    for norm in ("l1", "l2", "linf"):
        errors = [row[norm] for row in rows]
        assert all(a > b > 0 for a, b in zip(errors[:-1], errors[1:], strict=True)), rows
        orders[norm] = [math.log2(a / b) for a, b in zip(errors[:-1], errors[1:], strict=True)]
        assert orders[norm][-1] >= 1.75, (rows, orders)
    record_property("original_m5_be_spatial_orders", json.dumps(orders))


def test_uniform_implicit_iteration_budget_rolls_back_state_clock_and_exchanges(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    n = 16
    case, layout = make_case(n, dt=1.0, nonlinear=True,
                             solver=m5_qualification_solver(max_iterations=1))
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    initial = np.zeros((1, n, n))
    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})

    def accepted_envelope():
        return (runtime.time(), runtime.macro_step(),
                runtime._executor._program_exchange_records(),
                np.asarray(runtime.state_global("material")).tobytes())

    before = accepted_envelope()
    # The valid seed T=.2 has Q(T)=.24. One Newton correction cannot solve
    # Q(T)=1 to tolerance; this exercises numerical exhaustion, not invalid input.
    with pytest.raises(RuntimeError, match="field_newton_iteration_limit") as failed:
        pops.run(runtime, t_end=1.0, max_steps=1)
    assert "action=fail_run" in str(failed.value)
    assert accepted_envelope() == before
    np.testing.assert_array_equal(np.asarray(runtime.state_global("material")).reshape(initial.shape), initial)
    record_property("failure_reason", str(failed.value))
    record_property("outer_iteration_budget", 1)
    record_property("accepted_exchange_count_after_failure", len(before[2]))


def test_uniform_backward_euler_accepted_face_quadrature(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    nu = 0.1
    for n in (16, 32, 64):
        initial = 2 + 0.4 * cell_average_mode(n)
        dt = 0.05 / (nu * n*n)
        actual, runtime = run_case(n, initial, dt=dt, solver=m5_qualification_solver())
        records = runtime._executor._program_exchange_records()
        assert len(records) == 4*n*n
        assert len({row["evaluation_context"] for row in records}) == 1
        assert len({row["occurrence_identity"] for row in records}) == 1
        incidences = set()
        delta = np.zeros_like(initial)
        flux_defect = 0.0
        for row in records:
            cell_token, axis_token, side_token = row["quadrature_identity"].split("/")
            i, j = map(int, cell_token.split(":")[1:])
            axis, side = int(axis_token.split(":")[1]), int(side_token.split(":")[1])
            assert 0 <= i < n and 0 <= j < n and axis in (0, 1) and side in (0, 1)
            incidence = (i, j, axis, side)
            assert incidence not in incidences
            incidences.add(incidence)
            assert row["face_measure"] == 1/n and row["multiplicity"] == 1
            assert row["temporal_weight"] == dt
            assert row["orientation"] == (1 if side else -1)
            other = [j, i]
            other[1-axis] = (other[1-axis] + (1 if side else -1)) % n
            neighbor = actual[tuple(other)]
            left, right = (actual[j, i], neighbor) if side else (neighbor, actual[j, i])
            flux_defect = max(flux_defect, abs(row["numerical_flux"] - nu*n*(right-left)))
            expected_amount = row["orientation"] * row["numerical_flux"] * dt/n
            assert abs(row["integrated_amount"] - expected_amount) <= 2e-15
            delta[j, i] += row["integrated_amount"]
        assert flux_defect <= 2e-11
        state_delta = (actual - initial) / n**2
        np.testing.assert_allclose(delta, state_delta, rtol=0, atol=2e-11)
        laplacian = nu*n*n * sum(np.roll(actual, 1, axis) + np.roll(actual, -1, axis)
                                - 2*actual for axis in (0, 1))
        residual = actual - initial - dt*laplacian
        assert float(np.max(np.abs(residual))) <= 2e-11
        diagnostics = runtime._executor.program_diagnostics()
        evaluations = [v for k, v in diagnostics.items() if k.endswith(".full_residual_evaluations")]
        derivatives = [v for k, v in diagnostics.items() if k.endswith(".finite_difference_jvps")]
        assert len(evaluations) == len(derivatives) == 1
        assert evaluations[0] > derivatives[0] > 0
        record_property(f"original_m5_implicit_exchanges_N{n}", json.dumps({
            "n": n, "dt": dt, "records": len(records), "accepted_contexts": 1,
            "face_flux_linf_defect": flux_defect,
            "cell_exchange_linf_defect": float(np.max(np.abs(delta-state_delta))),
            "independent_final_residual_l2": float(np.sqrt(np.mean(residual**2))),
            "independent_final_residual_linf": float(np.max(np.abs(residual))),
            "full_residual_evaluations": evaluations[0], "finite_difference_jvps": derivatives[0],
            "solver": M5_SOLVER_CONTROLS}))
