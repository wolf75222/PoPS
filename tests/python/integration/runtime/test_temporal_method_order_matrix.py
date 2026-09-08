"""M6.3 complete declared temporal refinement matrix through the native lifecycle.

Two noncommuting shear generators avoid a commuting/scalar-only IMEX oracle.
The fixed spatially constant problem has no spatial truncation error. Every method
uses all four step counts; coefficients/expansion alone are not execution evidence.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pops
import pops.lib.time as lt
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.time import ExternalTimeGrid
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
CELLS = 16
STEP_COUNTS = (8, 16, 32, 64)
FINAL_TIME = 1.0
INITIAL = np.array((1.0, 0.3))
# name, expected order, rate evaluations and implicit solves per accepted step.
METHODS = (
    ("forward_euler", 1, 1, 0),
    ("ssprk2", 2, 2, 0),
    ("ssprk3", 3, 3, 0),
    ("rk4", 4, 4, 0),
    ("backward_euler", 1, 1, 1),
    ("implicit_midpoint", 2, 1, 1),
    ("imex_euler", 1, 2, 1),
    ("imex_ars222", 2, 6, 2),
)


def author_case(method):
    frame = Rectangle("temporal_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("noncommuting_temporal_system", frame=frame)
    state = model.state("U", components=("first", "second"))
    u, v = state
    upper_source = model.source("upper_source", on=state, value=(v, 0.0 * u))
    lower_source = model.source("lower_source", on=state, value=(0.0 * v, -2.0 * u))
    full = model.rate("full", equation=ddt(state) == upper_source + lower_source)
    upper = full.select(upper_source)
    lower_rate = full.select(lower_source)
    lower = model.operator("lower", returns=model.local_linear_operator(
        "lower", on=state, matrix=((0.0, 0.0), (-2.0, 0.0))))
    combined = model.operator("combined", returns=model.local_linear_operator(
        "combined", on=state, matrix=((0.0, 1.0), (-2.0, 0.0))))
    invalid_subflow = model.local_transform("invalid_subflow", (u, v), valid_if=u < 0)
    case = pops.Case("temporal_" + method)
    block = case.block("oscillator", model=model, states=(state,))
    numerics = DiscretizationPlan()
    if method.startswith("imex_"):
        numerics.rates.add(upper, StateStorage())
        numerics.rates.add(lower_rate, StateStorage())
    else:
        numerics.rates.add(full, StateStorage())
    case.numerics(numerics, block=block)
    explicit = {"forward_euler": lt.ForwardEuler, "ssprk2": lt.SSPRK2,
                "ssprk3": lt.SSPRK3, "rk4": lt.RK4}
    if method == "split_failure":
        def first(program, current, fraction, *, at):
            candidate = program.value("first_candidate", current + fraction * program.dt * full(current), at=at)
            program.store_history("provisional_split", candidate, depth=2)
            return candidate

        def second(program, current, fraction, *, at):
            transformed = program.transform(current, transform=invalid_subflow)
            return program.value("second_candidate", transformed, at=at)

        program = lt.Strang(block[state], first=first, second=second)
    elif method in explicit:
        program = explicit[method](block[state], rate=full)
    elif method in ("backward_euler", "implicit_midpoint"):
        tableau = (lt.BACKWARD_EULER_TABLEAU if method == "backward_euler"
                   else lt.IMPLICIT_MIDPOINT_TABLEAU)
        program = lt.DIRK(block[state], implicit_operator=combined, tableau=tableau)
    else:
        assert method in ("imex_euler", "imex_ars222")
        tableau = lt.IMEX_EULER_TABLEAU if method == "imex_euler" else lt.IMEX_ARS222_TABLEAU
        program = lt.IMEX(block[state], explicit_operator=upper, implicit_operator=lower,
                          tableau=tableau)
    program.step_strategy(ExternalTimeGrid("time_grid"))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program


def exact_solution():
    omega = math.sqrt(2.0)
    return math.cos(omega * FINAL_TIME) * INITIAL + math.sin(omega * FINAL_TIME) / omega * np.array(
        (INITIAL[1], -2.0 * INITIAL[0]))


def run_artifact(artifact, steps):
    initial = np.broadcast_to(INITIAL[:, None, None], (2, CELLS, CELLS)).copy()
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=FINAL_TIME, max_steps=steps,
                      time_grid=tuple(FINAL_TIME * i / steps for i in range(steps + 1)))
    assert report.accepted_steps == steps
    assert report.rejected_steps == 0
    assert runtime.time() == FINAL_TIME
    values = np.asarray(runtime.state_global("oscillator")).reshape(2, CELLS, CELLS)
    np.testing.assert_allclose(values, np.broadcast_to(values[:, :1, :1], values.shape),
                               rtol=0, atol=2e-14)
    return float(np.linalg.norm(values[:, 0, 0] - exact_solution()))


@pytest.mark.parametrize("method,order,evaluations,solves", METHODS, ids=[row[0] for row in METHODS])
def test_native_temporal_order_matrix(method, order, evaluations, solves,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    case, layout, program = author_case(method)
    assert program.validate()
    operations = [value.op for value in program._values]
    assert operations.count("solve_local_linear") == solves
    assert operations.count("solve_outcome") == solves
    assert len(program.commits()) == 1
    resolved = pops.resolve(pops.validate(case), layout=layout)
    artifact = pops.compile(resolved)
    errors = [run_artifact(artifact, count) for count in STEP_COUNTS]
    orders = [math.log2(a / b) for a, b in zip(errors[:-1], errors[1:], strict=True)]
    record_property("temporal_matrix", json.dumps({"method": method, "cells": [CELLS, CELLS],
        "step_counts": STEP_COUNTS, "final_time": FINAL_TIME, "errors_l2": errors,
        "observed_orders": orders, "declared_order": order,
        "rate_evaluations_per_step": evaluations, "implicit_solves_per_step": solves,
        "cost_status": "operation_counts_only_no_performance_claim"}))
    assert all(a > b > 0 for a, b in zip(errors[:-1], errors[1:], strict=True)), errors
    assert all(order - 0.25 < actual < order + 0.25 for actual in orders), (errors, orders)


def test_split_native_failure_restores_complete_accepted_envelope(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    case, layout, _ = author_case("split_failure")
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.broadcast_to(INITIAL[:, None, None], (2, CELLS, CELLS)).copy()
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    def accepted():
        native = runtime._executor
        return (runtime.time(), runtime.macro_step(),
                np.asarray(runtime.state_global("oscillator")).tobytes(),
                native._program_exchange_records(), tuple(
                    (name, native.history_initialized(name), native.history_fill_count(name), tuple(
                        np.asarray(native.history_global(name, slot)).tobytes()
                        for slot in range(native.history_depth(name)))) for name in native.history_names()))
    before = accepted()
    with pytest.raises(RuntimeError, match="local_transform|invalid_subflow") as failed:
        pops.run(runtime, t_end=0.125, max_steps=1, time_grid=(0.0, 0.125))
    assert accepted() == before
    assert runtime._executor.history_fill_count("provisional_split") == 0
    record_property("failure", str(failed.value))


def test_spatial_backward_euler_temporal_order_at_fixed_mesh(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    from tests.python.integration.runtime.test_implicit_diffusion_lifecycle import mode, run_case
    n, final_time, counts = 16, 0.04, (4, 8, 16, 32)
    wave = mode(n)
    eigenvalue = 8 * 0.1 * math.sin(math.pi / n) ** 2 * n ** 2
    exact = 1 + math.exp(-eigenvalue * final_time) * wave
    errors = []
    for count in counts:
        actual, _ = run_case(n, 1 + wave, method="backward_euler", dt=final_time / count, steps=count)
        errors.append(float(np.sqrt(np.mean((actual - exact) ** 2))))
        assert abs(float(np.mean(actual)) - 1.0) < 1e-10
    orders = [math.log2(a / b) for a, b in zip(errors[:-1], errors[1:], strict=True)]
    record_property("spatial_implicit_temporal_matrix", json.dumps({"cells": [n, n],
        "step_counts": counts, "final_time": final_time, "errors_l2": errors,
        "orders": orders, "reference": "exact_semidiscrete_fourier_mode_with_nonzero_mean"}))
    assert all(0.85 < order < 1.15 for order in orders), (errors, orders)
