"""Public Program proof for nonlinear IMEX ARS(2,2,2).

This test owns no stepper and calls no compatibility runtime.  It proves that the public
``pops.lib.time.IMEX`` factory builds the two diagonal nonlinear stages through typed
``LocalResidual``/``LocalNewton`` solves, that every result is consumed fail-closed, and that the
exported binary64 ARS tableau retains second-order behavior on a split scalar problem.
"""

from __future__ import annotations

from math import exp

from pops.codegen.program_codegen import emit_cpp_program
from pops.lib.time import IMEX, IMEX_ARS222_TABLEAU
from pops.physics._facade import Model
from pops.solvers.nonlinear import LocalNewton
from pops.time import Program
from typed_program_support import state_refs


def _authoring():
    model = Model("imex_ars222_model")
    (u,) = model.conservative_vars("u")
    model.source_term("slow", [-0.2 * u])
    implicit = model.source_term("stiff", [-u * u])
    explicit = model.rate("explicit", flux=False, sources=("slow",))
    block, state = state_refs(Program("imex_ars222_refs"), "scalar", model=model)
    program = IMEX(
        block[state],
        explicit_operator=explicit,
        implicit_operator=implicit,
        tableau=IMEX_ARS222_TABLEAU,
        implicit_solver=LocalNewton(
            tolerance=1.0e-12,
            max_iterations=25,
            finite_difference_step=1.0e-7,
        ),
    )
    return model, program


def test_public_ars222_uses_two_consumed_local_newton_stages():
    model, program = _authoring()

    assert program.validate() is True
    nodes = program._serialize()["nodes"]
    operations = [node["op"] for node in nodes]
    assert operations.count("solve_local_nonlinear") == 2
    assert operations.count("solve_outcome") == 2
    assert operations.count("source") == 3

    solves = [node for node in nodes if node["op"] == "solve_local_nonlinear"]
    assert all(node["attrs"]["problem_kind"] == "local_residual" for node in solves)
    assert all(node["attrs"]["max_iter"] == 25 for node in solves)
    assert all(
        any(
            residual["op"] == "source" and residual["attrs"]["source"] == "stiff"
            for residual in node["attrs"]["residual_block"]
        )
        for node in solves
    )
    consumes = [node for node in nodes if node["op"] == "solve_outcome"]
    assert all(node["attrs"]["action"]["kind"] == "fail_run" for node in consumes)

    generated = emit_cpp_program(program, model=model)
    assert generated.count("for (int it_ = 0;") == 2
    assert "pops::detail::mat_inverse<1>(" in generated


def _ars222_step(value: float, dt: float) -> float:
    tableau = IMEX_ARS222_TABLEAU
    explicit_rates: list[float] = []
    implicit_rates: list[float] = []

    for stage in range(tableau.stages):
        predictor = value
        for previous in range(stage):
            predictor += dt * tableau.explicit.A[stage][previous] * explicit_rates[previous]
            predictor += dt * tableau.implicit_A[stage][previous] * implicit_rates[previous]

        diagonal = tableau.implicit_A[stage][stage]
        candidate = predictor
        for _ in range(20):
            residual = candidate - predictor + dt * diagonal * candidate * candidate
            candidate -= residual / (1.0 + 2.0 * dt * diagonal * candidate)

        explicit_rates.append(-0.2 * candidate)
        implicit_rates.append(-candidate * candidate)

    result = value
    for weight, rate in zip(tableau.explicit.b, explicit_rates, strict=True):
        result += dt * weight * rate
    for weight, rate in zip(tableau.implicit_b, implicit_rates, strict=True):
        result += dt * weight * rate
    return result


def _integrate(steps: int) -> float:
    value = 1.0
    dt = 1.0 / steps
    for _ in range(steps):
        value = _ars222_step(value, dt)
    return value


def test_ars222_tableau_has_observed_second_order_on_a_split_nonlinear_problem():
    # u' = -0.2 u - u^2, u(0)=1:
    # u(t) = a*u0*exp(-a*t) / (a + u0*(1-exp(-a*t))).
    decay = 0.2
    exact = decay * exp(-decay) / (decay + 1.0 - exp(-decay))
    errors = [abs(_integrate(steps) - exact) for steps in (20, 40, 80)]

    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > 3.7
    assert errors[1] / errors[2] > 3.7
