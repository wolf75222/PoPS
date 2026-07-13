"""Generic pre-implemented IMEX Programs built from exact operator handles."""
from __future__ import annotations

from typing import Any

from pops.time.method_tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau
from pops.time import FailRun, LocalLinear, SolveAction
from pops.solvers import DenseLU

from ._factory import call_at, instance_state, operator_handle, program_factory
from ._helpers import _op_space_arity, _stage_point


IMEX_EULER_TABLEAU = AdditiveRungeKuttaTableau(
    RungeKuttaTableau(A=[[]], b=[1], c=[0], name="imex-euler-explicit"),
    implicit_A=[[1]], implicit_b=[1], implicit_c=[1], name="imex-euler",
)
_DEFAULT_SOLVE_ACTION = FailRun()


def _build_imex(
    program: Any,
    state: Any,
    explicit_operator: Any,
    implicit_operator: Any,
    fields_operator: Any,
    tableau: Any,
    solve_action: Any,
) -> None:
    if type(tableau) is not AdditiveRungeKuttaTableau:
        raise TypeError("IMEX tableau must be an exact AdditiveRungeKuttaTableau")
    explicit_operator = operator_handle(explicit_operator, "IMEX explicit_operator")
    implicit_operator = operator_handle(implicit_operator, "IMEX implicit_operator")
    if fields_operator is not None:
        fields_operator = operator_handle(fields_operator, "IMEX fields_operator")
    if not isinstance(solve_action, SolveAction):
        raise TypeError("IMEX solve_action must be FailRun(...) or RejectAttempt(...)")

    temporal = instance_state(program, state, "IMEX")
    explicit_arity = _op_space_arity(program, explicit_operator)
    implicit_arity = _op_space_arity(program, implicit_operator)
    if implicit_arity != 0:
        raise ValueError(
            "IMEX local-linear preset requires a field-independent implicit operator; "
            "field-coupled implicit systems must be authored as a residual Program")
    if fields_operator is None and explicit_arity != 1:
        raise ValueError(
            "IMEX explicit_operator requires fields but fields_operator was not provided")
    if fields_operator is not None and explicit_arity != 2:
        raise ValueError(
            "IMEX fields_operator is present but explicit_operator does not consume its FieldContext")

    u0 = temporal.n
    explicit_rates: list[Any] = []
    implicit_rates: list[Any] = []
    tag = (tableau.name + "_") if tableau.name else "imex_"

    for i in range(tableau.stages):
        point = _stage_point(program, "%sstage_%d" % (tag, i), partitions={
            "explicit": tableau.explicit.c[i],
            "implicit": tableau.implicit_c[i],
        })
        predictor = 1 * u0
        for j in range(i):
            explicit_weight = tableau.explicit.A[i][j]
            implicit_weight = tableau.implicit_A[i][j]
            if explicit_weight != 0:
                predictor = predictor + (program.dt * explicit_weight) * explicit_rates[j]
            if implicit_weight != 0:
                predictor = predictor + (program.dt * implicit_weight) * implicit_rates[j]
        predictor = program.value("%spredictor_%d" % (tag, i), predictor, at=point)
        linear = call_at(
            program, implicit_operator,
            name="%sL_%d" % (tag, i), point=point)
        diagonal = tableau.implicit_A[i][i]
        stage = predictor
        if diagonal != 0:
            stage = program.solve(
                LocalLinear(
                    operator=program.I - (program.dt * diagonal) * linear,
                    rhs=predictor),
                solver=DenseLU(), name="%sstage_solve_%d" % (tag, i),
            ).consume(action=FailRun())
            stage = program.value("%sstage_%d" % (tag, i), stage, at=point)
        fields = None
        if fields_operator is not None:
            # A split StagePoint may carry different explicit/implicit abscissae. The field is
            # solved from the actual implicit stage state, so give the solve one unambiguous
            # logical TimePoint before lifting its FieldContext back onto the joint stage.
            field_state = program.value(
                "%sfield_state_%d" % (tag, i), stage,
                at=point.time_for("implicit"),
            )
            outcome = fields_operator(field_state)
            fields = outcome.consume(action=solve_action)
            fields = program.value("%sfields_%d" % (tag, i), fields, at=point)
        explicit_rates.append(call_at(
            program, explicit_operator, stage, fields,
            name="%sk_exp_%d" % (tag, i), point=point))
        implicit_rates.append(program.value(
            "%sk_imp_%d" % (tag, i),
            program.apply(linear, stage),
            at=point,
        ))

    final = u0
    for weight, rate in zip(tableau.explicit.b, explicit_rates, strict=True):
        if weight != 0:
            final = final + (program.dt * weight) * rate
    for weight, rate in zip(tableau.implicit_b, implicit_rates, strict=True):
        if weight != 0:
            final = final + (program.dt * weight) * rate
    out = program.value("%sstep" % tag, final, at=temporal.next.point)
    program.commit(temporal.next, out)


def IMEX(
    state: Any,
    *,
    explicit_operator: Any,
    implicit_operator: Any,
    fields_operator: Any = None,
    tableau: Any = IMEX_EULER_TABLEAU,
    solve_action: Any = _DEFAULT_SOLVE_ACTION,
) -> Any:
    """Return a generic ordinary IMEX Program for one exact ``block[state]`` handle.

    The explicit rate, implicit local-linear map and optional field operator are exact qualified
    model handles. ``tableau`` is the complete additive RK configuration. Every field solve remains
    unreadable until ``solve_action`` explicitly consumes its outcome. No method name, hidden runtime
    route, fallback operator or preset-specific scheme object participates in lowering.
    """
    return program_factory(
        "IMEX",
        _build_imex,
        state,
        explicit_operator,
        implicit_operator,
        fields_operator,
        tableau,
        solve_action,
    )


__all__ = ["IMEX", "IMEX_EULER_TABLEAU"]
