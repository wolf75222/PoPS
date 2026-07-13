"""Canonical local-linear predictor--corrector Program factory."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.solvers import DenseLU
from pops.time import FailRun, LocalLinear

from ._factory import call_at, instance_state, operator_handle, program_factory
from ._helpers import _stage_point


def _build_predictor_corrector(
    program: Any,
    state: Any,
    fields: Any,
    explicit: Any,
    implicit: Any,
) -> None:
    fields = operator_handle(fields, "PredictorCorrector fields")
    explicit = operator_handle(explicit, "PredictorCorrector explicit")
    implicit = operator_handle(implicit, "PredictorCorrector implicit")
    temporal = instance_state(program, state, "PredictorCorrector")
    initial = temporal.n
    predictor = _stage_point(
        program, "predictor", partitions={"explicit": 0, "implicit": 1})
    fields_initial = call_at(
        program, fields, initial, name="fields_n", point=predictor)
    rate_initial = call_at(
        program, explicit, initial, fields_initial, name="rate_n", point=predictor)
    linear_initial = call_at(
        program, implicit, fields_initial, name="linear_n", point=predictor)
    predictor_rhs = program.value(
        "predictor_rhs", initial + program.dt * rate_initial, at=predictor)
    predicted = program.solve(
        LocalLinear(
            operator=program.I - program.dt * linear_initial,
            rhs=predictor_rhs, fields=fields_initial),
        solver=DenseLU(), name="predictor_solve",
    ).consume(action=FailRun())
    predicted = program.value("predicted_state", predicted, at=predictor)

    corrector = _stage_point(
        program, "corrector", partitions={"explicit": 1, "implicit": 1})
    fields_predicted = call_at(
        program, fields, predicted, name="fields_predicted", point=corrector)
    rate_predicted = call_at(
        program, explicit, predicted, fields_predicted,
        name="rate_predicted", point=corrector,
    )
    linear_predicted = call_at(
        program, implicit, fields_predicted,
        name="linear_predicted", point=corrector,
    )
    applied = program.apply(
        linear_predicted, predicted, fields=fields_predicted, name="implicit_predicted")
    applied = program.value("implicit_predicted", applied, at=corrector)
    half = Fraction(1, 2)
    corrector_rhs = program.value(
        "corrector_rhs",
        initial
        + half * program.dt * rate_initial
        + half * program.dt * rate_predicted
        + half * program.dt * applied,
        at=temporal.next.point,
    )
    corrected = program.solve(
        LocalLinear(
            operator=program.I - half * program.dt * linear_predicted,
            rhs=corrector_rhs, fields=fields_predicted),
        solver=DenseLU(), name="corrector_solve",
    ).consume(action=FailRun())
    endpoint = program.value(
        "predictor_corrector_step", corrected, at=temporal.next.point)
    program.commit(temporal.next, endpoint)


def PredictorCorrector(
    state: Any,
    *,
    fields: Any,
    explicit: Any,
    implicit: Any,
) -> Any:
    """Return a trapezoidal predictor--corrector Program from three typed operators."""
    return program_factory(
        "PredictorCorrector",
        _build_predictor_corrector,
        state,
        fields,
        explicit,
        implicit,
    )


__all__ = ["PredictorCorrector"]
