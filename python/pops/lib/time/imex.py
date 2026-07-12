"""pops.lib.time.imex -- IMEX (implicit-explicit) time-stepping schemes.

Exports: imex_local, imex_local_linear.
"""
from __future__ import annotations

from typing import Any

from pops.time.method_tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau

from ._helpers import (
    _DEFAULT_SOURCES, _at_point, _block_label, _commit, _exact_coefficient, _opcall,
    _operator_handle, _stage_point, _time_state, _typed_rhs, program_macro,
)


IMEX_EULER_TABLEAU = AdditiveRungeKuttaTableau(
    RungeKuttaTableau(A=[[]], b=[1], c=[0], name="imex-euler-explicit"),
    implicit_A=[[1]], implicit_b=[1], implicit_c=[1], name="imex-euler",
)


@program_macro
def imex_local(P: Any, block: Any, state: Any = None, *,
               linear_source: Any, sources: Any = _DEFAULT_SOURCES,
               flux: Any = True, theta: Any = 1) -> Any:
    """IMEX with an EXPLICIT flux/source and an IMPLICIT cell-local linear source (ADC-423).

    One step of a theta-implicit splitting of ``dU/dt = R_explicit(U) + L U`` where ``L`` is a named
    model ``m.linear_source`` (e.g. a Lorentz operator) solved cell by cell:

        R   = R_explicit(U)                                     (P.rhs: -div F + the named sources)
        U^{n+1} = (I - theta*dt*L)^{-1} (U + dt*R)              (P.solve_local_linear)

    The explicit part is assembled with `P.rhs` (flux + the requested named @p sources, on the fields
    solved from U); the implicit part is the local solve of ``(I - theta*dt*L) U^{n+1} = U + dt*R``
    via `P.solve_local_linear`, exactly the predictor half of the codebase's predictor-corrector
    pattern (``test_time_local_solve``). At ``theta == 1`` this is backward Euler on the L term and
    forward Euler on R; ``theta == 0`` would drop the implicit solve (use `forward_euler` instead) and
    is rejected. @p linear_source is the typed handle of the model ``m.linear_source`` /
    ``m.local_linear_map``; @p theta the implicitness of the L term (0 < theta <= 1)."""
    linear_source = _operator_handle(linear_source, "linear_source")
    theta = _exact_coefficient(theta, "imex_local: theta")
    if not (0 < theta <= 1):
        raise ValueError(
            "imex_local: theta must be in (0, 1] (got %r); theta == 0 is fully explicit -- use "
            "forward_euler instead" % (theta,))
    temporal = _time_state(P, block, state)
    label = _block_label(temporal)
    U = temporal.n
    point = _stage_point(
        P, label + "_imex", partitions={"explicit": 0, "implicit": theta})
    fields = _at_point(P, P.solve_fields(U), point) if flux else None
    R = _at_point(P, _typed_rhs(P, U, fields=fields, sources=sources, flux=flux), point)
    rhs = P.linear_combine(
        label + "_imex_rhs", U + P.dt * R, at=temporal.next.point)
    linear = _at_point(P, P.linear_source(linear_source), point)
    operator = P.I - (theta * P.dt) * linear
    out = P.solve_local_linear(
        name=label + "_imex_step", operator=operator, rhs=rhs, fields=fields)
    _commit(P, temporal, out)
    return out


@program_macro
def imex_local_linear(P: Any, block: Any, state: Any = None, *,
                      explicit_operator: Any, implicit_operator: Any,
                      fields_operator: Any = None, theta: Any = 1,
                      ) -> Any:
    """Generic IMEX with an explicit rate and an implicit local linear operator (Spec 2).

    One theta-implicit step of ``dU/dt = R(U[, fields]) + L([fields]) U``::

        U^{n+1} = (I - theta dt L)^{-1} (U^n + dt R)

    composing the typed ``explicit_operator`` and ``implicit_operator`` handles (and an optional
    ``fields_operator`` handle). Each is a :class:`pops.model.OperatorHandle` from an ``m.*``
    declarer, not a name string. Requires ``P.bind_operators(module)``.
    """
    theta = _exact_coefficient(theta, "imex_local_linear: theta")
    if not (0 < theta <= 1):
        raise ValueError("imex_local_linear: theta must be in (0, 1]")
    explicit_operator = _operator_handle(explicit_operator, "explicit_operator")
    implicit_operator = _operator_handle(implicit_operator, "implicit_operator")
    if fields_operator is not None:
        fields_operator = _operator_handle(fields_operator, "fields_operator")
    temporal = _time_state(P, block, state)
    u = temporal.n
    point = _stage_point(
        P, "imex", partitions={"explicit": 0, "implicit": theta})
    fields = (
        _opcall(P, fields_operator, u, value_name="fields", point=point)
        if fields_operator else None
    )
    r = _opcall(P, explicit_operator, u, fields, value_name="R", point=point)
    lin = _opcall(P, implicit_operator, fields, value_name="L", point=point)
    q = P.linear_combine(
        "imex_rhs", u + P.dt * r, at=temporal.next.point)
    u1 = P.solve_local_linear(
        "imex_step", operator=P.I - theta * P.dt * lin, rhs=q, fields=fields)
    _commit(P, temporal, u1)
    return u1


@program_macro
def ark_local_linear(P: Any, block: Any, state: Any = None, *,
                     explicit_operator: Any, implicit_operator: Any,
                     tableau: AdditiveRungeKuttaTableau,
                     fields_operator: Any = None) -> Any:
    """Lower an exact diagonally-implicit ARK tableau to ordinary Program operations.

    The implicit operator denotes a local linear map.  Every stage records both partition
    abscissae in one :class:`StagePoint`; no ARK-specific runtime stepper or hidden schedule exists.
    """
    if type(tableau) is not AdditiveRungeKuttaTableau:
        raise TypeError("ark_local_linear: tableau must be an AdditiveRungeKuttaTableau")
    explicit_operator = _operator_handle(explicit_operator, "explicit_operator")
    implicit_operator = _operator_handle(implicit_operator, "implicit_operator")
    if fields_operator is not None:
        fields_operator = _operator_handle(fields_operator, "fields_operator")
    temporal = _time_state(P, block, state)
    u0 = temporal.n
    explicit_rates: list[Any] = []
    implicit_rates: list[Any] = []
    tag = (tableau.name + "_") if tableau.name else "ark_"
    for i in range(tableau.stages):
        point = _stage_point(P, "%sstage_%d" % (tag, i), partitions={
            "explicit": tableau.explicit.c[i], "implicit": tableau.implicit_c[i]})
        predictor = u0
        for j in range(i):
            ae = tableau.explicit.A[i][j]
            ai = tableau.implicit_A[i][j]
            if ae != 0:
                predictor = predictor + (P.dt * ae) * explicit_rates[j]
            if ai != 0:
                predictor = predictor + (P.dt * ai) * implicit_rates[j]
        predictor = P.linear_combine("%spredictor_%d" % (tag, i), predictor, at=point)
        fields = (_opcall(P, fields_operator, predictor,
                          value_name="%sfields_%d" % (tag, i), point=point)
                  if fields_operator else None)
        linear = _opcall(P, implicit_operator, fields,
                         value_name="%sL_%d" % (tag, i), point=point)
        diagonal = tableau.implicit_A[i][i]
        stage = predictor
        if diagonal != 0:
            stage = P.solve_local_linear(
                "%sstage_solve_%d" % (tag, i),
                operator=P.I - (P.dt * diagonal) * linear,
                rhs=predictor, fields=fields)
            stage = _at_point(P, stage, point)
        explicit_rates.append(_opcall(
            P, explicit_operator, stage, fields,
            value_name="%sk_exp_%d" % (tag, i), point=point))
        implicit_rates.append(_at_point(P, P.apply(
            linear, stage, fields=fields, name="%sk_imp_%d" % (tag, i)), point))
    final = u0
    for weight, rate in zip(tableau.explicit.b, explicit_rates, strict=True):
        if weight != 0:
            final = final + (P.dt * weight) * rate
    for weight, rate in zip(tableau.implicit_b, implicit_rates, strict=True):
        if weight != 0:
            final = final + (P.dt * weight) * rate
    out = P.linear_combine("%sstep" % tag, final, at=temporal.next.point)
    _commit(P, temporal, out)
    return out


__all__ = [
    "IMEX_EULER_TABLEAU", "ark_local_linear", "imex_local", "imex_local_linear",
]
