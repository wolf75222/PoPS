"""pops.lib.time.rk -- Classic explicit Runge-Kutta schemes (RK4, generic rk) and Butcher tableaux.

Exports: rk4, rk, explicit_rk, ButcherTableau, RK4_TABLEAU, SSPRK2_TABLEAU.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.time.method_tableau import RungeKuttaTableau

from ._helpers import (
    _DEFAULT_SOURCES, _commit, _opcall, _operator_handle, _stage_point, _stage_rhs, _time_state,
    program_macro,
)


ButcherTableau = RungeKuttaTableau


# RK4 (classic): the same tableau the rk4 macro hard-codes, written data-driven.
RK4_TABLEAU = ButcherTableau(
    A=[[],
       [Fraction(1, 2)],
       [0, Fraction(1, 2)],
       [0, 0, 1]],
    b=[Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6)],
    c=[0, Fraction(1, 2), Fraction(1, 2), 1],
    name="rk4")

# SSPRK2 (Heun) in NON-Shu-Osher Butcher form: k1 at U, k2 at U+dt*k1, U^{n+1}=U+dt(1/2 k1+1/2 k2).
SSPRK2_TABLEAU = ButcherTableau(
    A=[[],
       [1]],
    b=[Fraction(1, 2), Fraction(1, 2)],
    c=[0, 1],
    name="ssprk2")


def _rk_from_tableau(P: Any, block: Any, state: Any, tableau: RungeKuttaTableau,
                     sources: Any, flux: Any) -> None:
    tag = (tableau.name + "_") if tableau.name else "rk_"
    temporal = _time_state(P, block, state)
    U0 = temporal.n
    ks: list[Any] = []
    for i in range(tableau.stages):
        point = _stage_point(P, "%sstage_%d" % (tag, i), tableau.c[i])
        Ui = U0
        if i:
            expr = U0
            for j, aij in enumerate(tableau.A[i]):
                if aij != 0:
                    expr = expr + (P.dt * aij) * ks[j]
            Ui = P.linear_combine("%sU%d" % (tag, i), expr, at=point)
        ks.append(_stage_rhs(
            P, Ui, sources, flux, name="%sstage_%d" % (tag, i), offset=tableau.c[i]))
    final = U0
    for bi, ki in zip(tableau.b, ks, strict=True):
        if bi != 0:
            final = final + (P.dt * bi) * ki
    _commit(P, temporal, P.linear_combine("%sstep" % tag, final, at=temporal.next.point))


@program_macro
def rk4(P: Any, block: Any, state: Any = None, *,
        sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """Classic fourth-order RK, lowered solely through :data:`RK4_TABLEAU`."""
    return _rk_from_tableau(P, block, state, RK4_TABLEAU, sources, flux)


@program_macro
def rk(P: Any, block: Any, state: Any = None, tableau: Any = None, *,
       sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """Generic explicit Runge-Kutta from a Butcher @p tableau (ADC-423), lowered to the SAME stage chain
    the hard-coded `rk4` macro emits -- ``solve_fields`` + ``rhs`` + ``linear_combine``, no RK class:

        k_i      = R( U + dt * sum_{j<i} A[i][j] * k_j )       (the i-th stage RHS)
        U^{n+1}  = U + dt * sum_i b[i] * k_i

    @p tableau is a `ButcherTableau` (or a raw ``(A, b, c)`` triple); ``A`` must be strictly
    lower-triangular (explicit). ``RK4_TABLEAU`` and ``SSPRK2_TABLEAU`` are provided as the classic
    constants: ``rk(P, blk, RK4_TABLEAU)`` builds the identical final affine combination as
    ``rk4(P, blk)`` (a permutation of the same ``U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)`` inputs),
    and ``rk(P, blk, SSPRK2_TABLEAU)`` matches Heun's ``U + dt(1/2 k1 + 1/2 k2)``."""
    if tableau is None:
        raise TypeError("rk: tableau is required")
    if not isinstance(tableau, ButcherTableau):
        A, b, c = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(A, b, c)
    return _rk_from_tableau(P, block, state, tableau, sources, flux)


@program_macro
def explicit_rk(P: Any, block: Any, state: Any = None, *,
                rhs_operator: Any, fields_operator: Any = None,
                tableau: Any = None, A: Any = None, b: Any = None, c: Any = None,
                ) -> Any:
    """Generic explicit Runge-Kutta over a typed rate operator (Spec 2, operator-first).

    Each stage is ``k_i = rhs_operator(U_i[, fields_operator(U_i)])``; the tableau lowers to the same
    affine stage chain as :func:`rk`. ``rhs_operator`` / ``fields_operator`` are typed
    :class:`pops.model.OperatorHandle` selectors (from ``m.rate`` / ``m.field_solve``), not name
    strings. Pass a ``ButcherTableau`` / ``(A, b, c)`` via ``tableau`` or the raw ``A`` / ``b`` / ``c``.
    ``fields_operator`` is optional (a pure-flux rate needs no fields).
    """
    rhs_operator = _operator_handle(rhs_operator, "rhs_operator")
    if fields_operator is not None:
        fields_operator = _operator_handle(fields_operator, "fields_operator")
    if tableau is None:
        if A is None or b is None:
            raise ValueError("explicit_rk: provide a tableau or A and b")
        tableau = ButcherTableau(A, b, c)
    elif not isinstance(tableau, ButcherTableau):
        ta, tb, tc = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(ta, tb, tc)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    temporal = _time_state(P, block, state)
    u0 = temporal.n
    ks: list[Any] = []
    for i in range(tableau.stages):
        point = _stage_point(P, "%sstage_%d" % (tag, i), tableau.c[i])
        if i == 0:
            u_i = u0
        else:
            expr = u0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0:
                    expr = expr + (P.dt * aij) * ks[j]
            u_i = P.linear_combine("%sU%d" % (tag, i), expr, at=point)
        if fields_operator is not None:
            f_i = _opcall(P, fields_operator, u_i, point=point)
            ks.append(_opcall(
                P, rhs_operator, u_i, f_i,
                value_name="%sk%d" % (tag, i), point=point))
        else:
            ks.append(_opcall(
                P, rhs_operator, u_i, value_name="%sk%d" % (tag, i), point=point))
    final = u0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0:
            final = final + (P.dt * bi) * ks[i]
    _commit(P, temporal, P.linear_combine(
        "%sstep" % tag, final, at=temporal.next.point))
