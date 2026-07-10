"""pops.lib.time.rk -- Classic explicit Runge-Kutta schemes (RK4, generic rk) and Butcher tableaux.

Exports: rk4, rk, explicit_rk, ButcherTableau, RK4_TABLEAU, SSPRK2_TABLEAU.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ._helpers import (
    _exact_coefficient, _exact_fraction, _opcall, _operator_handle, _stage_rhs,
    program_macro,
)


@program_macro
def rk4(P: Any, block: Any, *, sources: Any = ("default",), flux: Any = True) -> Any:
    """Classic RK4, expressed with NO special RK4 class (spec acceptance 29):
    U^{n+1} = U0 + dt/6 (k1 + 2 k2 + 2 k3 + k4)."""
    U0 = P.state(block)
    k1 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("rk4_U1", U0 + Fraction(1, 2) * P.dt * k1)
    k2 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine("rk4_U2", U0 + Fraction(1, 2) * P.dt * k2)
    k3 = _stage_rhs(P, U2, sources, flux)
    U3 = P.linear_combine("rk4_U3", U0 + P.dt * k3)
    k4 = _stage_rhs(P, U3, sources, flux)
    P._commit_block(block, P.linear_combine(
        "rk4_step",
        U0 + Fraction(1, 6) * P.dt * k1 + Fraction(1, 3) * P.dt * k2
        + Fraction(1, 3) * P.dt * k3 + Fraction(1, 6) * P.dt * k4))


# Classic explicit Butcher tableaux (A lower-triangular, b weights, c nodes) for `rk` (ADC-423).
@dataclass(frozen=True, slots=True, init=False)
class ButcherTableau:
    """An explicit Butcher tableau ``(A, b, c)`` for `rk`: ``A`` is strictly lower-triangular (stage i
    depends only on stages j < i), ``b`` the final weights, ``c`` the (unused-by-the-lowering) nodes.
    Coefficients retain their authoring domain (integer, rational, decimal or
    binary64); validation compares their exact rational values, never a tolerance.
    The stored representation is immutable and canonical: rows keep only their
    strictly-lower entries, while an equivalent full square matrix with zero
    diagonal/upper entries is accepted and normalized to the same value.
    """

    A: tuple[tuple[Any, ...], ...]
    b: tuple[Any, ...]
    c: tuple[Any, ...]
    name: str | None

    def __init__(self, A: Any, b: Any, c: Any = None, name: Any = None) -> None:
        try:
            weights = tuple(
                _exact_coefficient(value, "ButcherTableau.b[%d]" % i)
                for i, value in enumerate(b))
            rows = tuple(tuple(row) for row in A)
        except TypeError as exc:
            raise TypeError("ButcherTableau: A and b must be finite coefficient sequences") from exc
        s = len(weights)
        if s == 0:
            raise ValueError("ButcherTableau: at least one stage is required")
        if len(rows) != s:
            raise ValueError("ButcherTableau: A, b, c must share the stage count")

        lower_rows = []
        row_sums = []
        for i, raw_row in enumerate(rows):
            if len(raw_row) < i or len(raw_row) > s:
                raise ValueError(
                    "ButcherTableau: row %d must provide its %d lower coefficient(s), optionally "
                    "followed by zero diagonal/upper entries" % (i, i))
            row = tuple(
                _exact_coefficient(value, "ButcherTableau.A[%d][%d]" % (i, j))
                for j, value in enumerate(raw_row))
            if any(value != 0 for value in row[i:]):
                raise ValueError(
                    "ButcherTableau: A must be strictly lower-triangular (stage %d reads stage >= %d); "
                    "rk lowers EXPLICIT tableaux only" % (i, i))
            lower = row[:i]
            lower_rows.append(lower)
            row_sums.append(sum(
                (_exact_fraction(value, "ButcherTableau.A[%d]" % i) for value in lower),
                Fraction(0, 1)))

        weight_sum = sum(
            (_exact_fraction(value, "ButcherTableau.b") for value in weights), Fraction(0, 1))
        if weight_sum != 1:
            raise ValueError(
                "ButcherTableau: weights b must sum exactly to 1 (got %r)" % weight_sum)

        if c is None:
            nodes = tuple(row_sums)
        else:
            try:
                nodes = tuple(
                    _exact_coefficient(value, "ButcherTableau.c[%d]" % i)
                    for i, value in enumerate(c))
            except TypeError as exc:
                raise TypeError("ButcherTableau: c must be a finite coefficient sequence") from exc
            if len(nodes) != s:
                raise ValueError("ButcherTableau: A, b, c must share the stage count")
            for i, (node, expected) in enumerate(zip(nodes, row_sums, strict=True)):
                if _exact_fraction(node, "ButcherTableau.c[%d]" % i) != expected:
                    raise ValueError(
                        "ButcherTableau: c[%d] must equal the exact row sum of A[%d] (%r); got %r"
                        % (i, i, expected, node))
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("ButcherTableau: name must be a non-empty string or None")

        object.__setattr__(self, "A", tuple(lower_rows))
        object.__setattr__(self, "b", weights)
        object.__setattr__(self, "c", nodes)
        object.__setattr__(self, "name", name)

    @property
    def stages(self) -> int:
        return len(self.b)


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


@program_macro
def rk(P: Any, block: Any, tableau: Any, *, sources: Any = ("default",), flux: Any = True) -> Any:
    """Generic explicit Runge-Kutta from a Butcher @p tableau (ADC-423), lowered to the SAME stage chain
    the hard-coded `rk4` macro emits -- ``solve_fields`` + ``rhs`` + ``linear_combine``, no RK class:

        k_i      = R( U + dt * sum_{j<i} A[i][j] * k_j )       (the i-th stage RHS)
        U^{n+1}  = U + dt * sum_i b[i] * k_i

    @p tableau is a `ButcherTableau` (or a raw ``(A, b, c)`` triple); ``A`` must be strictly
    lower-triangular (explicit). ``RK4_TABLEAU`` and ``SSPRK2_TABLEAU`` are provided as the classic
    constants: ``rk(P, blk, RK4_TABLEAU)`` builds the identical final affine combination as
    ``rk4(P, blk)`` (a permutation of the same ``U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)`` inputs),
    and ``rk(P, blk, SSPRK2_TABLEAU)`` matches Heun's ``U + dt(1/2 k1 + 1/2 k2)``."""
    if not isinstance(tableau, ButcherTableau):
        A, b, c = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(A, b, c)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    U0 = P.state(block)
    ks: list[Any] = []
    for i in range(tableau.stages):
        if i == 0:
            Ui = U0  # the first stage reads U^n directly (no scratch combine, like rk4)
        else:
            expr = U0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0:
                    expr = expr + (P.dt * aij) * ks[j]
            Ui = P.linear_combine("%sU%d" % (tag, i), expr)
        ks.append(_stage_rhs(P, Ui, sources, flux))
    final = U0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0:
            final = final + (P.dt * bi) * ks[i]
    P._commit_block(block, P.linear_combine("%sstep" % tag, final))


@program_macro
def explicit_rk(P: Any, block: Any, *, rhs_operator: Any, fields_operator: Any = None,
                tableau: Any = None, A: Any = None, b: Any = None, c: Any = None,
                state_space: Any = "U") -> Any:
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
    u0 = P.state(block)
    ks: list[Any] = []
    for i in range(tableau.stages):
        if i == 0:
            u_i = u0
        else:
            expr = u0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0:
                    expr = expr + (P.dt * aij) * ks[j]
            u_i = P.linear_combine("%sU%d" % (tag, i), expr)
        if fields_operator is not None:
            f_i = _opcall(P, fields_operator, u_i)
            ks.append(_opcall(P, rhs_operator, u_i, f_i, value_name="%sk%d" % (tag, i)))
        else:
            ks.append(_opcall(P, rhs_operator, u_i, value_name="%sk%d" % (tag, i)))
    final = u0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0:
            final = final + (P.dt * bi) * ks[i]
    P._commit_block(block, P.linear_combine("%sstep" % tag, final))
