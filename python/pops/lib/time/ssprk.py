"""Strong-stability-preserving Runge--Kutta Program factories."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.time.method_tableau import RungeKuttaTableau

from ._factory import program_factory
from .rk import SSPRK2_TABLEAU, _build_explicit_runge_kutta


SSPRK3_TABLEAU = RungeKuttaTableau(
    A=[[], [1], [Fraction(1, 4), Fraction(1, 4)]],
    b=[Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)],
    c=[0, 1, Fraction(1, 2)],
    name="ssprk3",
)


def SSPRK2(state: Any, *, rate: Any, fields: Any = None) -> Any:
    """Return the ordinary two-stage, second-order SSP Program."""
    return program_factory(
        "SSPRK2", _build_explicit_runge_kutta,
        state, rate, fields, SSPRK2_TABLEAU,
    )


def SSPRK3(state: Any, *, rate: Any, fields: Any = None) -> Any:
    """Return the ordinary three-stage, third-order SSP Program."""
    return program_factory(
        "SSPRK3", _build_explicit_runge_kutta,
        state, rate, fields, SSPRK3_TABLEAU,
    )


__all__ = ["SSPRK2", "SSPRK2_TABLEAU", "SSPRK3", "SSPRK3_TABLEAU"]
