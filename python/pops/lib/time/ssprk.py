"""Strong-stability-preserving Runge--Kutta Program factories."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.time._methods.tableau import RungeKuttaTableau

from .rk import SSPRK2_TABLEAU, _runge_kutta


SSPRK3_TABLEAU = RungeKuttaTableau(
    A=[[], [1], [Fraction(1, 4), Fraction(1, 4)]],
    b=[Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)],
    c=[0, 1, Fraction(1, 2)],
    name="ssprk3",
)


def SSPRK2(state: Any, *, rate: Any, fields: Any = None, solve_action: Any = None) -> Any:
    """Return the ordinary two-stage, second-order SSP Program."""
    return _runge_kutta(
        state,
        rate=rate,
        fields=fields,
        tableau=SSPRK2_TABLEAU,
        solve_action=solve_action,
        program_name="SSPRK2",
        where="SSPRK2",
    )


def SSPRK3(state: Any, *, rate: Any, fields: Any = None, solve_action: Any = None) -> Any:
    """Return the ordinary three-stage, third-order SSP Program."""
    return _runge_kutta(
        state,
        rate=rate,
        fields=fields,
        tableau=SSPRK3_TABLEAU,
        solve_action=solve_action,
        program_name="SSPRK3",
        where="SSPRK3",
    )


__all__ = ["SSPRK2", "SSPRK2_TABLEAU", "SSPRK3", "SSPRK3_TABLEAU"]
