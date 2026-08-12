"""Canonical forward-Euler Program factory."""

from __future__ import annotations

from typing import Any

from pops.time._methods.tableau import RungeKuttaTableau

from .rk import _runge_kutta


FORWARD_EULER_TABLEAU = RungeKuttaTableau(A=[[]], b=[1], c=[0], name="forward_euler")


def ForwardEuler(state: Any, *, rate: Any, fields: Any = None, solve_action: Any = None) -> Any:
    """Return an ordinary first-order explicit Program."""
    return _runge_kutta(
        state,
        rate=rate,
        fields=fields,
        tableau=FORWARD_EULER_TABLEAU,
        solve_action=solve_action,
        program_name="ForwardEuler",
        where="ForwardEuler",
    )


__all__ = ["FORWARD_EULER_TABLEAU", "ForwardEuler"]
