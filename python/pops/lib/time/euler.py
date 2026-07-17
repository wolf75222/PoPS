"""Canonical forward-Euler Program factory."""
from __future__ import annotations

from typing import Any

from pops.time._methods.tableau import RungeKuttaTableau

from ._factory import program_factory, resolve_solve_action
from .rk import _build_explicit_runge_kutta


FORWARD_EULER_TABLEAU = RungeKuttaTableau(
    A=[[]], b=[1], c=[0], name="forward_euler")


def ForwardEuler(state: Any, *, rate: Any, fields: Any = None, solve_action: Any = None) -> Any:
    """Return an ordinary first-order explicit Program."""
    action = resolve_solve_action(solve_action, "ForwardEuler")
    return program_factory(
        "ForwardEuler",
        _build_explicit_runge_kutta,
        state,
        rate,
        fields,
        FORWARD_EULER_TABLEAU,
        action,
    )


__all__ = ["FORWARD_EULER_TABLEAU", "ForwardEuler"]
