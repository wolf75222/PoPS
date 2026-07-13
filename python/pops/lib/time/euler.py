"""Canonical forward-Euler Program factory."""
from __future__ import annotations

from typing import Any

from pops.time.method_tableau import RungeKuttaTableau

from ._factory import program_factory
from .rk import _build_explicit_runge_kutta


FORWARD_EULER_TABLEAU = RungeKuttaTableau(
    A=[[]], b=[1], c=[0], name="forward_euler")


def ForwardEuler(state: Any, *, rate: Any, fields: Any = None) -> Any:
    """Return an ordinary first-order explicit Program."""
    return program_factory(
        "ForwardEuler",
        _build_explicit_runge_kutta,
        state,
        rate,
        fields,
        FORWARD_EULER_TABLEAU,
    )


__all__ = ["FORWARD_EULER_TABLEAU", "ForwardEuler"]
