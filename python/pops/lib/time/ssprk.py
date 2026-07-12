"""pops.lib.time.ssprk -- Strong Stability Preserving Runge-Kutta schemes (SSPRK2 / SSPRK3)."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._helpers import (
    _DEFAULT_SOURCES, _commit, _stage_point, _stage_rhs, _time_state, program_macro,
)


@program_macro
def ssprk2(P: Any, block: Any, state: Any = None, *,
           sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """SSPRK2 (Heun / Shu-Osher): U1 = U0 + dt k0; U^{n+1} = 1/2 U0 + 1/2 (U1 + dt k1)."""
    temporal = _time_state(P, block, state)
    U0 = temporal.n
    k0 = _stage_rhs(P, U0, sources, flux, name="ssprk2_0", offset=0)
    point1 = _stage_point(P, "ssprk2_1", 1)
    U1 = P.linear_combine("ssprk2_U1", U0 + P.dt * k0, at=point1)
    k1 = _stage_rhs(P, U1, sources, flux, name="ssprk2_1", offset=1)
    _commit(P, temporal, P.linear_combine(
        "ssprk2_step", Fraction(1, 2) * U0 + Fraction(1, 2) * (U1 + P.dt * k1),
        at=temporal.next.point))


@program_macro
def ssprk3(P: Any, block: Any, state: Any = None, *,
           sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    temporal = _time_state(P, block, state)
    U0 = temporal.n
    k0 = _stage_rhs(P, U0, sources, flux, name="ssprk3_0", offset=0)
    point1 = _stage_point(P, "ssprk3_1", 1)
    U1 = P.linear_combine("ssprk3_U1", U0 + P.dt * k0, at=point1)
    k1 = _stage_rhs(P, U1, sources, flux, name="ssprk3_1", offset=1)
    point2 = _stage_point(P, "ssprk3_2", Fraction(1, 2))
    U2 = P.linear_combine(
        "ssprk3_U2", Fraction(3, 4) * U0 + Fraction(1, 4) * (U1 + P.dt * k1),
        at=point2)
    k2 = _stage_rhs(P, U2, sources, flux, name="ssprk3_2", offset=Fraction(1, 2))
    _commit(
        P,
        temporal,
        P.linear_combine(
            "ssprk3_step", Fraction(1, 3) * U0 + Fraction(2, 3) * (U2 + P.dt * k2),
            at=temporal.next.point,
        ),
    )
