"""pops.lib.time.ssprk -- Strong Stability Preserving Runge-Kutta schemes (SSPRK2 / SSPRK3)."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._helpers import _stage_rhs, program_macro


@program_macro
def ssprk2(P: Any, block: Any, *, sources: Any = ("default",), flux: Any = True) -> Any:
    """SSPRK2 (Heun / Shu-Osher): U1 = U0 + dt k0; U^{n+1} = 1/2 U0 + 1/2 (U1 + dt k1)."""
    U0 = P.state(block)
    k0 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("ssprk2_U1", U0 + P.dt * k0)
    k1 = _stage_rhs(P, U1, sources, flux)
    P._commit_block(block, P.linear_combine(
        "ssprk2_step", Fraction(1, 2) * U0 + Fraction(1, 2) * (U1 + P.dt * k1)))


@program_macro
def ssprk3(P: Any, block: Any, *, sources: Any = ("default",), flux: Any = True) -> Any:
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    U0 = P.state(block)
    k0 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("ssprk3_U1", U0 + P.dt * k0)
    k1 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine(
        "ssprk3_U2", Fraction(3, 4) * U0 + Fraction(1, 4) * (U1 + P.dt * k1))
    k2 = _stage_rhs(P, U2, sources, flux)
    P._commit_block(
        block,
        P.linear_combine(
            "ssprk3_step", Fraction(1, 3) * U0 + Fraction(2, 3) * (U2 + P.dt * k2)),
    )
