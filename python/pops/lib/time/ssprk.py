"""pops.lib.time.ssprk -- Strong Stability Preserving Runge-Kutta schemes (SSPRK2 / SSPRK3)."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._helpers import _DEFAULT_SOURCES, program_macro
from .rk import ButcherTableau, SSPRK2_TABLEAU, _rk_from_tableau


SSPRK3_TABLEAU = ButcherTableau(
    A=[[], [1], [Fraction(1, 4), Fraction(1, 4)]],
    b=[Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)],
    c=[0, 1, Fraction(1, 2)],
    name="ssprk3",
)


@program_macro
def SSPRK2(P: Any, block: Any, state: Any = None, *,
           sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """SSPRK2 (Heun / Shu-Osher): U1 = U0 + dt k0; U^{n+1} = 1/2 U0 + 1/2 (U1 + dt k1)."""
    return _rk_from_tableau(P, block, state, SSPRK2_TABLEAU, sources, flux)


@program_macro
def ssprk3(P: Any, block: Any, state: Any = None, *,
           sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    return _rk_from_tableau(P, block, state, SSPRK3_TABLEAU, sources, flux)


__all__ = ["SSPRK2", "SSPRK3_TABLEAU", "ssprk3"]
