"""pops.lib.time.ssprk -- Strong Stability Preserving Runge-Kutta schemes (SSPRK2 / SSPRK3)."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._factory import call_at, instance_state, operator_handle, program_factory
from ._helpers import _DEFAULT_SOURCES, _stage_point, program_macro
from .rk import ButcherTableau, _rk_from_tableau


SSPRK3_TABLEAU = ButcherTableau(
    A=[[], [1], [Fraction(1, 4), Fraction(1, 4)]],
    b=[Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)],
    c=[0, 1, Fraction(1, 2)],
    name="ssprk3",
)


def _build_ssprk2(program: Any, state: Any, rate: Any, fields: Any) -> None:
    rate = operator_handle(rate, "SSPRK2 rate")
    if fields is not None:
        fields = operator_handle(fields, "SSPRK2 fields")
    temporal = instance_state(program, state, "SSPRK2")
    u0 = temporal.n
    point0 = _stage_point(program, "ssprk2_stage_0", 0)
    fields0 = call_at(program, fields, u0, name="ssprk2_fields_0", point=point0) \
        if fields is not None else None
    k0 = call_at(program, rate, u0, fields0, name="ssprk2_k_0", point=point0)
    point1 = _stage_point(program, "ssprk2_stage_1", 1)
    u1 = program.value("ssprk2_U1", u0 + program.dt * k0, at=point1)
    fields1 = call_at(program, fields, u1, name="ssprk2_fields_1", point=point1) \
        if fields is not None else None
    k1 = call_at(program, rate, u1, fields1, name="ssprk2_k_1", point=point1)
    half = Fraction(1, 2)
    out = program.value(
        "ssprk2_step",
        u0 + (program.dt * half) * k0 + (program.dt * half) * k1,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)


def SSPRK2(state: Any, *, rate: Any, fields: Any = None) -> Any:
    """Return the ordinary SSPRK2 Program for one exact ``block[state]`` handle.

    ``rate`` and optional ``fields`` are exact owner-qualified operator handles. The returned value
    is a normal :class:`pops.time.Program`; no preset-specific executor or scheme class exists.
    """
    return program_factory("SSPRK2", _build_ssprk2, state, rate, fields)


@program_macro
def ssprk3(P: Any, block: Any, state: Any = None, *,
           sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    return _rk_from_tableau(P, block, state, SSPRK3_TABLEAU, sources, flux)


__all__ = ["SSPRK2", "SSPRK3_TABLEAU", "ssprk3"]
