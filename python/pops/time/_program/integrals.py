"""Authored persistent scalar integrals delivered by exact accepted external traces."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pops.time.values import ProgramValue


@dataclass(frozen=True, slots=True, eq=False)
class IntegralState:
    program: Any
    name: str

    @property
    def identity(self) -> str:
        from pops.identity.semantic import semantic_identity_of

        return "pops.integral.v1/" + semantic_identity_of(program=self.program).token + "/" + self.name


class _ProgramIntegrals:
    def integral_state(self, name: str, *, initial: float) -> IntegralState:
        """Declare a global persistent scalar, independent of cell-state and diagnostic storage."""
        self._guard_mutable("declare integral state")
        if type(name) is not str or not name or "/" in name:
            raise ValueError("integral_state requires a non-empty unqualified name")
        if type(initial) not in (int, float) or not math.isfinite(float(initial)):
            raise ValueError("integral_state initial value must be finite")
        if name in self._integral_states:
            raise ValueError("integral state is already declared in this Program")
        self._integral_states[name] = float(initial)
        return IntegralState(self, name)

    def accept_external_trace(
        self, state: IntegralState, *, rate: ProgramValue, axis: int, side: int,
        component: int, scale: float = 1.0,
    ) -> None:
        """Deliver an accepted conservative rate's selected physical face to an integral state.

        ``scale`` is authored conversion from the signed outward/inward face amount to the
        integral quantity, e.g. 1/C for a normalized capacitive state. A rate must resolve to
        one exact conservative occurrence; the compiler refuses any other realization.
        """
        self._guard_mutable("accept external trace")
        if type(state) is not IntegralState or state.program is not self \
                or state.name not in self._integral_states:
            raise ValueError("accepted trace requires an integral state owned by this Program")
        if type(rate) is not ProgramValue or rate.prog is not self or rate.op != "rhs":
            raise ValueError("accepted trace requires this Program's conservative RHS value")
        if type(axis) is not int or axis < 0 or type(side) is not int or side not in (0, 1) \
                or type(component) is not int or component < 0:
            raise ValueError("accepted trace requires exact axis/side/component indices")
        if type(scale) not in (int, float) or not math.isfinite(float(scale)) or float(scale) == 0.:
            raise ValueError("accepted trace scale must be finite and nonzero")
        row = (state.name, rate.id, axis, side, component, float(scale))
        if row[1:5] in (other[1:5] for other in self._integral_transfers):
            raise ValueError("accepted external trace already has a persistent-state consumer")
        self._integral_transfers.append(row)


__all__ = ["IntegralState"]
