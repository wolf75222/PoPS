"""Authored persistent scalar integrals delivered by exact accepted external traces."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pops.time.values import ProgramValue
from pops.time._authoring import atomic_authoring


def integral_units_bytes(units) -> str:
    import json
    from pops._ir.quantity import PhysicalDimension
    if type(units) is not PhysicalDimension:
        raise TypeError("integral units require an exact PhysicalDimension")
    encoded = json.dumps(units.to_data(), sort_keys=True, separators=(",", ":"))
    message = "integral units require a lossless canonical PhysicalDimension JSON roundtrip"
    try:
        decoded = PhysicalDimension.from_data(json.loads(encoded))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(message) from error
    if decoded != units:
        raise ValueError(message)
    return encoded


def integral_identity(program, name: str) -> str:
    from pops.identity.semantic import semantic_identity_of
    from hashlib import sha256
    units = program._integral_units.get(name)
    base = semantic_identity_of(program=program).token
    if units is None:
        return "pops.integral.v1/" + base + "/" + name
    digest = sha256(integral_units_bytes(units).encode("utf-8")).hexdigest()
    return "pops.integral.v2/" + base + "/" + digest + "/" + name


@dataclass(frozen=True, slots=True, eq=False)
class IntegralState:
    program: Any
    name: str

    @property
    def identity(self) -> str:
        return integral_identity(self.program, self.name)


class _ProgramIntegrals:
    def integral_state(self, name: str, *, initial: float, units=None) -> IntegralState:
        """Declare a global persistent scalar, independent of cell-state and diagnostic storage."""
        self._guard_mutable("declare integral state")
        if type(name) is not str or not name or "/" in name or "\0" in name:
            raise ValueError("integral_state requires a non-empty unqualified name without NUL")
        if type(initial) not in (int, float) or not math.isfinite(float(initial)):
            raise ValueError("integral_state initial value must be finite")
        if name in self._integral_states:
            raise ValueError("integral state is already declared in this Program")
        from pops._ir.quantity import PhysicalDimension
        if units is not None and type(units) is not PhysicalDimension:
            raise TypeError("integral units require an exact PhysicalDimension")
        if units is not None:
            integral_units_bytes(units)
        self._integral_states[name] = float(initial)
        if units is not None:
            self._integral_units[name] = units
        return IntegralState(self, name)

    @atomic_authoring
    def integral_value(self, state: IntegralState, *, at, scope: str):
        """Capture a global candidate in an explicit Program pointwise expression.

        Direct Equation/FieldProblem argument capture is not realized by this API.
        Candidate scope participates in the enclosing transaction and can be revoked.
        """
        self._guard_mutable("capture integral candidate")
        if type(state) is not IntegralState or state.program is not self \
                or state.name not in self._integral_states:
            raise ValueError("integral capture requires this Program's exact IntegralState")
        if scope != "candidate":
            raise ValueError("integral capture currently realizes explicit candidate scope")
        units = self._integral_units.get(state.name)
        if units is None:
            raise ValueError("integral capture requires explicitly declared physical units")
        from pops.time._program.value_validation import point_clock
        if point_clock(at, "integral capture") != self.clock:
            raise ValueError("integral capture requires this Program's exact clock")
        from pops.time.expressions import ProgramGlobal
        value = self._new("scalar", "integral_candidate", (), {
            "integral": state.name, "units": integral_units_bytes(units),
            "scope": scope, "capture_version": 1,
        }, None, None, point=at)
        from pops.time._evaluation_point import evaluation_stage_fraction
        evaluation_stage_fraction(value)
        return ProgramGlobal(value)

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
        if type(rate) is not ProgramValue or rate.prog is not self \
                or rate.op not in {"rhs", "diffusive_rhs"}:
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
