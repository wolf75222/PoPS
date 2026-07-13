"""Typed boundary needs and complete provider dependency declarations."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING, Any, ClassVar

from .topology import BoundaryHandle

if TYPE_CHECKING:
    from pops.model import Handle, ParamHandle


_SCHEMA_VERSION = 1


def _handle(value: Any, *, where: str, kinds: frozenset[str]) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle, never a string" % where)
    if value.kind not in kinds:
        raise TypeError("%s requires Handle.kind in %s, got %r" %
                        (where, sorted(kinds), value.kind))
    identity = value.canonical_identity()
    if identity.get("qualified_id") != value.qualified_id:
        raise ValueError("%s Handle identity does not authenticate qualified_id" % where)
    return value


def _unique_handles(values: Any, *, where: str, kinds: frozenset[str]) -> tuple[Handle, ...]:
    if not isinstance(values, tuple):
        raise TypeError("%s must be a tuple" % where)
    rows = tuple(_handle(value, where=where, kinds=kinds) for value in values)
    if len(rows) != len(set(rows)):
        raise ValueError("%s contains double dependencies" % where)
    return rows


@dataclass(frozen=True, slots=True)
class BoundaryPort:
    boundary: BoundaryHandle
    subject: Handle
    representation: Handle

    port_type: ClassVar[str]
    subject_kinds: ClassVar[frozenset[str]]

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, BoundaryHandle):
            raise TypeError("BoundaryPort.boundary must be a BoundaryHandle")
        _handle(self.subject, where="%s.subject" % type(self).__name__,
                kinds=self.subject_kinds)
        _handle(self.representation, where="%s.representation" % type(self).__name__,
                kinds=frozenset(("representation",)))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "port_type": self.port_type,
                "boundary": self.boundary.canonical_identity(),
                "subject": self.subject.canonical_identity(),
                "representation": self.representation.canonical_identity()}

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "boundary_port", "canonical_id": self.canonical_id,
                **self.canonical_identity()}


@dataclass(frozen=True, slots=True)
class GhostState(BoundaryPort):
    port_type = "ghost_state"
    subject_kinds = frozenset(("state",))


@dataclass(frozen=True, slots=True)
class ExteriorTrace(BoundaryPort):
    port_type = "exterior_trace"
    subject_kinds = frozenset(("state", "field"))


@dataclass(frozen=True, slots=True)
class NumericalFlux(BoundaryPort):
    port_type = "numerical_flux"
    subject_kinds = frozenset(("state",))


@dataclass(frozen=True, slots=True)
class ConstraintResidual(BoundaryPort):
    port_type = "constraint_residual"
    subject_kinds = frozenset(("state", "field"))


class ClosureMode(Enum):
    NONE = "none"
    CHARACTERISTIC = "characteristic"
    DIRECTIONAL = "directional"


class SignDependence(Enum):
    FIXED = "fixed"
    RUNTIME = "runtime"
    SPATIAL = "spatial"
    RUNTIME_SPATIAL = "runtime_spatial"

    @property
    def requires_runtime(self) -> bool:
        return self in (SignDependence.RUNTIME, SignDependence.RUNTIME_SPATIAL)

    @property
    def requires_spatial(self) -> bool:
        return self in (SignDependence.SPATIAL, SignDependence.RUNTIME_SPATIAL)


class SonicPolicy(Enum):
    NEUTRAL = "neutral"
    ERROR = "error"


class IncomingMultiplicity(Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"


@dataclass(frozen=True, slots=True)
class CharacteristicClosure:
    mode: ClosureMode
    sign_dependence: SignDependence
    sonic: SonicPolicy
    incoming: IncomingMultiplicity
    characteristics: tuple[Handle, ...]

    def __post_init__(self) -> None:
        for value, expected, name in (
            (self.mode, ClosureMode, "mode"),
            (self.sign_dependence, SignDependence, "sign_dependence"),
            (self.sonic, SonicPolicy, "sonic"),
            (self.incoming, IncomingMultiplicity, "incoming"),
        ):
            if not isinstance(value, expected):
                raise TypeError("CharacteristicClosure.%s must be %s" %
                                (name, expected.__name__))
        rows = _unique_handles(
            self.characteristics, where="CharacteristicClosure.characteristics",
            kinds=frozenset(("state", "field")))
        if self.mode is ClosureMode.NONE and rows:
            raise ValueError("ClosureMode.NONE cannot carry characteristic data")
        if self.mode is ClosureMode.NONE and (
                self.sign_dependence is not SignDependence.FIXED or
                self.sonic is not SonicPolicy.NEUTRAL or
                self.incoming is not IncomingMultiplicity.SINGLE):
            raise ValueError("ClosureMode.NONE requires fixed/neutral/single closure semantics")
        if self.mode is not ClosureMode.NONE and not rows:
            raise ValueError("characteristic closure requires explicit characteristic data")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "mode": self.mode.value,
                "sign_dependence": self.sign_dependence.value, "sonic": self.sonic.value,
                "incoming": self.incoming.value,
                "characteristics": [row.canonical_identity() for row in self.characteristics]}


@dataclass(frozen=True, slots=True)
class RepresentationFlow:
    source: Handle
    target: Handle
    converter: Handle | None

    def __post_init__(self) -> None:
        _handle(self.source, where="RepresentationFlow.source",
                kinds=frozenset(("representation",)))
        _handle(self.target, where="RepresentationFlow.target",
                kinds=frozenset(("representation",)))
        if self.source == self.target and self.converter is not None:
            raise ValueError("identity representation flow must not invent a converter")
        if self.source != self.target:
            if self.converter is None:
                raise ValueError("primitive->conservative conversion requires an explicit provider")
            _handle(self.converter, where="RepresentationFlow.converter",
                    kinds=frozenset(("representation_conversion",)))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION,
                "source": self.source.canonical_identity(),
                "target": self.target.canonical_identity(),
                "converter": (None if self.converter is None else
                              self.converter.canonical_identity())}


@dataclass(frozen=True, slots=True)
class BoundaryDependencies:
    states: tuple[Handle, ...]
    fields: tuple[Handle, ...]
    time: tuple[Handle, ...]
    runtime_params: tuple[ParamHandle, ...]
    representation: RepresentationFlow
    characteristic: CharacteristicClosure

    def __post_init__(self) -> None:
        states = _unique_handles(self.states, where="BoundaryDependencies.states",
                                 kinds=frozenset(("state",)))
        fields = _unique_handles(self.fields, where="BoundaryDependencies.fields",
                                 kinds=frozenset(("field",)))
        time = _unique_handles(self.time, where="BoundaryDependencies.time",
                               kinds=frozenset(("time",)))
        params = _unique_handles(self.runtime_params, where="BoundaryDependencies.runtime_params",
                                 kinds=frozenset(("parameter",)))
        if any(getattr(row, "param_kind", None) != "runtime" for row in params):
            raise TypeError("BoundaryDependencies.runtime_params requires RuntimeParam handles")
        if not isinstance(self.representation, RepresentationFlow):
            raise TypeError("BoundaryDependencies.representation must be a RepresentationFlow")
        if not isinstance(self.characteristic, CharacteristicClosure):
            raise TypeError("BoundaryDependencies.characteristic must be explicit")
        dependence = self.characteristic.sign_dependence
        if dependence.requires_runtime and not params:
            raise ValueError("runtime-varying transport sign requires a RuntimeParam dependency")
        if dependence.requires_spatial and not (self.states or self.fields):
            raise ValueError("spatially varying transport sign requires a state/field dependency")
        for key, rows in (("states", states), ("fields", fields), ("time", time),
                          ("runtime_params", params)):
            object.__setattr__(self, key, tuple(sorted(rows, key=lambda row: row.qualified_id)))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION,
                "states": [row.canonical_identity() for row in self.states],
                "fields": [row.canonical_identity() for row in self.fields],
                "time": [row.canonical_identity() for row in self.time],
                "runtime_params": [row.canonical_identity() for row in self.runtime_params],
                "representation": self.representation.canonical_identity(),
                "characteristic": self.characteristic.canonical_identity()}


__all__ = [
    "BoundaryDependencies", "BoundaryPort", "CharacteristicClosure", "ClosureMode",
    "ConstraintResidual", "ExteriorTrace", "GhostState", "IncomingMultiplicity",
    "NumericalFlux", "RepresentationFlow", "SignDependence", "SonicPolicy",
]
