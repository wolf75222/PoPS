"""Generic admissibility, explicit projection, and enforcement scheduling contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from pops.identity import canonical_bytes, make_identity

from .inversion import _capture, _plain, _text


_SCHEMA_VERSION = 1


class ConstraintKind(str, Enum):
    FINITE = "finite"
    POSITIVE = "positive"
    REALIZABILITY = "realizability"
    CUSTOM_INEQUALITY = "custom_inequality"


@dataclass(frozen=True, slots=True)
class AdmissibilityConstraint:
    """One named, observable model inequality without a physical-component vocabulary."""

    constraint_id: str
    diagnostic_code: int
    kind: ConstraintKind
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.constraint_id, where="admissibility constraint_id")
        if type(self.diagnostic_code) is not int or self.diagnostic_code < 1:
            raise ValueError("admissibility diagnostic_code must be a positive exact integer")
        if type(self.kind) is not ConstraintKind:
            raise TypeError("admissibility constraint kind must be ConstraintKind")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("admissibility constraint parameters must be a mapping")
        object.__setattr__(
            self,
            "parameters",
            _capture(self.parameters, where="admissibility constraint parameters"),
        )

    @classmethod
    def finite(
        cls, constraint_id: str, diagnostic_code: int, *, components: tuple[int, ...]
    ) -> AdmissibilityConstraint:
        if type(components) is not tuple or not components:
            raise TypeError("finite admissibility components must be a non-empty exact tuple")
        if any(type(component) is not int or component < 0 for component in components):
            raise ValueError("finite admissibility component indices must be non-negative")
        if len(set(components)) != len(components):
            raise ValueError("finite admissibility component collision")
        return cls(
            constraint_id, diagnostic_code, ConstraintKind.FINITE, {"components": components}
        )

    @classmethod
    def positive(
        cls, constraint_id: str, diagnostic_code: int, *, component: int, lower_bound: float = 0.0
    ) -> AdmissibilityConstraint:
        if type(component) is not int or component < 0:
            raise ValueError("positive admissibility component must be non-negative")
        if type(lower_bound) is not float:
            raise TypeError("positive admissibility lower_bound must be an exact float")
        if not math.isfinite(lower_bound):
            raise ValueError("positive admissibility lower_bound must be finite")
        return cls(
            constraint_id,
            diagnostic_code,
            ConstraintKind.POSITIVE,
            {"component": component, "lower_bound": lower_bound},
        )

    @classmethod
    def realizability(
        cls, constraint_id: str, diagnostic_code: int, *, provider: str, **parameters: Any
    ) -> AdmissibilityConstraint:
        _text(provider, where="realizability provider")
        return cls(
            constraint_id,
            diagnostic_code,
            ConstraintKind.REALIZABILITY,
            {"provider": provider, "parameters": parameters},
        )

    @classmethod
    def custom(
        cls, constraint_id: str, diagnostic_code: int, *, provider: str, **parameters: Any
    ) -> AdmissibilityConstraint:
        _text(provider, where="custom inequality provider")
        return cls(
            constraint_id,
            diagnostic_code,
            ConstraintKind.CUSTOM_INEQUALITY,
            {"provider": provider, "parameters": parameters},
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "diagnostic_code": self.diagnostic_code,
            "kind": self.kind.value,
            "parameters": _plain(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class AdmissibleSet:
    """Ordered model declaration; first failure and its diagnostic are deterministic."""

    constraints: tuple[AdmissibilityConstraint, ...]

    def __post_init__(self) -> None:
        if type(self.constraints) is not tuple or not self.constraints:
            raise TypeError("AdmissibleSet.constraints must be a non-empty exact tuple")
        if any(type(item) is not AdmissibilityConstraint for item in self.constraints):
            raise TypeError("AdmissibleSet requires exact AdmissibilityConstraint records")
        ids = [item.constraint_id for item in self.constraints]
        codes = [item.diagnostic_code for item in self.constraints]
        if len(set(ids)) != len(ids):
            raise ValueError("admissibility constraint id collision")
        if len(set(codes)) != len(codes):
            raise ValueError("admissibility diagnostic code collision")

    @classmethod
    def declare(cls, *constraints: AdmissibilityConstraint) -> AdmissibleSet:
        return cls(tuple(constraints))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "constraints": [constraint.to_data() for constraint in self.constraints],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_data())

    @property
    def identity(self) -> str:
        return make_identity("admissible-set", self.to_data()).token


@dataclass(frozen=True, slots=True)
class ProjectionProvider:
    """Authenticated explicit projection that returns a detached candidate downstream."""

    provider_id: str
    interface_version: int
    dimension: int
    candidate_identity: str
    inputs_identity: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.provider_id, where="projection provider_id")
        if type(self.interface_version) is not int or self.interface_version < 1:
            raise ValueError("projection provider interface_version must be positive")
        if type(self.dimension) is not int or self.dimension not in (1, 2, 3):
            raise ValueError("projection provider dimension must be exactly 1, 2, or 3")
        _text(self.candidate_identity, where="projection candidate_identity")
        _text(self.inputs_identity, where="projection inputs_identity")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("projection provider parameters must be a mapping")
        object.__setattr__(
            self,
            "parameters",
            _capture(self.parameters, where="projection provider parameters"),
        )

    @classmethod
    def author(
        cls,
        provider_id: str,
        *,
        dimension: int,
        candidate_identity: str,
        inputs_identity: str,
        interface_version: int = 1,
        **parameters: Any,
    ) -> ProjectionProvider:
        return cls(
            provider_id,
            interface_version,
            dimension,
            candidate_identity,
            inputs_identity,
            parameters,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "interface_version": self.interface_version,
            "dimension": self.dimension,
            "candidate_identity": self.candidate_identity,
            "inputs_identity": self.inputs_identity,
            "parameters": _plain(self.parameters),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_data())

    @property
    def identity(self) -> str:
        return make_identity("projection-provider", self.to_data()).token


class EnforcementPhase(str, Enum):
    INITIALIZATION = "initialization"
    RECONSTRUCTION = "reconstruction"
    SOURCE_SOLVE = "source_solve"
    BOUNDARY = "boundary"
    ACCEPTANCE = "acceptance"


@dataclass(frozen=True, slots=True)
class EnforcementRule:
    check: bool
    project_if_invalid: bool = False

    def __post_init__(self) -> None:
        if type(self.check) is not bool or type(self.project_if_invalid) is not bool:
            raise TypeError("enforcement rule fields must be exact bools")
        if self.project_if_invalid and not self.check:
            raise ValueError("scheduled projection requires an admissibility check")

    def to_data(self) -> dict[str, bool]:
        return {"check": self.check, "project_if_invalid": self.project_if_invalid}


@dataclass(frozen=True, slots=True)
class EnforcementSchedule:
    """Exact five-phase schedule, serialized in protocol order rather than mapping order."""

    initialization: EnforcementRule
    reconstruction: EnforcementRule
    source_solve: EnforcementRule
    boundary: EnforcementRule
    acceptance: EnforcementRule

    def __post_init__(self) -> None:
        if any(type(rule) is not EnforcementRule for _phase, rule in self.rules()):
            raise TypeError("EnforcementSchedule phases require exact EnforcementRule records")

    @classmethod
    def from_mapping(cls, rules: Mapping[EnforcementPhase, EnforcementRule]) -> EnforcementSchedule:
        if not isinstance(rules, Mapping):
            raise TypeError("enforcement schedule rules must be a mapping")
        expected = set(EnforcementPhase)
        if set(rules) != expected:
            raise ValueError("enforcement schedule must declare every phase exactly once")
        return cls(*(rules[phase] for phase in EnforcementPhase))

    def rules(self) -> tuple[tuple[EnforcementPhase, EnforcementRule], ...]:
        return tuple(
            zip(
                EnforcementPhase,
                (
                    self.initialization,
                    self.reconstruction,
                    self.source_solve,
                    self.boundary,
                    self.acceptance,
                ),
                strict=True,
            )
        )

    def at(self, phase: EnforcementPhase) -> EnforcementRule:
        if type(phase) is not EnforcementPhase:
            raise TypeError("schedule phase must be EnforcementPhase")
        return dict(self.rules())[phase]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "phases": [{"phase": phase.value, **rule.to_data()} for phase, rule in self.rules()],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_data())

    @property
    def identity(self) -> str:
        return make_identity("enforcement-schedule", self.to_data()).token


__all__ = [
    "AdmissibilityConstraint",
    "AdmissibleSet",
    "ConstraintKind",
    "EnforcementPhase",
    "EnforcementRule",
    "EnforcementSchedule",
    "ProjectionProvider",
]
