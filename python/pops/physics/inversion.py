"""Typed authoring contracts for fallible, prepared variable inversion.

These records are immutable semantic inputs to Uniform, AMR, native and compiled lowering.  They do
not execute an inversion in Python and do not grant publication authority to a provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pops.identity import canonical_bytes, make_identity


_SCHEMA_VERSION = 1


def _text(value: Any, *, where: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError(
            "%s must be a non-empty exact string without surrounding space or NUL" % where
        )
    return value


def _dimension(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in (1, 2, 3):
        raise ValueError("%s must be exactly 1, 2, or 3" % where)
    return value


def _capture(value: Any, *, where: str, active: set[int] | None = None) -> Any:
    """Capture strict identity data, spelling binary64 values by exact bits."""
    if value is None or type(value) in (bool, int, str, bytes):
        canonical_bytes(value)
        return value
    if type(value) is float:
        return MappingProxyType({"binary64": value.hex()})
    if isinstance(value, Mapping):
        active = set() if active is None else active
        marker = id(value)
        if marker in active:
            raise ValueError("%s contains a reference cycle" % where)
        active.add(marker)
        try:
            captured = {}
            for key, item in value.items():
                if type(key) is not str or not key:
                    raise TypeError("%s keys must be non-empty exact strings" % where)
                captured[key] = _capture(item, where="%s.%s" % (where, key), active=active)
            return MappingProxyType(captured)
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        active = set() if active is None else active
        marker = id(value)
        if marker in active:
            raise ValueError("%s contains a reference cycle" % where)
        active.add(marker)
        try:
            return tuple(
                _capture(item, where="%s[%d]" % (where, index), active=active)
                for index, item in enumerate(value)
            )
        finally:
            active.remove(marker)
    raise TypeError("%s contains non-canonical %s" % (where, type(value).__name__))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class InversionWorkspaceBudget:
    """Exact reusable backend allocation required by one inversion provider."""

    bytes: int
    alignment: int = 16

    def __post_init__(self) -> None:
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("inversion workspace bytes must be a non-negative exact integer")
        if (
            type(self.alignment) is not int
            or self.alignment < 1
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("inversion workspace alignment must be a positive power of two")

    def to_data(self) -> dict[str, int]:
        return {"bytes": self.bytes, "alignment": self.alignment}


@dataclass(frozen=True, slots=True)
class VariableInversionProblem:
    """Dimensioned state/input/candidate/failure vocabulary of one inversion problem."""

    dimension: int
    state_identity: str
    provider_inputs_identity: str
    candidate_identity: str
    failure_identity: str
    failure_codes: tuple[tuple[int, str], ...]
    workspace: InversionWorkspaceBudget

    def __post_init__(self) -> None:
        _dimension(self.dimension, where="VariableInversionProblem.dimension")
        for name in (
            "state_identity",
            "provider_inputs_identity",
            "candidate_identity",
            "failure_identity",
        ):
            _text(getattr(self, name), where="VariableInversionProblem.%s" % name)
        if type(self.workspace) is not InversionWorkspaceBudget:
            raise TypeError("VariableInversionProblem.workspace requires InversionWorkspaceBudget")
        if type(self.failure_codes) is not tuple or not self.failure_codes:
            raise TypeError(
                "VariableInversionProblem.failure_codes must be a non-empty exact tuple"
            )
        seen_codes: set[int] = set()
        seen_names: set[str] = set()
        for code, name in self.failure_codes:
            if type(code) is not int or code < 1:
                raise ValueError("inversion failure codes must be positive exact integers")
            _text(name, where="inversion failure name")
            if code in seen_codes or name in seen_names:
                raise ValueError("inversion failure code/name collision")
            seen_codes.add(code)
            seen_names.add(name)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "dimension": self.dimension,
            "state_identity": self.state_identity,
            "provider_inputs_identity": self.provider_inputs_identity,
            "candidate_identity": self.candidate_identity,
            "failure_identity": self.failure_identity,
            "failure_codes": [{"code": code, "name": name} for code, name in self.failure_codes],
            "workspace": self.workspace.to_data(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_data())

    @property
    def identity(self) -> str:
        return make_identity("variable-inversion-problem", self.to_data()).token


@dataclass(frozen=True, slots=True)
class PreparedInversionProvider:
    """Authenticated provider declaration consumed by native/codegen preparation."""

    provider_id: str
    interface_version: int
    problem: VariableInversionProblem
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.provider_id, where="inversion provider_id")
        if type(self.interface_version) is not int or self.interface_version < 1:
            raise ValueError("inversion provider interface_version must be positive")
        if type(self.problem) is not VariableInversionProblem:
            raise TypeError("inversion provider requires VariableInversionProblem")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("inversion provider parameters must be a mapping")
        object.__setattr__(
            self,
            "parameters",
            _capture(self.parameters, where="inversion provider parameters"),
        )

    @classmethod
    def author(
        cls,
        provider_id: str,
        problem: VariableInversionProblem,
        *,
        interface_version: int = 1,
        **parameters: Any,
    ) -> PreparedInversionProvider:
        """Convenience authoring with the same canonical contract as direct construction."""
        return cls(provider_id, interface_version, problem, parameters)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "interface_version": self.interface_version,
            "problem": self.problem.to_data(),
            "parameters": _plain(self.parameters),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_data())

    @property
    def identity(self) -> str:
        return make_identity("prepared-inversion-provider", self.to_data()).token


class InversionProviderCatalog:
    """Append-only collision guard used while composing a resolved model contract."""

    __slots__ = ("_by_id", "_by_identity")

    def __init__(self) -> None:
        self._by_id: dict[str, PreparedInversionProvider] = {}
        self._by_identity: dict[str, PreparedInversionProvider] = {}

    def register(self, provider: PreparedInversionProvider) -> PreparedInversionProvider:
        if type(provider) is not PreparedInversionProvider:
            raise TypeError("inversion catalog requires an exact PreparedInversionProvider")
        if provider.provider_id in self._by_id:
            raise ValueError("inversion provider id collision: %r" % provider.provider_id)
        if provider.identity in self._by_identity:
            raise ValueError("inversion provider semantic identity collision")
        self._by_id[provider.provider_id] = provider
        self._by_identity[provider.identity] = provider
        return provider

    def providers(self) -> tuple[PreparedInversionProvider, ...]:
        return tuple(self._by_id[name] for name in sorted(self._by_id))


__all__ = [
    "InversionProviderCatalog",
    "InversionWorkspaceBudget",
    "PreparedInversionProvider",
    "VariableInversionProblem",
]
