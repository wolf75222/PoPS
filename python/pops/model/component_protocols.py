"""Small structural facets implemented by source and IR components.

Components opt in to facets in their :class:`ComponentManifest`.  The registry
then checks these protocols at its trust boundary; inheritance from a PoPS base
class is deliberately not required.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._generated_component_schema import COMPONENT_INTERFACE_SPECS


@runtime_checkable
class Requirement(Protocol):
    def requirements(self) -> Any: ...


@runtime_checkable
class Lowering(Protocol):
    def lower(self, context: Any) -> Any: ...


@runtime_checkable
class Stencil(Protocol):
    def stencil(self) -> Any: ...


@runtime_checkable
class Stability(Protocol):
    def stability(self) -> Any: ...


@runtime_checkable
class Provider(Protocol):
    def providers(self) -> Any: ...


@runtime_checkable
class Effects(Protocol):
    def effects(self) -> Any: ...


@runtime_checkable
class Restart(Protocol):
    def restart(self) -> Any: ...


@runtime_checkable
class Report(Protocol):
    def report(self) -> Any: ...


@runtime_checkable
class FallibleEvaluation(Protocol):
    def evaluate(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class Format(Protocol):
    def format(self, value: Any) -> Any: ...


FACET_PROTOCOLS = {
    "requirement": Requirement,
    "lowering": Lowering,
    "stencil": Stencil,
    "stability": Stability,
    "provider": Provider,
    "effects": Effects,
    "restart": Restart,
    "report": Report,
    "fallible_evaluation": FallibleEvaluation,
    "format": Format,
}

_generated_names = {row["name"] for row in COMPONENT_INTERFACE_SPECS}
if set(FACET_PROTOCOLS) != _generated_names:
    raise RuntimeError(
        "component protocol declarations drifted from generated interface vocabulary: "
        "missing=%r unknown=%r"
        % (sorted(_generated_names - set(FACET_PROTOCOLS)),
           sorted(set(FACET_PROTOCOLS) - _generated_names)))


__all__ = [
    "Requirement", "Lowering", "Stencil", "Stability", "Provider", "Effects",
    "Restart", "Report", "FallibleEvaluation", "Format", "FACET_PROTOCOLS",
]
