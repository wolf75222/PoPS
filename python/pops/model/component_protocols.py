"""Small structural facets implemented by source and IR components.

Components opt in to facets in their :class:`ComponentManifest`.  The registry
then checks these protocols at its trust boundary; inheritance from a PoPS base
class is deliberately not required.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
}


__all__ = [
    "Requirement", "Lowering", "Stencil", "Stability", "Provider", "Effects",
    "Restart", "Report", "FallibleEvaluation", "FACET_PROTOCOLS",
]
