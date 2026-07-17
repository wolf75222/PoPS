"""Execution scopes for linear solves.

A scope describes where one mathematical solve lives.  ``Level`` is the normal
single-grid solve.  ``Hierarchy`` gathers every AMR level, solves once on the
composite hierarchy, then publishes one solution per level.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolveScope:
    """Immutable extension interface implemented by solve scopes."""

    scope_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("SolveScope.scope_id must be a non-empty string")


class Level(SolveScope):
    """Solve independently on the current mesh level (the default)."""

    def __init__(self) -> None:
        super().__init__("level")


class Hierarchy(SolveScope):
    """Solve once over the complete AMR hierarchy."""

    def __init__(self) -> None:
        super().__init__("hierarchy")


def solve_scope_id(scope: object | None) -> str:
    if scope is None:
        return "level"
    if not isinstance(scope, SolveScope):
        raise TypeError("solve_linear: scope must implement SolveScope; got %r" % (scope,))
    return scope.scope_id


__all__ = ["SolveScope", "Level", "Hierarchy"]
