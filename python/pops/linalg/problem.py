"""Typed algebraic problems consumed by :meth:`pops.time.Program.solve`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LinearOperatorProperties:
    """Explicit mathematical certificate consumed by property-dependent solvers.

    These facts are never inferred from a stencil name or a preconditioner.  ``CG`` requires the
    complete SPD certificate; general nonsymmetric methods accept :meth:`general`.
    """

    symmetric: bool = False
    positive_definite: bool = False

    def __post_init__(self) -> None:
        if type(self.symmetric) is not bool or type(self.positive_definite) is not bool:
            raise TypeError("linear operator properties must be exact booleans")
        if self.positive_definite and not self.symmetric:
            raise ValueError("positive_definite requires symmetric=True")

    @classmethod
    def general(cls) -> LinearOperatorProperties:
        return cls()

    @classmethod
    def symmetric_positive_definite(cls) -> LinearOperatorProperties:
        return cls(symmetric=True, positive_definite=True)

    @property
    def certifies_spd(self) -> bool:
        return self.symmetric and self.positive_definite

    def canonical_data(self) -> dict[str, bool]:
        return {
            "symmetric": self.symmetric,
            "positive_definite": self.positive_definite,
        }


@dataclass(frozen=True, slots=True)
class LinearProblem:
    """One matrix-free system ``operator(solution) = rhs``.

    The problem names only the algebra and its temporal placement. Algorithm, tolerance,
    iteration budget, restart and preconditioner belong exclusively to the typed solver
    descriptor passed to ``Program.solve``.
    """

    operator: Any
    rhs: Any
    initial_guess: Any = None
    at: Any = None
    scope: Any = None
    properties: LinearOperatorProperties = LinearOperatorProperties()

    def __post_init__(self) -> None:
        if not isinstance(self.properties, LinearOperatorProperties):
            raise TypeError(
                "LinearProblem.properties must be pops.linalg.LinearOperatorProperties")

    def build_matrix_free_linear(self, *, program: Any, prepared_solver: Any,
                                 name: Any = None) -> Any:
        return program._solve_linear(
            operator=self.operator, rhs=self.rhs,
            initial_guess=self.initial_guess, prepared=prepared_solver,
            name=name, at=self.at, scope=self.scope, properties=self.properties)


__all__ = ["LinearOperatorProperties", "LinearProblem"]
