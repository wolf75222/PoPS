"""Typed algebraic problems consumed by :meth:`pops.time.Program.solve`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def build_matrix_free_linear(self, *, program: Any, prepared_solver: Any,
                                 name: Any = None) -> Any:
        return program._solve_linear(
            operator=self.operator, rhs=self.rhs,
            initial_guess=self.initial_guess, prepared=prepared_solver,
            name=name, at=self.at, scope=self.scope)


__all__ = ["LinearProblem"]
