"""Small immutable problem protocols consumed by :meth:`Program.solve`."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pops._ir.literals import exact_numeric_scalar


def _frozen_product(value: Any, *, where: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(dict(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise TypeError("%s must be a typed mapping or non-empty sequence" % where)


@dataclass(frozen=True, slots=True)
class CoupledImplicitEuler:
    """The explicit equation ``U = U0 + coefficient * dt * Q(U)``."""

    operator: Any
    inputs: Any
    coefficient: Any = 1
    at: Any = None

    def __post_init__(self) -> None:
        from pops.model import OperatorHandle

        if not isinstance(self.operator, OperatorHandle):
            raise TypeError("CoupledImplicitEuler operator must be a typed OperatorHandle")
        inputs = _frozen_product(self.inputs, where="CoupledImplicitEuler inputs")
        if not inputs:
            raise ValueError("CoupledImplicitEuler inputs must be non-empty")
        object.__setattr__(self, "inputs", inputs)
        coefficient = exact_numeric_scalar(
            self.coefficient, where="CoupledImplicitEuler coefficient")
        object.__setattr__(self, "coefficient", coefficient)
        at = self.at
        if isinstance(at, Mapping):
            at = MappingProxyType(dict(at))
        elif isinstance(at, Sequence) and not isinstance(at, (str, bytes)):
            at = tuple(at)
        object.__setattr__(self, "at", at)

    def build_with(self, *, program: Any, prepared_solver: Any, name: Any = None) -> Any:
        return program._solve_coupled_implicit(
            self.operator, self.inputs, prepared=prepared_solver, name=name, at=self.at,
            coefficient=self.coefficient)


@dataclass(frozen=True, slots=True)
class LocalResidual:
    """A cell-local residual callback and its exact initial temporal state."""

    residual: Any
    initial: Any

    def __post_init__(self) -> None:
        if not callable(self.residual):
            raise TypeError("LocalResidual residual must be an IR-building callable")

    def build_with(self, *, program: Any, prepared_solver: Any, name: Any = None) -> Any:
        return program._solve_local_nonlinear(
            residual=self.residual, initial_guess=self.initial,
            prepared=prepared_solver, name=name)


@dataclass(frozen=True, slots=True)
class LocalLinear:
    """One cell-local linear system with an optional exact field context."""

    operator: Any
    rhs: Any
    fields: Any = None

    def build_local_linear(self, *, program: Any, prepared_solver: Any,
                           name: Any = None) -> Any:
        return program._solve_local_linear(
            operator=self.operator, rhs=self.rhs, fields=self.fields,
            prepared=prepared_solver, name=name)


__all__ = ["CoupledImplicitEuler", "LocalLinear", "LocalResidual"]
