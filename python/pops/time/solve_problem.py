"""Small immutable problem protocols consumed by :meth:`Program.solve`."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pops.identity.scalar import exact_numeric_scalar


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
    derivative: Any = None

    def __post_init__(self) -> None:
        from pops.model import OperatorHandle

        if not isinstance(self.operator, OperatorHandle):
            raise TypeError("CoupledImplicitEuler operator must be a typed OperatorHandle")
        from pops.time.solve_request import DerivativeStrategy
        if self.derivative is not None and type(self.derivative) is not DerivativeStrategy:
            raise TypeError("CoupledImplicitEuler derivative must be a DerivativeStrategy")
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
            coefficient=self.coefficient, derivative=self.derivative)


@dataclass(frozen=True, slots=True)
class LocalResidual:
    """A local equation, its algorithmic seed, and optional frozen equation inputs.

    With captures, the body is called as ``residual(P, iterate, **captures)``.
    Without captures the historical ``residual(P, iterate, initial)`` form remains.
    A capture is an equation argument; changing the seed does not change it.
    """

    residual: Any
    initial: Any
    captures: Any = None

    def __post_init__(self) -> None:
        if not callable(self.residual):
            raise TypeError("LocalResidual residual must be an IR-building callable")
        if self.captures is not None:
            if not isinstance(self.captures, Mapping) or any(
                    not isinstance(key, str) or not key.isidentifier() for key in self.captures):
                raise TypeError("LocalResidual captures require named equation inputs")
            object.__setattr__(self, "captures", MappingProxyType(dict(self.captures)))

    def build_with(self, *, program: Any, prepared_solver: Any, name: Any = None) -> Any:
        return program._solve_local_nonlinear(
            residual=self.residual, initial_guess=self.initial, captures=self.captures,
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
