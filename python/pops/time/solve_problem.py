"""Small immutable problem protocols consumed by :meth:`Program.solve`."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pops.identity import make_identity
from pops.ir.literals import exact_numeric_scalar, scalar_data


def _frozen_product(value: Any, *, where: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(dict(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise TypeError("%s must be a typed mapping or non-empty sequence" % where)


def _problem_identity(kind: str, details: dict[str, Any]) -> Any:
    return make_identity("program-solve-problem", {"schema_version": 1, "kind": kind, **details})


@dataclass(frozen=True, slots=True)
class CoupledImplicitEuler:
    """The explicit equation ``U = U0 + coefficient * dt * Q(U)``."""

    operator: Any
    inputs: Any
    coefficient: Any = 1
    at: Any = None
    identity: Any = None

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
        owner = self.operator.owner_path.canonical().to_data()
        identity = _problem_identity("coupled_implicit_euler", {
            "operator": {"owner": owner, "kind": self.operator.kind,
                         "name": self.operator.local_id},
            "coefficient": scalar_data(coefficient),
        })
        if self.identity is not None and self.identity != identity:
            raise ValueError("CoupledImplicitEuler identity is not canonical")
        object.__setattr__(self, "identity", identity)

    def build_with(self, *, program: Any, prepared_solver: Any, name: Any = None) -> Any:
        return program._solve_coupled_implicit(
            self.operator, self.inputs, prepared=prepared_solver, name=name, at=self.at,
            coefficient=self.coefficient, problem_identity=self.identity)


@dataclass(frozen=True, slots=True)
class LocalResidual:
    """A cell-local residual callback and its exact initial temporal state."""

    residual: Any
    initial: Any
    identity: Any = None

    def __post_init__(self) -> None:
        if not callable(self.residual):
            raise TypeError("LocalResidual residual must be an IR-building callable")
        identity = _problem_identity("local_residual", {})
        if self.identity is not None and self.identity != identity:
            raise ValueError("LocalResidual identity is not canonical")
        object.__setattr__(self, "identity", identity)

    def build_with(self, *, program: Any, prepared_solver: Any, name: Any = None) -> Any:
        return program._solve_local_nonlinear(
            residual=self.residual, initial_guess=self.initial,
            prepared=prepared_solver, name=name, problem_identity=self.identity)


__all__ = ["CoupledImplicitEuler", "LocalResidual"]
