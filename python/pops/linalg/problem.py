"""Typed algebraic problems consumed by :meth:`pops.time.Program.solve`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _freeze_canonical(value: Any) -> Any:
    """Make strict JSON-shaped identity data recursively immutable and hashable."""
    if isinstance(value, dict):
        return tuple((key, _freeze_canonical(value[key])) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_canonical(item) for item in value)
    to_data = getattr(value, "to_data", None)
    if getattr(value, "__pops_ir_immutable__", False) is True and callable(to_data):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _freeze_canonical(to_data()),
        )
    return value


@dataclass(frozen=True, slots=True)
class LinearOperatorProperties:
    """Explicit mathematical certificate consumed by property-dependent solvers.

    These facts are never inferred from a stencil name or a preconditioner.  ``CG`` requires the
    complete SPD certificate; general nonsymmetric methods accept :meth:`general`.
    """

    symmetric: bool = False
    positive_definite: bool = False
    positive_definite_on_nullspace_complement: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.symmetric) is not bool
            or type(self.positive_definite) is not bool
            or type(self.positive_definite_on_nullspace_complement) is not bool
        ):
            raise TypeError("linear operator properties must be exact booleans")
        if self.positive_definite and self.positive_definite_on_nullspace_complement:
            raise ValueError(
                "global positive_definite and positive_definite_on_nullspace_complement "
                "are mutually exclusive certificates"
            )
        if (
            self.positive_definite or self.positive_definite_on_nullspace_complement
        ) and not self.symmetric:
            raise ValueError(
                "positive_definite certificates require symmetric=True"
            )

    @classmethod
    def general(cls) -> LinearOperatorProperties:
        return cls()

    @classmethod
    def symmetric_operator(cls) -> LinearOperatorProperties:
        """Certify symmetry without certifying positive definiteness."""
        return cls(symmetric=True)

    @classmethod
    def symmetric_positive_definite(cls) -> LinearOperatorProperties:
        return cls(symmetric=True, positive_definite=True)

    @classmethod
    def symmetric_positive_definite_on_nullspace_complement(
        cls,
    ) -> LinearOperatorProperties:
        """Certify SPD only after removing an explicitly declared nullspace."""
        return cls(
            symmetric=True,
            positive_definite_on_nullspace_complement=True,
        )

    @property
    def certifies_spd(self) -> bool:
        return self.symmetric and self.positive_definite

    def certifies_cg(self, *, declared_nullspace: bool) -> bool:
        if declared_nullspace:
            return (
                self.symmetric
                and self.positive_definite_on_nullspace_complement
            )
        return self.certifies_spd

    def canonical_data(self) -> dict[str, bool]:
        return {
            "symmetric": self.symmetric,
            "positive_definite": self.positive_definite,
            "positive_definite_on_nullspace_complement": (
                self.positive_definite_on_nullspace_complement
            ),
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
    nullspace: Any = field(kw_only=True, compare=False, hash=False)
    gauge: Any = field(default=None, kw_only=True, compare=False, hash=False)
    _prepared_nullspace_provider: Any = field(
        init=False, repr=False, compare=False, hash=False
    )
    _prepared_nullspace_contracts: Any = field(
        init=False, repr=False, compare=False, hash=False
    )
    _nullspace_identity: Any = field(init=False, repr=False)
    _gauge_identity: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Keep the abstract linalg layer dependency-free at import time. Provider resolution and
        # descriptor/gauge validation are needed only when a concrete problem is authored.
        from pops.fields._prepared_nullspace_registry import (
            prepared_nullspace_binding_from_descriptor,
        )

        if not isinstance(self.properties, LinearOperatorProperties):
            raise TypeError(
                "LinearProblem.properties must be pops.linalg.LinearOperatorProperties")
        provider, options = prepared_nullspace_binding_from_descriptor(self.nullspace)
        contracts = provider.prepare(
            options=options,
            gauge=self.gauge,
            operator_properties=self.properties.canonical_data(),
            where="LinearProblem nullspace provider %r" % provider.provider_id,
        )
        nullspace_contract = provider.enveloped_contract(contracts)
        _, gauge_contract = contracts.detached()
        object.__setattr__(self, "_prepared_nullspace_provider", provider)
        object.__setattr__(self, "_prepared_nullspace_contracts", contracts)
        object.__setattr__(
            self,
            "_nullspace_identity",
            _freeze_canonical(nullspace_contract),
        )
        object.__setattr__(
            self,
            "_gauge_identity",
            _freeze_canonical(gauge_contract),
        )

    def canonical_nullspace_provider(self) -> dict[str, Any]:
        """Return the exact registered compiler authority selected at construction."""
        return self._prepared_nullspace_provider.authority()

    def canonical_nullspace_contract(self) -> dict[str, Any]:
        """Return a detached canonical view of the authored nullspace assertion."""
        return self._prepared_nullspace_provider.enveloped_contract(
            self._prepared_nullspace_contracts
        )

    def canonical_gauge_contract(self) -> dict[str, Any]:
        """Return the construction-time gauge snapshot used by every lowering path."""
        _, gauge = self._prepared_nullspace_contracts.detached()
        return gauge

    def build_matrix_free_linear(self, *, program: Any, prepared_solver: Any,
                                 name: Any = None) -> Any:
        return program._solve_linear(
            operator=self.operator, rhs=self.rhs,
            initial_guess=self.initial_guess, prepared=prepared_solver,
            name=name, at=self.at, scope=self.scope, properties=self.properties,
            nullspace_provider=self.canonical_nullspace_provider(),
            nullspace_contract=self.canonical_nullspace_contract(),
            gauge_contract=self.canonical_gauge_contract())


__all__ = ["LinearOperatorProperties", "LinearProblem"]
