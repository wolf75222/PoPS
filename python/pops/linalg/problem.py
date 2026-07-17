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
    _nullspace_kind: str = field(init=False, repr=False)
    _gauge_value: Any = field(init=False, repr=False, compare=False, hash=False)
    _gauge_identity: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Keep the abstract linalg layer dependency-free at import time. These exact public
        # descriptor/literal checks are needed only when a concrete problem is authored.
        from pops._ir.literals import scalar_literal
        from pops.fields.gauges import MeanValueGauge
        from pops.fields.nullspace import ConstantNullspace

        if not isinstance(self.properties, LinearOperatorProperties):
            raise TypeError(
                "LinearProblem.properties must be pops.linalg.LinearOperatorProperties")
        if self.nullspace is None:
            if self.gauge is not None:
                raise ValueError(
                    "LinearProblem.gauge must be None when nullspace=None"
                )
            if self.properties.positive_definite_on_nullspace_complement:
                raise ValueError(
                    "positive_definite_on_nullspace_complement requires "
                    "nullspace=ConstantNullspace()"
                )
            object.__setattr__(self, "_nullspace_kind", "none")
            object.__setattr__(self, "_gauge_value", None)
            object.__setattr__(self, "_gauge_identity", None)
            return
        if type(self.nullspace) is not ConstantNullspace:
            raise TypeError(
                "LinearProblem.nullspace must be explicitly None or exactly "
                "pops.fields.ConstantNullspace()"
            )
        if type(self.gauge) is not MeanValueGauge:
            raise TypeError(
                "LinearProblem with ConstantNullspace() requires exactly "
                "pops.fields.MeanValueGauge(value)"
            )
        if not self.properties.symmetric:
            raise ValueError(
                "ConstantNullspace requires an explicit symmetric operator certificate; "
                "a right constant kernel alone does not prove that the mean-zero complement "
                "is invariant"
            )
        if self.properties.positive_definite:
            raise ValueError(
                "a constant-nullspace operator cannot carry a global "
                "positive_definite certificate; certify positive definiteness on the "
                "nullspace complement instead"
            )
        try:
            gauge_value = scalar_literal(self.gauge.value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TypeError(
                "LinearProblem MeanValueGauge value must be one finite scalar literal"
            ) from exc
        object.__setattr__(self, "_nullspace_kind", "constant")
        # ScalarLiteral is a frozen, recursively immutable value.  Lowering consumes this snapshot,
        # never the mutable descriptor object retained for author-facing inspection.
        object.__setattr__(self, "_gauge_value", gauge_value)
        object.__setattr__(
            self, "_gauge_identity", _freeze_canonical(gauge_value.to_data()))

    def canonical_nullspace_contract(self) -> dict[str, Any]:
        """Return a detached canonical view of the authored nullspace assertion."""
        return {"schema_version": 1, "kind": self._nullspace_kind}

    def canonical_gauge_contract(self) -> dict[str, Any]:
        """Return the construction-time gauge snapshot used by every lowering path."""
        if self._gauge_value is None:
            return {"schema_version": 1, "kind": "none"}
        return {
            "schema_version": 1,
            "kind": "mean_value",
            "value": self._gauge_value,
        }

    def build_matrix_free_linear(self, *, program: Any, prepared_solver: Any,
                                 name: Any = None) -> Any:
        return program._solve_linear(
            operator=self.operator, rhs=self.rhs,
            initial_guess=self.initial_guess, prepared=prepared_solver,
            name=name, at=self.at, scope=self.scope, properties=self.properties,
            nullspace_contract=self.canonical_nullspace_contract(),
            gauge_contract=self.canonical_gauge_contract())


__all__ = ["LinearOperatorProperties", "LinearProblem"]
