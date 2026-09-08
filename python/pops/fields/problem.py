"""Joint physical field equations and independent storage declarations.

A FieldProblem is the equation authority. Numerical methods and solver settings are
bound separately; a species contributing to its load does not own the field storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.descriptors import Descriptor
from pops.identity import Identity
from pops.math import Equation, elliptic_terms
from pops.model import Handle

from ._identity import field_identity, strict_field_data
from ._references import collect_references, resolve_handle, resolve_value
from .bcs import BoundaryCondition


class FieldProblemError(ValueError):
    """A malformed physical field problem, reported before numerical lowering."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        self.report = {"phase": "field_problem", "code": code, "context": context}
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FieldBoundary:
    """One physical boundary relation of one exact field unknown."""

    unknown: Handle
    relation: BoundaryCondition

    def __post_init__(self) -> None:
        if not isinstance(self.unknown, Handle) or self.unknown.kind != "field":
            raise TypeError("FieldBoundary unknown must be a typed field Handle")
        if not isinstance(self.relation, BoundaryCondition):
            raise TypeError("FieldBoundary relation must be a typed BoundaryCondition")

    def freeze(self) -> FieldBoundary:
        # The snapshot boundary has detached and frozen the nested relation first.
        return self

    def declaration_references(self) -> tuple[Handle, ...]:
        return collect_references((self.unknown, self.relation))

    def resolve_references(self, resolver: Any) -> FieldBoundary:
        return FieldBoundary(
            resolve_handle(self.unknown, resolver, where="FieldBoundary unknown"),
            resolve_value(self.relation, resolver, where="FieldBoundary relation"),
        )

    def to_data(self) -> dict[str, Any]:
        return {"unknown": self.unknown.canonical_identity(),
                "relation": strict_field_data(self.relation)}


@dataclass(frozen=True, slots=True)
class SharedMeanGauge:
    """One representative condition on the shared constant mode of a tuple.

    The single condition is mean(sum(phi_i)) = value. It does not impose separate
    zero means, and never changes an incompatible physical right-hand side.
    """

    unknowns: tuple[Handle, ...]
    value: Any = 0

    def __post_init__(self) -> None:
        from pops.identity.scalar import scalar_literal
        if not isinstance(self.unknowns, tuple) or not self.unknowns or any(
                not isinstance(row, Handle) or row.kind != "field" for row in self.unknowns):
            raise TypeError("SharedMeanGauge requires an exact tuple of field Handles")
        if len(set(self.unknowns)) != len(self.unknowns):
            raise FieldProblemError("field.gauge.duplicate_unknown", "shared gauge repeats an unknown")
        scalar_literal(self.value)

    def freeze(self) -> SharedMeanGauge:
        return self

    def declaration_references(self) -> tuple[Handle, ...]:
        return self.unknowns

    def resolve_references(self, resolver: Any) -> SharedMeanGauge:
        return SharedMeanGauge(tuple(resolve_handle(row, resolver, where="shared gauge")
                                     for row in self.unknowns), self.value)

    def to_data(self) -> dict[str, Any]:
        from pops.identity.scalar import scalar_literal
        return {"kind": "shared_constant_mean", "unknowns": [row.canonical_identity()
                 for row in self.unknowns], "value": scalar_literal(self.value).to_data()}


@dataclass(frozen=True, slots=True)
class FieldStorageBinding:
    """A field-owned storage binding, independent of the load-producing state blocks."""

    unknowns: tuple[Handle, ...]
    layout: Handle

    def __post_init__(self) -> None:
        if not isinstance(self.unknowns, tuple) or not self.unknowns or any(
                not isinstance(row, Handle) or row.kind != "field" for row in self.unknowns):
            raise TypeError("FieldStorageBinding requires an exact tuple of field Handles")
        if len(set(self.unknowns)) != len(self.unknowns):
            raise FieldProblemError("field.storage.duplicate_unknown", "field storage repeats an unknown")
        if not isinstance(self.layout, Handle) or self.layout.kind != "layout":
            raise TypeError("FieldStorageBinding layout must be a typed LayoutHandle")

    def freeze(self) -> FieldStorageBinding:
        return self

    @property
    def identity(self) -> Identity:
        return field_identity("field-storage", self.to_data())

    def declaration_references(self) -> tuple[Handle, ...]:
        return (*self.unknowns, self.layout)

    def resolve_references(self, resolver: Any) -> FieldStorageBinding:
        return FieldStorageBinding(
            tuple(resolve_handle(row, resolver, where="field storage unknown") for row in self.unknowns),
            resolve_handle(self.layout, resolver, where="field storage layout"),
        )

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "unknowns": [row.canonical_identity()
                for row in self.unknowns], "layout": self.layout.canonical_identity()}


class FieldProblem(Descriptor):
    """One scalar or joint equation authority, without numerical configuration.

    The old FieldOperator is the single-unknown compatibility adapter of this
    representation. New equations infer their inputs from exact expression leaves.
    """

    category = "field_problem"

    def __init__(self, name: str, *, unknowns: tuple[Handle, ...],
                 equations: tuple[Equation, ...], boundaries: tuple[FieldBoundary, ...] = (),
                 gauge: SharedMeanGauge | None = None, branch: Any = None,
                 outputs: tuple[Any, ...] = ()) -> None:
        if type(name) is not str or not name:
            raise TypeError("FieldProblem name must be a nonempty string")
        if not isinstance(unknowns, tuple) or not unknowns or any(
                not isinstance(row, Handle) or row.kind != "field" for row in unknowns):
            raise TypeError("FieldProblem unknowns must be a nonempty tuple of field Handles")
        if len(set(unknowns)) != len(unknowns):
            raise FieldProblemError("field.unknown.duplicate", "field problem repeats an unknown")
        if not isinstance(equations, tuple) or len(equations) != len(unknowns) or any(
                not isinstance(row, Equation) for row in equations):
            raise FieldProblemError("field.equation.arity", "one symbolic equation per field unknown is required")
        if not isinstance(boundaries, tuple) or any(not isinstance(row, FieldBoundary)
                                                  for row in boundaries):
            raise TypeError("FieldProblem boundaries must be a tuple of FieldBoundary relations")
        if any(row.unknown not in unknowns for row in boundaries):
            raise FieldProblemError("field.boundary.foreign_unknown", "physical boundary belongs to a foreign field")
        if gauge is not None and (not isinstance(gauge, SharedMeanGauge)
                                  or gauge.unknowns != unknowns):
            raise FieldProblemError("field.gauge.joint_required", "a joint field problem requires one shared gauge over its exact unknown tuple")
        self._name = name
        self._unknowns = unknowns
        self._equations = equations
        self.boundaries = boundaries
        self.gauge = gauge
        self.branch = branch
        self.outputs = tuple(outputs)
        self.validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def unknowns(self) -> tuple[Handle, ...]:
        return self._unknowns

    @property
    def equations(self) -> tuple[Equation, ...]:
        return self._equations

    @property
    def identity(self) -> Identity:
        return field_identity("field-problem", self.to_data())

    def declaration_references(self) -> tuple[Handle, ...]:
        return collect_references((self.unknowns, self.equations, self.boundaries,
                                   self.gauge, self.branch, self.outputs))

    def dependencies(self) -> tuple[Handle, ...]:
        return tuple(row for row in collect_references(self.equations) if row not in self.unknowns)

    def validate(self, context: Any = None) -> bool:
        from .operator import _field_targets_unknown
        for index, equation in enumerate(self.equations):
            terms = elliptic_terms(equation.lhs)
            if not terms or not any(_field_targets_unknown(term.field, self.unknowns[index])
                                    for term in terms):
                raise FieldProblemError("field.equation.missing_unknown", "equation does not contain its declared unknown", equation=index)
            if any(not any(_field_targets_unknown(term.field, unknown) for unknown in self.unknowns)
                   for term in terms):
                raise FieldProblemError("field.equation.foreign_unknown", "equation contains a foreign field unknown", equation=index)
        authenticate = getattr(context, "authenticate", None)
        if callable(authenticate):
            for reference in self.declaration_references():
                authenticate(reference)
        return True

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "unknowns": [row.canonical_identity() for row in self.unknowns],
                "equations": [strict_field_data(row) for row in self.equations],
                "boundaries": [row.to_data() for row in self.boundaries],
                "gauge": None if self.gauge is None else self.gauge.to_data(),
                "branch": strict_field_data(self.branch),
                "outputs": [strict_field_data(row) for row in self.outputs]}

    def options(self) -> dict[str, Any]:
        return {"name": self.name, "unknown_count": len(self.unknowns),
                "equation_count": len(self.equations), "boundary_count": len(self.boundaries)}

    def resolve_references(self, resolver: Any) -> FieldProblem:
        from pops._ir.expr_references import resolve_expr_references
        return FieldProblem(self.name,
            unknowns=tuple(resolve_handle(row, resolver, where="FieldProblem unknown") for row in self.unknowns),
            equations=tuple(resolve_expr_references(row, resolver, {}, allow_formula_vars=False)
                            for row in self.equations),
            boundaries=tuple(resolve_value(self.boundaries, resolver, where="FieldProblem boundaries")),
            gauge=resolve_value(self.gauge, resolver, where="FieldProblem gauge"),
            branch=resolve_value(self.branch, resolver, where="FieldProblem branch"),
            outputs=tuple(resolve_value(self.outputs, resolver, where="FieldProblem outputs")))

    semantic_data = to_data
    artifact_data = to_data
