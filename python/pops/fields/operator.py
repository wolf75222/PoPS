"""Model-owned physical field operators, with no numerical configuration."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import math
from typing import Any

from pops.descriptors import Availability, Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.identity import Identity
from pops.math import Equation
from pops.model import Handle

from ._identity import field_identity, strict_field_data
from ._references import collect_references, reference_label, resolve_handle, resolve_value
from .outputs import _Output


class FieldProviderMeasure(Descriptor):
    """Small extension interface describing how one RHS contribution is measured."""

    category = "field_provider_measure"

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}

    def lower_native_provider(self) -> dict[str, Any]:
        raise NotImplementedError(
            "%s has no native field-provider lowering adapter" % type(self).__name__)


class SourceDensity(FieldProviderMeasure):
    """A pointwise source density consumed directly by the native field residual."""

    def options(self) -> dict[str, Any]:
        return {"measure": "source_density"}

    def lower_native_provider(self) -> dict[str, Any]:
        return {"native_measure": "source_density"}


@dataclass(frozen=True, slots=True)
class FieldProviderContribution:
    """One ordered, owner-qualified contribution to a field RHS."""

    provider: Handle
    coefficient: float = 1.0
    measure: FieldProviderMeasure = SourceDensity()

    def __post_init__(self) -> None:
        if not isinstance(self.provider, Handle) or self.provider.kind != "field_operator":
            raise TypeError(
                "FieldProviderContribution provider must be a field_operator Handle")
        if isinstance(self.coefficient, bool) or not isinstance(self.coefficient, (int, float)) \
                or not math.isfinite(float(self.coefficient)):
            raise TypeError("FieldProviderContribution coefficient must be a finite scalar")
        if not isinstance(self.measure, FieldProviderMeasure):
            raise TypeError(
                "FieldProviderContribution measure must implement FieldProviderMeasure")
        object.__setattr__(self, "coefficient", float(self.coefficient))

    def to_data(self) -> dict[str, Any]:
        return {
            "provider": self.provider.canonical_identity(),
            "coefficient": self.coefficient,
            "measure": self.measure.to_data(),
        }


class FieldProviderPack:
    """Immutable ordered composition of RHS providers.

    Ordering, coefficients and measures are part of the physical identity. An empty pack is
    rejected; an intentionally zero source must be represented by an explicit provider whose body
    is zero, rather than by absence of behavior.
    """

    __slots__ = ("_contributions",)
    __pops_ir_immutable__ = True

    def __init__(self, contributions: Any) -> None:
        values = tuple(contributions)
        if not values:
            raise ValueError("FieldProviderPack requires at least one typed contribution")
        normalized = tuple(
            value if isinstance(value, FieldProviderContribution)
            else FieldProviderContribution(value)
            for value in values
        )
        object.__setattr__(self, "_contributions", normalized)

    @classmethod
    def one(cls, provider: Handle) -> FieldProviderPack:
        return cls((FieldProviderContribution(provider),))

    def __iter__(self) -> Any:
        return iter(self._contributions)

    def __len__(self) -> int:
        return len(self._contributions)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "contributions": [item.to_data() for item in self._contributions],
        }


class FieldOperator(Descriptor):
    """Physical equation mapping one declared unknown to named field outputs.

    The operator deliberately owns only physics: ``unknown``, ``equation`` and ``outputs``.
    Methods, boundary conditions, solvers and hierarchy choices belong to
    :class:`pops.fields.FieldDiscretization`.
    """

    category = "field_operator"

    def __init__(
        self,
        name: Any,
        *,
        unknown: Any,
        equation: Any,
        providers: Any,
        outputs: Any = (),
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("FieldOperator name must be a non-empty string")
        if not isinstance(unknown, Handle):
            raise TypeError("FieldOperator unknown must be a declaration Handle")
        if isinstance(providers, Handle):
            providers = FieldProviderPack.one(providers)
        if not isinstance(providers, FieldProviderPack):
            raise TypeError("FieldOperator providers must be a FieldProviderPack")
        if isinstance(equation, bool):
            raise TypeError(
                "FieldOperator equation is a Python bool; build a symbolic pops Equation"
            )
        if not isinstance(equation, Equation):
            raise TypeError("FieldOperator equation must be a pops Equation")
        output_tuple = tuple(outputs)
        if any(not isinstance(output, _Output) for output in output_tuple):
            raise TypeError("FieldOperator outputs must contain typed field output descriptors")
        names = [output.name for output in output_tuple]
        if len(names) != len(set(names)):
            raise ValueError("FieldOperator output names must be unique")
        self._name = name
        self.unknown = unknown
        self.equation = equation
        self.providers = providers
        self.outputs = output_tuple

    @property
    def name(self) -> str:
        return self._name

    @property
    def identity(self) -> Identity:
        for reference in self.declaration_references():
            reference.canonical_identity()
        return field_identity("field-operator", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self._name,
            "unknown": self.unknown.canonical_identity(),
            "providers": self.providers.to_data(),
            "equation": strict_field_data(self.equation),
            "outputs": [output.to_data() for output in self.outputs],
        }

    def semantic_data(self) -> dict[str, Any]:
        """Exact physical identity; the equation is never reduced to display options."""
        return self.to_data()

    def artifact_data(self) -> dict[str, Any]:
        return self.to_data()

    def options(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "unknown": reference_label(self.unknown, where="FieldOperator unknown"),
            "providers": [
                {
                    "provider": reference_label(
                        item.provider, where="FieldOperator provider contribution"),
                    "coefficient": item.coefficient,
                    "measure": item.measure.options(),
                }
                for item in self.providers
            ],
            "outputs": [output.name for output in self.outputs],
        }

    def requirements(self) -> RequirementSet:
        return RequirementSet(
            {
                "unknown": reference_label(self.unknown, where="FieldOperator unknown"),
                "declaration_references": [
                    reference_label(reference, where="FieldOperator reference")
                    for reference in self.declaration_references()
                ],
            }
        )

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({"elliptic": True, "model_owned": True})

    def available(self, context: Any = None) -> Availability:
        del context
        return Availability.yes("physical field operator is structurally complete")

    def validate(self, context: Any = None) -> bool:
        references = self.declaration_references()
        authenticate = getattr(context, "authenticate", None)
        if callable(authenticate):
            for reference in references:
                authenticated = authenticate(reference)
                if not isinstance(authenticated, Handle):
                    raise TypeError("FieldOperator authenticator must return Handle values")
        if all(reference.is_resolved for reference in references):
            _ = self.identity
        return True

    def declaration_references(self) -> tuple[Handle, ...]:
        return collect_references(
            (self.unknown, tuple(item.provider for item in self.providers),
             self.equation, self.outputs))

    def resolve_references(self, resolver: Any) -> FieldOperator:
        resolved = copy(self)
        resolved.unknown = resolve_handle(self.unknown, resolver, where="FieldOperator unknown")
        resolved.providers = FieldProviderPack(
            FieldProviderContribution(
                resolve_handle(
                    item.provider, resolver, where="FieldOperator provider contribution"),
                item.coefficient,
                item.measure,
            )
            for item in self.providers
        )
        # Blackboard formula Vars are coordinates of the authenticated compiled provider, not
        # declaration identities. Resolve every Handle leaf while preserving those coordinate Vars;
        # the generic Expr API remains strict and still rejects a free-name Var.
        from pops._ir.expr_references import resolve_expr_references
        resolved.equation = resolve_expr_references(
            self.equation, resolver, {}, allow_formula_vars=True)
        resolved.outputs = tuple(
            resolve_value(self.outputs, resolver, where="FieldOperator outputs")
        )
        return resolved

    def inspect(self) -> dict[str, Any]:
        info = super().inspect()
        info["physics"] = {
            "equation": strict_field_data(self.equation),
            "outputs": [output.options() for output in self.outputs],
        }
        info["identity"] = (
            self.identity.token
            if all(reference.is_resolved for reference in self.declaration_references())
            else None
        )
        return info


__all__ = [
    "FieldOperator",
    "FieldProviderContribution",
    "FieldProviderMeasure",
    "FieldProviderPack",
    "SourceDensity",
]
