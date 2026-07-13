"""Model-owned physical field operators, with no numerical configuration."""

from __future__ import annotations

from copy import copy
from typing import Any

from pops.descriptors import Availability, Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.identity import Identity
from pops.math import Equation
from pops.model import Handle

from ._identity import field_identity, strict_field_data
from ._references import collect_references, reference_label, resolve_handle, resolve_value
from .outputs import _Output


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
        outputs: Any = (),
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("FieldOperator name must be a non-empty string")
        if not isinstance(unknown, Handle):
            raise TypeError("FieldOperator unknown must be a declaration Handle")
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
            "equation": strict_field_data(self.equation),
            "outputs": [output.to_data() for output in self.outputs],
        }

    def options(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "unknown": reference_label(self.unknown, where="FieldOperator unknown"),
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
        return collect_references((self.unknown, self.equation, self.outputs))

    def resolve_references(self, resolver: Any) -> FieldOperator:
        resolved = copy(self)
        resolved.unknown = resolve_handle(self.unknown, resolver, where="FieldOperator unknown")
        resolved.equation = self.equation.resolve_references(resolver)
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


__all__ = ["FieldOperator"]
