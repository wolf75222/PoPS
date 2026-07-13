"""Typed physical output maps produced by a :class:`FieldOperator`."""

from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.ir.expr import Expr
from pops.model import Handle

from ._identity import strict_field_data
from ._references import reference_label, resolve_handle


class _Output(Descriptor):
    """Base contract for one named physical output of a field operator."""

    category = "field_output"
    recipe = "field"

    def __init__(self, name: Any, source: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("field output name must be a non-empty string")
        if source is not None and not isinstance(source, Handle):
            raise TypeError("field output source must be a declaration Handle")
        self._name = name
        self.source = source

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "recipe": self.recipe,
            "source": (
                reference_label(self.source, where="field output source")
                if self.source is not None
                else None
            ),
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self._name,
            "recipe": self.recipe,
            "source": (self.source.canonical_identity() if self.source is not None else None),
        }

    def requirements(self) -> RequirementSet:
        return RequirementSet(
            {}
            if self.source is None
            else {"field": reference_label(self.source, where="field output source")}
        )

    def resolve_references(self, resolver: Any) -> _Output:
        from copy import copy

        resolved = copy(self)
        if self.source is not None:
            resolved.source = resolve_handle(self.source, resolver, where="field output source")
        return resolved

    def declaration_references(self) -> tuple[Handle, ...]:
        return () if self.source is None else (self.source,)


class FieldOutput(_Output):
    """Expose the solved unknown itself under ``name``."""


class GradientOutput(_Output):
    """Expose the physical gradient of ``source`` with an explicit sign."""

    recipe = "gradient"

    def __init__(self, name: Any, source: Any, *, sign: int = -1) -> None:
        if sign not in {-1, 1}:
            raise ValueError("GradientOutput sign must be exactly -1 or 1")
        super().__init__(name, source=source)
        self.sign = sign

    def options(self) -> dict[str, Any]:
        result = super().options()
        result["sign"] = self.sign
        return result

    def to_data(self) -> dict[str, Any]:
        result = super().to_data()
        result["sign"] = self.sign
        return result

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({"vector": True, "derivative_order": 1})


class DerivedField(_Output):
    """Expose a derived field defined by a symbolic expression, never a recipe string."""

    recipe = "expression"

    def __init__(self, name: Any, expression: Any) -> None:
        if not isinstance(expression, Expr):
            raise TypeError("DerivedField expression must be a pops Expr")
        super().__init__(name)
        self.expression = expression

    def options(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "recipe": self.recipe,
            "expression": strict_field_data(self.expression),
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self._name,
            "recipe": self.recipe,
            "expression": strict_field_data(self.expression),
        }

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({"derived": True})

    def declaration_references(self) -> tuple[Handle, ...]:
        return tuple(self.expression.declaration_references())

    def resolve_references(self, resolver: Any) -> DerivedField:
        return DerivedField(self.name, self.expression.resolve_references(resolver))


__all__ = ["DerivedField", "FieldOutput", "GradientOutput"]
