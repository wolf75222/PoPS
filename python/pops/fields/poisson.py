"""Physics-only Poisson-family FieldOperator presets."""

from __future__ import annotations

from typing import Any

from pops.descriptors_report import CapabilitySet
from pops.math import Equation, principal_kinds

from .operator import FieldOperator


_PRINCIPAL_OPERATORS = {"laplacian", "div_coeff_grad"}


class PoissonOperator(FieldOperator):
    """FieldOperator constrained to a Poisson principal operator."""

    category = "poisson_operator"

    def capabilities(self) -> CapabilitySet:
        result = super().capabilities().to_dict()
        result["poisson"] = True
        return CapabilitySet(result)

    def validate(self, context: Any = None) -> bool:
        super().validate(context)
        if isinstance(self.equation, Equation) and not (
            principal_kinds(self.equation.lhs) & _PRINCIPAL_OPERATORS
        ):
            raise ValueError(
                "%s expects a Laplacian or div(coeff*grad) principal operator" % self.name
            )
        return True


class ScreenedPoissonOperator(PoissonOperator):
    category = "screened_poisson_operator"

    def capabilities(self) -> CapabilitySet:
        result = super().capabilities().to_dict()
        result["screened"] = True
        return CapabilitySet(result)

    def validate(self, context: Any = None) -> bool:
        super().validate(context)
        if "reaction" not in principal_kinds(self.equation.lhs):
            raise ValueError("%s expects an explicit reaction term" % self.name)
        return True


class AnisotropicPoissonOperator(PoissonOperator):
    category = "anisotropic_poisson_operator"

    def capabilities(self) -> CapabilitySet:
        result = super().capabilities().to_dict()
        result["anisotropic"] = True
        return CapabilitySet(result)

    def validate(self, context: Any = None) -> bool:
        super().validate(context)
        if "div_coeff_grad" not in principal_kinds(self.equation.lhs):
            raise ValueError("%s expects an explicit div(coeff*grad) operator" % self.name)
        return True


__all__ = [
    "AnisotropicPoissonOperator",
    "PoissonOperator",
    "ScreenedPoissonOperator",
]
