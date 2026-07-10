"""pops.fields.coefficients -- typed elliptic-operator coefficients (Spec 5 sec.5.5).

A field problem's operator can carry spatially varying coefficients: a scalar coefficient
in front of the principal (e.g. diffusion / permittivity) term, and a zeroth-order
reaction coefficient (the ``k`` of a screened Poisson ``-div(a grad phi) + k phi``). Each
names an aux field that supplies the coefficient values; the runtime reads them.

Inert descriptors; they compute nothing.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import RequirementSet


class _ImmutableCoefficient(Descriptor):
    """Strict value descriptor safe to retain inside an immutable symbolic graph."""

    __pops_ir_immutable__ = True
    category = "coefficient"
    _role = ""

    def __init__(self, name: Any) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("coefficient name must be a non-empty string")
        object.__setattr__(self, "_name", name)

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict:
        return {"name": self._name, "role": self._role}

    def requirements(self) -> Any:
        return RequirementSet({"aux_field": self._name})

    def freeze(self) -> Any:
        """Already immutable; participate idempotently in Problem freeze."""
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)


class ScalarCoefficient(_ImmutableCoefficient):
    """A scalar coefficient field named :paramref:`name` (e.g. permittivity / diffusivity)."""

    _role = "scalar"


class ReactionCoefficient(_ImmutableCoefficient):
    """A zeroth-order reaction coefficient field named :paramref:`name` (screened term ``k phi``)."""

    _role = "reaction"


__all__ = ["ScalarCoefficient", "ReactionCoefficient"]
