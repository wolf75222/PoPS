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


class ScalarCoefficient(Descriptor):
    """A scalar coefficient field named :paramref:`name` (e.g. permittivity / diffusivity)."""

    category = "coefficient"

    def __init__(self, name: Any) -> None:
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict:
        return {"name": self._name, "role": "scalar"}

    def requirements(self) -> Any:
        return RequirementSet({"aux_field": self._name})


class ReactionCoefficient(Descriptor):
    """A zeroth-order reaction coefficient field named :paramref:`name` (screened term ``k phi``)."""

    category = "coefficient"

    def __init__(self, name: Any) -> None:
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict:
        return {"name": self._name, "role": "reaction"}

    def requirements(self) -> Any:
        return RequirementSet({"aux_field": self._name})


__all__ = ["ScalarCoefficient", "ReactionCoefficient"]
