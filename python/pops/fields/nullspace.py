"""pops.fields.nullspace -- typed nullspace declarations for a field solve (Spec 5 sec.5.5).

A pure-Neumann / fully periodic elliptic operator has a non-trivial nullspace. This module
declares the mathematical kernel only. Selecting a representative solution is a distinct
typed gauge choice in :mod:`pops.fields.gauges`.

Inert descriptors; they compute nothing.
"""

from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet


class ConstantNullspace(Descriptor):
    """The constant-function nullspace of a pure-Neumann / periodic elliptic operator.

    Declaring it tells the solver which mode must be projected. It does not choose a gauge.
    """

    category = "nullspace"

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_frozen":
            super().__setattr__(key, value)
            return
        raise AttributeError("ConstantNullspace has no configurable fields")

    def options(self) -> dict:
        return {"nullspace": "constant"}

    def to_data(self) -> dict:
        return {"type": type(self).__name__, "options": self.options()}

    def capabilities(self) -> Any:
        return CapabilitySet({"removes_constant": True})


__all__ = ["ConstantNullspace"]
