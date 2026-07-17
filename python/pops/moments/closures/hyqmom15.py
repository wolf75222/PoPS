"""pops.moments.closures.hyqmom15 -- the HyQMOM15 (order-4) closure.

The HyQMOM15 system is the 15-variable order-4 2D moment hierarchy. Its provided
closure is the Levermore / Gaussian closure of order 4 (:func:`gaussian_closure(4)`),
which is the closure the adc_cases HyQMOM15 reference validates against.

User closures use the same generic :class:`LocalClosure` contract; this provided class
contains no reserved variant selector or model-specific lowering branch.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet

from .gaussian import gaussian_closure

_HYQMOM15_ORDER = 4


class HyQMOM15Closure(Descriptor):
    """The HyQMOM15 (order-4) moment closure (route-choosing descriptor).

    Delegates to :func:`gaussian_closure(4)` -- the standardized order-5 moments of a
    Gaussian. Custom physics is authored with ``@closure(4)`` and supplied to the model
    factory, not selected by a string variant.

    A :class:`HyQMOM15Closure` instance is itself a closure callable: ``self(S)`` returns the
    order-5 standardized moments the generator consumes.
    """

    category = "closure"

    def __init__(self) -> None:
        self.order = _HYQMOM15_ORDER
        self._closure = gaussian_closure(_HYQMOM15_ORDER)

    def options(self) -> dict:
        return {"order": self.order, "local_operator": self._closure.contract_data()}

    def capabilities(self) -> Any:
        return CapabilitySet({"provides": "order_%d_standardized_moments" % self.order})

    def __call__(self, S: Any) -> Any:  # noqa: N803  (S mirrors the engine variable name)
        return self._closure(S)

    def __repr__(self) -> str:
        return "HyQMOM15Closure(order=4)"


__all__ = ["HyQMOM15Closure"]
