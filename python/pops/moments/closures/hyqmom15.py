"""The fourth-order, two-dimensional HyQMOM closure.

The six formulas below are the standardized fifth-order closure from the HyQMOM
reference.  They are evaluated once on symbolic expressions while the model is
authored, then folded into the ordinary moment-flux graph.  No HyQMOM-specific branch
exists in the compiler or in the native runtime.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet

from .protocol import LocalClosure

_HYQMOM15_ORDER = 4


def _hyqmom15_polynomial(S: Any) -> dict[str, Any]:  # noqa: N803
    """Return the six standardized fifth-order moments of the HyQMOM closure."""

    s03 = S["S03"]
    s04 = S["S04"]
    s11 = S["S11"]
    s12 = S["S12"]
    s13 = S["S13"]
    s21 = S["S21"]
    s22 = S["S22"]
    s30 = S["S30"]
    s31 = S["S31"]
    s40 = S["S40"]

    return {
        "S50": 0.5 * s30 * (5.0 * s40 - 3.0 * s30 * s30 - 1.0),
        "S41": (
            -0.25 * s30 * (8.0 * s40 - 9.0 * s30 * s30 - 4.0) * s11
            + 0.25 * (10.0 * s40 - 15.0 * s30 * s30 - 6.0) * s21
            + 2.0 * s30 * s31
        ),
        "S32": (
            0.5 * (2.0 * s40 - 3.0 * s30 * s30) * s12
            + 0.5 * (3.0 * s22 - 1.0) * s30
        ),
        "S23": (
            0.5 * (2.0 * s04 - 3.0 * s03 * s03) * s21
            + 0.5 * (3.0 * s22 - 1.0) * s03
        ),
        "S14": (
            -0.25 * s03 * (8.0 * s04 - 9.0 * s03 * s03 - 4.0) * s11
            + 0.25 * (10.0 * s04 - 15.0 * s03 * s03 - 6.0) * s12
            + 2.0 * s03 * s13
        ),
        "S05": 0.5 * s03 * (5.0 * s04 - 3.0 * s03 * s03 - 1.0),
    }


class HyQMOM15Closure(Descriptor):
    """The HyQMOM15 (order-4) moment closure (route-choosing descriptor).

    The descriptor delegates to the same generic :class:`LocalClosure` protocol used by
    user-authored ``@closure(4)`` functions.  It contributes only symbolic arithmetic;
    custom physics is supplied to the model factory, not selected by a string variant.

    A :class:`HyQMOM15Closure` instance is itself a closure callable: ``self(S)`` returns the
    order-5 standardized moments the generator consumes.
    """

    category = "closure"

    def __init__(self) -> None:
        self.order = _HYQMOM15_ORDER
        self._closure = LocalClosure(
            _HYQMOM15_ORDER,
            "hyqmom15_polynomial",
            _hyqmom15_polynomial,
        )

    def options(self) -> dict:
        return {"order": self.order, "local_operator": self._closure.contract_data()}

    def capabilities(self) -> Any:
        return CapabilitySet({"provides": "order_%d_standardized_moments" % self.order})

    def __call__(self, S: Any) -> Any:  # noqa: N803  (S mirrors the engine variable name)
        return self._closure(S)

    def __repr__(self) -> str:
        return "HyQMOM15Closure(order=4)"


__all__ = ["HyQMOM15Closure"]
