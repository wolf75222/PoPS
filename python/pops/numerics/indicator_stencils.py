"""Typed discrete stencils used to lower continuous-looking AMR indicators.

The spatial method owns this data.  AMR resolution copies the exact coefficients into the
resolved tag graph; neither code generation nor the runtime is allowed to select a stencil by
class name, reconstruction token, or fallback default.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, ClassVar

from pops.identity import make_identity
from pops.identity.semantic import semantic_value


_ROUTE = "linear_axis_stencil_l2_v1"
_BOUNDARY_MODE = "ghost_extension"
_SCALE = "inverse_cell_size"


@dataclass(frozen=True, slots=True)
class LinearAxisStencil:
    """One first-derivative row, applied along a named Cartesian axis."""

    offsets: tuple[int, ...]
    coefficients: tuple[float, ...]
    formal_order: int
    derivative_order: int = 1
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        offsets = tuple(self.offsets)
        coefficients = tuple(self.coefficients)
        if not offsets or len(offsets) != len(coefficients):
            raise ValueError("LinearAxisStencil requires matching non-empty terms")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in offsets):
            raise TypeError("LinearAxisStencil offsets must be exact integers")
        if len(set(offsets)) != len(offsets):
            raise ValueError("LinearAxisStencil offsets must be unique")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in coefficients):
            raise TypeError("LinearAxisStencil coefficients must be finite scalars")
        coefficients = tuple(float(value) for value in coefficients)
        if isinstance(self.formal_order, bool) or not isinstance(self.formal_order, int) \
                or self.formal_order < 1:
            raise ValueError("LinearAxisStencil formal_order must be an integer >= 1")
        if self.formal_order > len(offsets):
            raise ValueError(
                "LinearAxisStencil formal_order exceeds its finite term capacity")
        if self.derivative_order != 1:
            raise NotImplementedError(
                "linear_axis_stencil_l2_v1 supports first derivatives only")
        for power in range(self.formal_order + 1):
            terms = tuple(
                coefficient * (offset ** power)
                for offset, coefficient in zip(offsets, coefficients, strict=True))
            moment = math.fsum(terms)
            expected = 1.0 if power == 1 else 0.0
            tolerance = 1.0e-13 * max(1.0, math.fsum(abs(value) for value in terms))
            if not math.isclose(moment, expected, rel_tol=1.0e-13,
                                abs_tol=tolerance):
                raise ValueError(
                    "LinearAxisStencil coefficients do not authenticate formal_order=%d "
                    "(moment %d)" % (self.formal_order, power))
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def ghost_lower(self) -> int:
        return max(0, -min(self.offsets))

    @property
    def ghost_upper(self) -> int:
        return max(0, max(self.offsets))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stencil_type": "linear_first_derivative",
            "offsets": list(self.offsets),
            "coefficients": [
                {"binary64": value.hex()} for value in self.coefficients],
            "formal_order": self.formal_order,
            "derivative_order": self.derivative_order,
            "ghost_lower": self.ghost_lower,
            "ghost_upper": self.ghost_upper,
        }

    @classmethod
    def from_data(cls, data: Any) -> LinearAxisStencil:
        if not isinstance(data, dict) or set(data) != {
                "schema_version", "stencil_type", "offsets", "coefficients",
                "formal_order", "derivative_order", "ghost_lower", "ghost_upper"} \
                or data.get("schema_version") != 1 \
                or data.get("stencil_type") != "linear_first_derivative":
            raise TypeError("AMR gradient axis stencil has an unsupported schema")
        encoded_coefficients = data["coefficients"]
        if not isinstance(encoded_coefficients, list) or any(
                not isinstance(value, dict) or set(value) != {"binary64"}
                or not isinstance(value["binary64"], str)
                for value in encoded_coefficients):
            raise TypeError(
                "AMR gradient axis coefficients must be canonical binary64 values")
        try:
            coefficients = tuple(
                float.fromhex(value["binary64"]) for value in encoded_coefficients)
        except (ValueError, OverflowError):
            raise ValueError(
                "AMR gradient axis coefficient is not a valid binary64 value") from None
        result = cls(
            tuple(data["offsets"]), coefficients,
            data["formal_order"], data["derivative_order"])
        if data["ghost_lower"] != result.ghost_lower \
                or data["ghost_upper"] != result.ghost_upper:
            raise ValueError("AMR gradient axis stencil repeats inconsistent halo depths")
        return result


@dataclass(frozen=True, slots=True)
class DiscreteGradientStencil:
    """Exact separable Cartesian gradient and L2 norm used by one AMR leaf."""

    axes: tuple[LinearAxisStencil, ...]
    route: str = _ROUTE
    norm: str = "l2"
    scale: str = _SCALE
    boundary_mode: str = _BOUNDARY_MODE
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not 1 <= len(axes) <= 3 or any(type(axis) is not LinearAxisStencil for axis in axes):
            raise TypeError("DiscreteGradientStencil requires one typed axis per dimension")
        if self.route != _ROUTE or self.norm != "l2" or self.scale != _SCALE \
                or self.boundary_mode != _BOUNDARY_MODE:
            raise NotImplementedError(
                "unsupported AMR discrete-gradient route; no runtime fallback exists")
        object.__setattr__(self, "axes", axes)

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def identity(self) -> str:
        return make_identity(
            "amr-discrete-gradient-stencil",
            semantic_value(self._payload(), where="AMR discrete gradient stencil"),
        ).token

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stencil_type": "separable_cartesian_gradient",
            "route": self.route,
            "norm": self.norm,
            "scale": self.scale,
            "boundary_mode": self.boundary_mode,
            "dimension": self.dimension,
            "axes": [
                {"axis": index, **axis.to_data()}
                for index, axis in enumerate(self.axes)
            ],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity}

    @classmethod
    def from_data(cls, data: Any) -> DiscreteGradientStencil:
        expected = {
            "schema_version", "stencil_type", "route", "norm", "scale",
            "boundary_mode", "dimension", "axes", "identity",
        }
        if not isinstance(data, dict) or set(data) != expected \
                or data.get("schema_version") != 1 \
                or data.get("stencil_type") != "separable_cartesian_gradient":
            raise TypeError("AMR discrete gradient stencil has an unsupported schema")
        dimension = data["dimension"]
        rows = data["axes"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) \
                or not isinstance(rows, list) or len(rows) != dimension:
            raise ValueError("AMR discrete gradient stencil has inconsistent dimension")
        axes = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("axis") != index:
                raise ValueError("AMR discrete gradient axes must be canonical and contiguous")
            axes.append(LinearAxisStencil.from_data({
                key: value for key, value in row.items() if key != "axis"}))
        result = cls(tuple(axes), data["route"], data["norm"], data["scale"],
                     data["boundary_mode"])
        if data["identity"] != result.identity:
            raise ValueError("AMR discrete gradient stencil identity is not authentic")
        return result


SECOND_ORDER_AXIS = LinearAxisStencil((-1, 1), (-0.5, 0.5), formal_order=2)
FOURTH_ORDER_AXIS = LinearAxisStencil(
    (-2, -1, 1, 2), (1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0),
    formal_order=4,
)


def gradient_stencil(axis: LinearAxisStencil, *, dimension: int) -> DiscreteGradientStencil:
    """Expand one reconstruction-owned axis row to an exact Cartesian dimension."""

    if type(axis) is not LinearAxisStencil:
        raise TypeError("gradient_stencil requires a LinearAxisStencil")
    if isinstance(dimension, bool) or dimension not in (1, 2, 3):
        raise ValueError("gradient_stencil dimension must be 1, 2, or 3")
    return DiscreteGradientStencil((axis,) * dimension)


__all__ = [
    "DiscreteGradientStencil", "FOURTH_ORDER_AXIS", "LinearAxisStencil",
    "SECOND_ORDER_AXIS", "gradient_stencil",
]
