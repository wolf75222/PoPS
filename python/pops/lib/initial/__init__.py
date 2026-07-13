"""Pre-implemented, data-only initial profiles.

Profiles contain no Python callback.  They expose a small open protocol consumed by
``pops.initial.InitialCondition``: ``validate_for``, ``initial_source_options`` and ``to_data``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a finite numeric scalar" % where)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % where)
    return result


def _declaration(state: Any) -> Any:
    return getattr(state, "declaration_ref", None) or state


def _component_count(state: Any) -> int:
    declaration = _declaration(state)
    components = getattr(declaration, "components", None)
    if components is None:
        components = getattr(getattr(declaration, "space", None), "components", None)
    if not isinstance(components, tuple) or not components:
        raise TypeError("initial profile requires a state with declared components")
    return len(components)


@dataclass(frozen=True, slots=True, init=False)
class Constant:
    """One constant per state component; supported by the native AMR bootstrap route."""

    components: tuple[float, ...]
    native_route = "constant_field"
    reprojectable = True
    __pops_ir_immutable__ = True

    def __init__(self, components: Any) -> None:
        if isinstance(components, (str, bytes)):
            raise TypeError("Constant components must be an ordered numeric sequence")
        try:
            values = tuple(components)
        except TypeError as exc:
            raise TypeError("Constant components must be an ordered numeric sequence") from exc
        if not values:
            raise ValueError("Constant requires at least one component")
        object.__setattr__(self, "components", tuple(
            _finite(value, where="Constant.components[]") for value in values))

    def validate_for(self, state: Any) -> bool:
        if len(self.components) != _component_count(state):
            raise ValueError(
                "Constant component count does not match the target state")
        return True

    def initial_source_options(self) -> dict[str, Any]:
        return {"native_route": self.native_route, "components": list(self.components)}

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "constant",
            "components": list(self.components),
        }

    canonical_identity = to_data
    inspect = to_data


@dataclass(frozen=True, slots=True, init=False)
class Gaussian:
    """A scalar Gaussian profile expressed in one immutable physical frame."""

    frame: Any
    center: tuple[tuple[Any, float], ...]
    background: float
    amplitude: float
    inverse_width: float
    native_route = "gaussian_field"
    reprojectable = True
    __pops_ir_immutable__ = True

    def __init__(
        self,
        *,
        frame: Any,
        center: Any,
        background: Any = 0.0,
        amplitude: Any = 1.0,
        inverse_width: Any = 1.0,
    ) -> None:
        axes = getattr(frame, "axes", None)
        if not isinstance(axes, tuple) or not axes:
            raise TypeError("Gaussian frame must expose immutable typed axes")
        if not isinstance(center, Mapping) or set(center) != set(axes):
            raise ValueError("Gaussian center must map every typed frame axis exactly once")
        width = _finite(inverse_width, where="Gaussian.inverse_width")
        if width <= 0.0:
            raise ValueError("Gaussian.inverse_width must be > 0")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "center", tuple(
            (axis, _finite(center[axis], where="Gaussian.center[%s]" % axis.name))
            for axis in axes))
        object.__setattr__(self, "background", _finite(background, where="Gaussian.background"))
        object.__setattr__(self, "amplitude", _finite(amplitude, where="Gaussian.amplitude"))
        object.__setattr__(self, "inverse_width", width)

    def validate_for(self, state: Any) -> bool:
        if _component_count(state) != 1:
            raise ValueError("Gaussian is a scalar profile and requires a one-component state")
        space = getattr(_declaration(state), "space", None)
        frame_id = getattr(self.frame, "canonical_id", None)
        if space is None or getattr(space, "frame", None) != frame_id:
            raise ValueError("Gaussian frame differs from the target state frame")
        return True

    def initial_source_options(self) -> dict[str, Any]:
        return {
            "native_route": self.native_route,
            "frame_id": self.frame.canonical_id,
            "center": {axis.name: value for axis, value in self.center},
            "background": self.background,
            "amplitude": self.amplitude,
            "inverse_width": self.inverse_width,
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "gaussian",
            "frame_id": self.frame.canonical_id,
            "center": {axis.name: value for axis, value in self.center},
            "background": self.background,
            "amplitude": self.amplitude,
            "inverse_width": self.inverse_width,
        }

    canonical_identity = to_data
    inspect = to_data


__all__ = ["Constant", "Gaussian"]
