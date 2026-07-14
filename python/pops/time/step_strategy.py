"""Typed, executable macro-step strategies.

Strategies are immutable authoring descriptors.  They validate their complete runtime-control
schema and construct a small runtime controller through a lazy protocol; the numerical work stays
in the native executor.  No ``run`` keyword is allowed to infer or replace a strategy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pops._ir.literals import scalar_data


_STRATEGY_TYPES: dict[str, type[StepStrategy]] = {}


def _positive_float(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a positive finite numeric scalar" % where)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("%s must be finite and > 0" % where)
    return result


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be a positive integer" % where)
    if value <= 0:
        raise ValueError("%s must be > 0" % where)
    return value


def _controls(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("runtime controls must be a mapping")
    if any(not isinstance(name, str) or not name for name in value):
        raise TypeError("runtime control names must be non-empty strings")
    return dict(value)


def _binary64(value: Any, *, where: str) -> float:
    if (not isinstance(value, Mapping) or set(value) != {"kind", "value"}
            or value.get("kind") != "binary64" or not isinstance(value.get("value"), str)):
        raise TypeError("%s must be a canonical binary64 scalar literal" % where)
    try:
        result = float.fromhex(value["value"])
    except ValueError:
        raise ValueError("%s has an invalid binary64 value" % where) from None
    if not math.isfinite(result) or result.hex() != value["value"]:
        raise ValueError("%s must be canonical and finite" % where)
    return result


@dataclass(frozen=True, slots=True)
class StepStrategy:
    """Registered strategy-provider protocol for macro-step attempt controllers."""

    kind: ClassVar[str] = "strategy"
    __pops_ir_immutable__ = True

    def __new__(cls, *args: Any, **kwargs: Any) -> StepStrategy:
        if cls is StepStrategy:
            raise TypeError(
                "StepStrategy is closed to direct construction; use a registered provider")
        # ``slots=True`` replaces the class object during dataclass decoration; zero-argument
        # ``super()`` in ``__new__`` would retain the pre-decoration class cell and reject every
        # concrete strategy. Object allocation is the complete intended protocol here.
        return object.__new__(cls)

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind}

    @classmethod
    def from_data(cls, payload: Mapping[str, Any]) -> StepStrategy:
        """Reconstruct this provider's exact descriptor for strict next-attempt restart."""
        if not isinstance(payload, Mapping) or dict(payload) != {"kind": cls.kind}:
            raise ValueError("%s strategy manifest has invalid keys" % cls.kind)
        return cls()

    def restore_runtime_controls(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Decode canonical restart controls; providers may override for non-numeric controls."""
        if not isinstance(payload, Mapping):
            raise TypeError("step strategy restart controls must be a mapping")
        return {
            name: [_binary64(item, where="controls.%s" % name) for item in value]
            if isinstance(value, list) else _binary64(value, where="controls.%s" % name)
            for name, value in payload.items()
        }

    def runtime_controls_data(self, controls: Mapping[str, Any]) -> dict[str, Any]:
        """Canonical inverse of :meth:`restore_runtime_controls`."""
        return {
            name: [scalar_data(float(item)) for item in value]
            if isinstance(value, (tuple, list)) else scalar_data(float(value))
            for name, value in controls.items()
        }

    def validate_runtime_controls(self, controls: Mapping[str, Any] | None = None) -> None:
        values = _controls(controls)
        if values:
            raise ValueError(
                "%s does not accept runtime control(s): %s"
                % (self.kind, ", ".join(sorted(values))))

    def runtime_controller(self, controls: Mapping[str, Any] | None = None) -> Any:
        raise NotImplementedError


def register_step_strategy_type(cls: type[StepStrategy]) -> type[StepStrategy]:
    """Register one immutable strategy provider, rejecting kind collisions deterministically."""
    if not isinstance(cls, type) or not issubclass(cls, StepStrategy) or cls is StepStrategy:
        raise TypeError("registered step strategy must subclass StepStrategy")
    kind = getattr(cls, "kind", None)
    if not isinstance(kind, str) or not kind or kind.strip() != kind:
        raise TypeError("registered step strategy kind must be canonical non-empty text")
    existing = _STRATEGY_TYPES.get(kind)
    if existing is not None and existing is not cls:
        raise ValueError("step strategy kind %r is already registered" % kind)
    _STRATEGY_TYPES[kind] = cls
    return cls


def registered_step_strategy_type(kind: str) -> type[StepStrategy] | None:
    """Return the exact provider authorized for ``kind``; never infer from a class name."""
    return _STRATEGY_TYPES.get(kind)


def validate_step_strategy_manifest(value: Any) -> dict[str, Any]:
    """Validate a restart strategy through its registered provider, without a central kind switch."""
    if not isinstance(value, Mapping) or set(value) != {"strategy", "controls"}:
        raise TypeError("step strategy manifest must contain exact strategy/controls mappings")
    descriptor = value["strategy"]
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("kind"), str):
        raise TypeError("step strategy descriptor must contain a non-empty kind")
    provider = registered_step_strategy_type(descriptor["kind"])
    if provider is None:
        raise ValueError("unregistered step strategy kind %r" % descriptor["kind"])
    strategy = provider.from_data(descriptor)
    if type(strategy) is not provider:
        raise TypeError("step strategy provider from_data() returned the wrong concrete type")
    controls = strategy.restore_runtime_controls(value["controls"])
    strategy.validate_runtime_controls(controls)
    canonical = {
        "strategy": strategy.to_data(),
        "controls": strategy.runtime_controls_data(controls),
    }
    if canonical != dict(value):
        raise ValueError("step strategy manifest is not the provider's canonical representation")
    return canonical


@register_step_strategy_type
@dataclass(frozen=True, slots=True)
class FixedDt(StepStrategy):
    """Advance with one authored fixed step, clipped only by the declared final time."""

    dt: float
    kind: ClassVar[str] = "fixed_dt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _positive_float(self.dt, where="FixedDt.dt"))

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, "dt": scalar_data(self.dt)}

    @classmethod
    def from_data(cls, payload: Mapping[str, Any]) -> FixedDt:
        if not isinstance(payload, Mapping) or set(payload) != {"kind", "dt"} \
                or payload.get("kind") != cls.kind:
            raise ValueError("FixedDt strategy manifest has invalid keys")
        return cls(_binary64(payload["dt"], where="FixedDt.dt"))

    def runtime_controller(self, controls: Mapping[str, Any] | None = None) -> Any:
        self.validate_runtime_controls(controls)
        from pops.runtime._step_strategy import FixedDtController
        return FixedDtController(self)


@register_step_strategy_type
@dataclass(frozen=True, slots=True)
class AdaptiveCFL(StepStrategy):
    """Use the native stability reduction with explicit optional runtime clamps."""

    cfl: float
    max_dt: float | None = None
    kind: ClassVar[str] = "adaptive_cfl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cfl", _positive_float(self.cfl, where="AdaptiveCFL.cfl"))
        if self.max_dt is not None:
            object.__setattr__(
                self, "max_dt", _positive_float(self.max_dt, where="AdaptiveCFL.max_dt"))

    def to_data(self) -> dict[str, Any]:
        data = {"kind": self.kind, "cfl": scalar_data(self.cfl)}
        if self.max_dt is not None:
            data["max_dt"] = scalar_data(self.max_dt)
        return data

    @classmethod
    def from_data(cls, payload: Mapping[str, Any]) -> AdaptiveCFL:
        if (not isinstance(payload, Mapping) or payload.get("kind") != cls.kind
                or set(payload) not in ({"kind", "cfl"}, {"kind", "cfl", "max_dt"})):
            raise ValueError("AdaptiveCFL strategy manifest has invalid keys")
        return cls(
            _binary64(payload["cfl"], where="AdaptiveCFL.cfl"),
            _binary64(payload["max_dt"], where="AdaptiveCFL.max_dt")
            if "max_dt" in payload else None)

    def validate_runtime_controls(self, controls: Mapping[str, Any] | None = None) -> None:
        values = _controls(controls)
        unknown = sorted(set(values) - {"dt_min", "dt_max"})
        if unknown:
            raise ValueError(
                "AdaptiveCFL runtime controls are dt_min/dt_max only; got %s"
                % ", ".join(unknown))
        for name, value in values.items():
            _positive_float(value, where="AdaptiveCFL runtime %s" % name)
        if "dt_min" in values and "dt_max" in values \
                and float(values["dt_min"]) > float(values["dt_max"]):
            raise ValueError("AdaptiveCFL runtime dt_min must be <= dt_max")

    def runtime_controller(self, controls: Mapping[str, Any] | None = None) -> Any:
        self.validate_runtime_controls(controls)
        from pops.runtime._step_strategy import AdaptiveCFLController
        return AdaptiveCFLController(self, _controls(controls))


@register_step_strategy_type
@dataclass(frozen=True, slots=True)
class ErrorControlledDt(StepStrategy):
    """Retry rejected error-guarded attempts and adapt the next proposal explicitly.

    The compiled Program must contain an error-estimate acceptance guard.  ``shrink`` is applied to a
    rejected attempt and ``growth`` after an accepted attempt; both are inspectable policy values.
    """

    dt_init: float
    rtol: float
    atol: float
    dt_min: float
    dt_max: float
    max_rejections: int
    shrink: float = 0.5
    growth: float = 1.25
    kind: ClassVar[str] = "error_controlled_dt"

    def __post_init__(self) -> None:
        for name in ("dt_init", "rtol", "atol", "dt_min", "dt_max", "shrink", "growth"):
            object.__setattr__(self, name, _positive_float(
                getattr(self, name), where="ErrorControlledDt.%s" % name))
        object.__setattr__(self, "max_rejections", _positive_int(
            self.max_rejections, where="ErrorControlledDt.max_rejections"))
        if self.dt_min > self.dt_init or self.dt_init > self.dt_max:
            raise ValueError("ErrorControlledDt requires dt_min <= dt_init <= dt_max")
        if not self.shrink < 1.0:
            raise ValueError("ErrorControlledDt.shrink must be < 1")
        if not self.growth >= 1.0:
            raise ValueError("ErrorControlledDt.growth must be >= 1")

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            **{name: scalar_data(getattr(self, name)) for name in (
                "dt_init", "rtol", "atol", "dt_min", "dt_max", "shrink", "growth")},
            "max_rejections": self.max_rejections,
        }

    @classmethod
    def from_data(cls, payload: Mapping[str, Any]) -> ErrorControlledDt:
        names = {"dt_init", "rtol", "atol", "dt_min", "dt_max", "shrink", "growth"}
        if (not isinstance(payload, Mapping) or payload.get("kind") != cls.kind
                or set(payload) != names | {"kind", "max_rejections"}):
            raise ValueError("ErrorControlledDt strategy manifest has invalid keys")
        return cls(
            **{name: _binary64(payload[name], where="ErrorControlledDt.%s" % name)
               for name in names},
            max_rejections=payload["max_rejections"])

    def runtime_controller(self, controls: Mapping[str, Any] | None = None) -> Any:
        self.validate_runtime_controls(controls)
        from pops.runtime._step_strategy import ErrorControlledDtController
        return ErrorControlledDtController(self)


@register_step_strategy_type
@dataclass(frozen=True, slots=True)
class ExternalTimeGrid(StepStrategy):
    """Follow the strictly increasing grid supplied under ``grid_id`` at run time."""

    grid_id: str
    kind: ClassVar[str] = "external_time_grid"

    def __post_init__(self) -> None:
        if not isinstance(self.grid_id, str) or not self.grid_id:
            raise ValueError("ExternalTimeGrid.grid_id must be a non-empty string")

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, "grid_id": self.grid_id}

    @classmethod
    def from_data(cls, payload: Mapping[str, Any]) -> ExternalTimeGrid:
        if not isinstance(payload, Mapping) or set(payload) != {"kind", "grid_id"} \
                or payload.get("kind") != cls.kind:
            raise ValueError("ExternalTimeGrid strategy manifest has invalid keys")
        return cls(payload["grid_id"])

    def validate_runtime_controls(self, controls: Mapping[str, Any] | None = None) -> None:
        values = _controls(controls)
        if set(values) != {self.grid_id}:
            raise ValueError(
                "ExternalTimeGrid(%r) requires exactly that named runtime grid" % self.grid_id)
        grid = values[self.grid_id]
        if not isinstance(grid, (tuple, list)) or len(grid) < 2:
            raise ValueError("ExternalTimeGrid runtime grid must contain at least two times")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in grid):
            raise TypeError("ExternalTimeGrid runtime grid values must be numeric")
        if any(not math.isfinite(float(value)) for value in grid):
            raise ValueError("ExternalTimeGrid runtime grid values must be finite")
        if any(float(b) <= float(a) for a, b in zip(grid, grid[1:], strict=False)):
            raise ValueError("ExternalTimeGrid runtime grid must be strictly increasing")

    def runtime_controller(self, controls: Mapping[str, Any] | None = None) -> Any:
        self.validate_runtime_controls(controls)
        from pops.runtime._step_strategy import ExternalTimeGridController
        return ExternalTimeGridController(self, tuple(float(value) for value in controls[self.grid_id]))


__all__ = [
    "AdaptiveCFL", "ErrorControlledDt", "ExternalTimeGrid", "FixedDt", "StepStrategy",
    "register_step_strategy_type", "registered_step_strategy_type",
    "validate_step_strategy_manifest",
]
