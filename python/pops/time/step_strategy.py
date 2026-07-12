"""Explicit runtime step-strategy descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pops.ir.literals import scalar_data


def _positive_number(value: Any, *, where: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a positive numeric scalar" % where)
    if not value > 0:
        raise ValueError("%s must be > 0" % where)
    return scalar_data(value)


@dataclass(frozen=True)
class StepStrategy:
    """Closed base for explicit attempt controllers."""

    kind: ClassVar[str] = "strategy"
    __pops_ir_immutable__ = True

    def __new__(cls, *args: Any, **kwargs: Any) -> StepStrategy:
        if cls is StepStrategy:
            raise TypeError(
                "StepStrategy is closed; use FixedDt/AdaptiveCFL/ErrorControlledDt/ExternalTimeGrid")
        return super().__new__(cls)

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def validate_runtime_controls(self, controls: dict[str, Any] | None = None) -> None:
        controls = {} if controls is None else dict(controls)
        if controls:
            raise ValueError(
                "%s does not accept runtime control(s): %s"
                % (self.kind, ", ".join(sorted(controls))))


@dataclass(frozen=True)
class FixedDt(StepStrategy):
    """Use an authored fixed dt; runtime CFL/error kwargs cannot select another controller."""

    dt: Any | None = None
    kind: ClassVar[str] = "fixed_dt"

    def __post_init__(self) -> None:
        if self.dt is not None:
            object.__setattr__(self, "dt", _positive_number(self.dt, where="FixedDt.dt"))

    def to_data(self) -> dict[str, Any]:
        data = super().to_data()
        if self.dt is not None:
            data["dt"] = self.dt
        return data


@dataclass(frozen=True)
class AdaptiveCFL(StepStrategy):
    """Controller selected explicitly by the Program, not inferred from run kwargs."""

    cfl: Any
    max_dt: Any | None = None
    kind: ClassVar[str] = "adaptive_cfl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cfl", _positive_number(self.cfl, where="AdaptiveCFL.cfl"))
        if self.max_dt is not None:
            object.__setattr__(
                self, "max_dt", _positive_number(self.max_dt, where="AdaptiveCFL.max_dt"))

    def to_data(self) -> dict[str, Any]:
        data = {"kind": self.kind, "cfl": self.cfl}
        if self.max_dt is not None:
            data["max_dt"] = self.max_dt
        return data

    def validate_runtime_controls(self, controls: dict[str, Any] | None = None) -> None:
        controls = {} if controls is None else dict(controls)
        allowed = {"dt_min", "dt_max"}
        unknown = sorted(set(controls) - allowed)
        if unknown:
            raise ValueError(
                "AdaptiveCFL runtime controls must be dt_min/dt_max only; got %s"
                % ", ".join(unknown))
        for name, value in controls.items():
            _positive_number(value, where="AdaptiveCFL runtime %s" % name)


@dataclass(frozen=True)
class ErrorControlledDt(StepStrategy):
    """Attempt controller driven by a declared local/global error estimator."""

    rtol: Any
    atol: Any
    kind: ClassVar[str] = "error_controlled_dt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rtol", _positive_number(self.rtol, where="ErrorControlledDt.rtol"))
        object.__setattr__(self, "atol", _positive_number(self.atol, where="ErrorControlledDt.atol"))

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, "rtol": self.rtol, "atol": self.atol}


@dataclass(frozen=True)
class ExternalTimeGrid(StepStrategy):
    """Controller following a runtime-supplied monotonically increasing grid."""

    grid_id: str
    kind: ClassVar[str] = "external_time_grid"

    def __post_init__(self) -> None:
        if not isinstance(self.grid_id, str) or not self.grid_id:
            raise ValueError("ExternalTimeGrid.grid_id must be a non-empty string")

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind, "grid_id": self.grid_id}

    def validate_runtime_controls(self, controls: dict[str, Any] | None = None) -> None:
        controls = {} if controls is None else dict(controls)
        if set(controls) != {"grid"}:
            raise ValueError("ExternalTimeGrid requires exactly one runtime control: grid")
        grid = controls["grid"]
        if not isinstance(grid, (tuple, list)) or len(grid) < 2:
            raise ValueError("ExternalTimeGrid runtime grid must contain at least two times")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in grid):
            raise TypeError("ExternalTimeGrid runtime grid values must be numeric")
        if any(float(b) <= float(a) for a, b in zip(grid, grid[1:], strict=False)):
            raise ValueError("ExternalTimeGrid runtime grid must be strictly increasing")


__all__ = [
    "AdaptiveCFL", "ErrorControlledDt", "ExternalTimeGrid", "FixedDt", "StepStrategy",
]
