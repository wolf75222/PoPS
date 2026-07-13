"""Execution protocol for the four typed :mod:`pops.time` step strategies."""
from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pops._bootstrap import StepAttemptRejected
from pops.time.step_strategy import StepStrategy
from pops.time.step_transaction import StepTransactionReport


def _stores(engine: Any) -> tuple[str, ...]:
    plan = getattr(engine, "_step_transaction_plan", None)
    return tuple(store.value for store in plan.stores) if plan is not None else ()


def _phase(error: BaseException) -> str:
    value = getattr(error, "phase", None)
    if callable(value):
        value = value()
    if value in {
        "prepare", "stage", "solve", "synchronize", "guard", "effect", "commit",
    }:
        return value
    message = str(error)
    for phase in ("prepare", "stage", "solve", "synchronize", "guard", "effect", "commit"):
        if " during %s:" % phase in message:
            return phase
    return "solve"


def _control_identity(controls: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    values = {} if controls is None else dict(controls)
    return tuple(sorted(
        (name, tuple(value) if isinstance(value, (tuple, list)) else value)
        for name, value in values.items()
    ))


def _record_failure(engine: Any, error: BaseException, attempts: int) -> None:
    rejected = isinstance(error, StepAttemptRejected)
    stores = _stores(engine)
    engine._last_step_transaction_report = StepTransactionReport(
        status="rejected" if rejected else "failed",
        phase=_phase(error),
        action="reject_attempt" if rejected else "fail_run",
        attempts=attempts,
        staged_effects=stores,
        rolled_back_effects=stores,
        diagnostics=(str(error),),
    )


def _native_attempt(engine: Any, native: Any, advance: Any) -> Any:
    temporal = getattr(engine, "_temporal_restart_state", None)
    before_time, before_step = native.time(), native.macro_step()
    if temporal is not None:
        temporal.before_attempt(time=before_time, macro_step=before_step)
    try:
        result = advance()
    except BaseException as error:
        if temporal is not None:
            from pops.runtime._temporal_restart import is_rejected_attempt
            recorder = temporal.reject if is_rejected_attempt(error) else temporal.fail
            recorder(time=native.time(), macro_step=native.macro_step())
        raise
    if temporal is not None:
        temporal.accept(
            before_time=before_time, before_step=before_step,
            time=native.time(), macro_step=native.macro_step())
    return result


class StepController(ABC):
    """Small runtime protocol; implementations choose dt, native executors advance fields."""

    def __init__(self, strategy: StepStrategy, controls: Mapping[str, Any] | None = None) -> None:
        self.strategy = strategy
        self.controls = _control_identity(controls)
        self.attempts = 0

    def matches(self, strategy: StepStrategy, controls: Mapping[str, Any] | None) -> bool:
        return self.strategy == strategy and self.controls == _control_identity(controls)

    def restore_temporal_state(self, temporal: Any) -> None:
        """Restore provider-owned proposal state; stateless controllers need no action."""

    @abstractmethod
    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        """Execute until one macro-step is accepted and return the number of native attempts."""


class FixedDtController(StepController):
    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        self.attempts = 1
        dt = min(self.strategy.dt, t_end - float(native.time()))
        if not dt > 0.0:
            raise RuntimeError("FixedDt has no positive interval left before the final time")
        _native_attempt(engine, native, lambda: native.step(dt))
        return 1


class AdaptiveCFLController(StepController):
    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        self.attempts = 1
        remaining = t_end - float(native.time())
        caps = [remaining]
        if self.strategy.max_dt is not None:
            caps.append(self.strategy.max_dt)
        controls = dict(self.controls)
        if "dt_max" in controls:
            caps.append(float(controls["dt_max"]))
        max_dt = min(caps)
        min_dt = float(controls.get("dt_min", 0.0))
        _native_attempt(
            engine, native,
            lambda: native.step_cfl(self.strategy.cfl, max_dt=max_dt, min_dt=min_dt),
        )
        return 1


class ErrorControlledDtController(StepController):
    def __init__(self, strategy: StepStrategy) -> None:
        super().__init__(strategy)
        self.next_dt = strategy.dt_init

    def restore_temporal_state(self, temporal: Any) -> None:
        if temporal is None or not getattr(temporal, "_restored_pending", False):
            return
        last_hex = getattr(temporal, "controller_state", {}).get("last_accepted_dt")
        if last_hex is None:
            raise RuntimeError(
                "ErrorControlledDt restart lacks the accepted dt needed for the next proposal")
        self.next_dt = min(
            self.strategy.dt_max, float.fromhex(last_hex) * self.strategy.growth)

    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        attempts = 0
        self.attempts = 0
        proposal = min(self.next_dt, self.strategy.dt_max, t_end - float(native.time()))
        while True:
            attempts += 1
            self.attempts = attempts
            try:
                _native_attempt(engine, native, lambda proposal=proposal: native.step(proposal))
            except StepAttemptRejected:
                if attempts > self.strategy.max_rejections:
                    raise
                reduced = proposal * self.strategy.shrink
                if reduced < self.strategy.dt_min:
                    raise
                proposal = reduced
                continue
            self.next_dt = min(self.strategy.dt_max, proposal * self.strategy.growth)
            return attempts


class ExternalTimeGridController(StepController):
    def __init__(self, strategy: StepStrategy, grid: tuple[float, ...]) -> None:
        super().__init__(strategy, {strategy.grid_id: grid})
        self.grid = grid

    @staticmethod
    def _same_time(left: float, right: float) -> bool:
        scale = max(1.0, abs(left), abs(right))
        return abs(left - right) <= 4.0 * math.ulp(scale)

    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        self.attempts = 1
        now = float(native.time())
        index = bisect.bisect_left(self.grid, now)
        if index == len(self.grid) or not self._same_time(self.grid[index], now):
            if index and self._same_time(self.grid[index - 1], now):
                index -= 1
            else:
                raise RuntimeError("ExternalTimeGrid current time is not a declared grid point")
        if index + 1 >= len(self.grid):
            raise RuntimeError("ExternalTimeGrid is exhausted")
        next_time = self.grid[index + 1]
        if next_time > t_end and not self._same_time(next_time, t_end):
            raise RuntimeError("ExternalTimeGrid final time is not a declared grid point")
        _native_attempt(engine, native, lambda: native.step(next_time - now))
        return 1


def resolve_run_strategy(engine: Any) -> StepStrategy:
    """Resolve the sole strategy authenticated by the installed Program."""
    from pops.time.step_transaction import ensure_step_strategy

    selected = getattr(engine, "_step_strategy", None)
    try:
        selected = ensure_step_strategy(selected)
    except TypeError:
        raise TypeError(
            "run requires a Program.step_strategy(...) contract authenticated at installation"
        ) from None
    return selected


def _controller(engine: Any, strategy: StepStrategy, controls: Mapping[str, Any] | None) -> StepController:
    strategy.validate_runtime_controls(controls)
    current = getattr(engine, "_step_controller", None)
    if current is None or not current.matches(strategy, controls):
        current = strategy.runtime_controller(controls)
        if not isinstance(current, StepController):
            raise TypeError("StepStrategy.runtime_controller must return a StepController")
        current.restore_temporal_state(getattr(engine, "_temporal_restart_state", None))
        engine._step_controller = current
    return current


def prepare_step_controller(
    engine: Any,
    strategy: StepStrategy,
    controls: Mapping[str, Any] | None = None,
) -> StepController:
    """Validate the complete execution contract before any attempt or side effect."""
    return _controller(engine, strategy, controls)


def run_step_attempt(
    engine: Any,
    native: Any,
    strategy: StepStrategy,
    *,
    t_end: float,
    controls: Mapping[str, Any] | None = None,
) -> StepTransactionReport:
    """Execute one accepted macro-step, retrying only through its declared controller."""
    controller = _controller(engine, strategy, controls)
    try:
        attempts = controller.execute(engine, native, t_end=float(t_end))
    except BaseException as error:
        attempts = max(1, getattr(controller, "attempts", 1))
        _record_failure(engine, error, attempts)
        raise
    stores = _stores(engine)
    report = StepTransactionReport(
        status="accepted", phase="commit", action="commit", attempts=attempts,
        staged_effects=stores, committed_effects=stores,
    )
    engine._last_step_transaction_report = report
    return report


def run_control_payload(
    strategy: StepStrategy, controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable execution manifest preserving the exact strategy and runtime controls."""
    from pops.ir.literals import scalar_data

    strategy.validate_runtime_controls(controls)
    values = {} if controls is None else dict(controls)
    canonical_controls = {
        name: [scalar_data(float(item)) for item in value]
        if isinstance(value, (tuple, list)) else scalar_data(float(value))
        for name, value in values.items()
    }
    return {
        "strategy": strategy.to_data(),
        "controls": canonical_controls,
    }


__all__ = [
    "AdaptiveCFLController", "ErrorControlledDtController", "ExternalTimeGridController",
    "FixedDtController", "StepAttemptRejected", "StepController", "prepare_step_controller",
    "resolve_run_strategy", "run_control_payload", "run_step_attempt",
]
