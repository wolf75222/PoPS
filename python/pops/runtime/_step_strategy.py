"""Runtime execution helpers for typed StepStrategy descriptors."""
from __future__ import annotations

from typing import Any

from pops.time.step_strategy import AdaptiveCFL, ErrorControlledDt, ExternalTimeGrid, FixedDt, StepStrategy
from pops._bootstrap import StepAttemptRejected


def _number(value: Any) -> float:
    if isinstance(value, dict) and value.get("kind") == "binary64":
        return float.fromhex(value["value"])
    return float(value)


def resolve_run_strategy(engine: Any, strategy: Any, cfl: Any) -> StepStrategy:
    """Return one explicit controller and reject mixed control surfaces."""
    if strategy is None:
        if cfl is None:
            cfl = getattr(engine, "_program_cadence_cfl", None)
            if cfl is None:
                cfl = 0.4
        return AdaptiveCFL(cfl)
    if not isinstance(strategy, StepStrategy):
        raise TypeError("run(strategy=...): expected a pops.time.StepStrategy")
    if cfl is not None:
        raise ValueError("run: cfl= cannot be combined with strategy=; use AdaptiveCFL(cfl)")
    return strategy


def run_step_attempt(engine: Any, native: Any, strategy: StepStrategy, *, t_end: float) -> None:
    """Execute one declared attempt through the selected controller."""
    temporal = getattr(engine, "_temporal_restart_state", None)
    if temporal is not None:
        before_time, before_step = native.time(), native.macro_step()
        temporal.before_attempt(time=before_time, macro_step=before_step)
    try:
        if isinstance(strategy, AdaptiveCFL):
            if strategy.max_dt is not None:
                raise NotImplementedError(
                    "AdaptiveCFL(max_dt=...) requires a clamped native attempt controller")
            native.step_cfl(_number(strategy.cfl))
        elif isinstance(strategy, FixedDt):
            if strategy.dt is None:
                raise ValueError("FixedDt runtime execution requires FixedDt(dt=...)")
            native.step(_number(strategy.dt))
        elif isinstance(strategy, ExternalTimeGrid):
            raise NotImplementedError(
                "ExternalTimeGrid execution requires a runtime grid control wired through StepTransaction")
        elif isinstance(strategy, ErrorControlledDt):
            raise NotImplementedError(
                "ErrorControlledDt requires an error estimator/controller; no runtime controller is wired yet")
        else:
            raise TypeError("unknown StepStrategy %r" % (strategy,))
    except BaseException as error:
        if temporal is not None:
            from pops.runtime._temporal_restart import is_rejected_attempt
            record = temporal.reject if is_rejected_attempt(error) else temporal.fail
            record(time=native.time(), macro_step=native.macro_step())
        raise
    if temporal is not None:
        temporal.accept(
            before_time=before_time, before_step=before_step,
            time=native.time(), macro_step=native.macro_step())


def run_control_payload(strategy: StepStrategy) -> dict[str, Any]:
    """Stable run-manifest payload preserving the explicit controller choice."""
    if isinstance(strategy, AdaptiveCFL):
        return {
            "strategy": "adaptive_cfl",
            "cfl": _number(strategy.cfl),
            "max_dt": None if strategy.max_dt is None else _number(strategy.max_dt),
        }
    if isinstance(strategy, FixedDt):
        return {"strategy": "fixed_dt", "dt": None if strategy.dt is None else _number(strategy.dt)}
    if isinstance(strategy, ExternalTimeGrid):
        return {"strategy": "external_time_grid", "grid_id": strategy.grid_id}
    if isinstance(strategy, ErrorControlledDt):
        return {
            "strategy": "error_controlled_dt",
            "rtol": _number(strategy.rtol),
            "atol": _number(strategy.atol),
        }
    raise TypeError("unknown StepStrategy %r" % (strategy,))


__all__ = ["StepAttemptRejected", "resolve_run_strategy", "run_control_payload", "run_step_attempt"]
