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
    if isinstance(strategy, AdaptiveCFL):
        native.step_cfl(_number(strategy.cfl))
        return
    if isinstance(strategy, FixedDt):
        if strategy.dt is None:
            raise ValueError("FixedDt runtime execution requires FixedDt(dt=...)")
        native.step(_number(strategy.dt))
        return
    if isinstance(strategy, ExternalTimeGrid):
        raise NotImplementedError(
            "ExternalTimeGrid execution requires a runtime grid control wired through StepTransaction")
    if isinstance(strategy, ErrorControlledDt):
        raise NotImplementedError(
            "ErrorControlledDt requires an error estimator/controller; no runtime controller is wired yet")
    raise TypeError("unknown StepStrategy %r" % (strategy,))


def run_control_payload(strategy: StepStrategy) -> dict[str, Any]:
    """Stable run-manifest payload preserving the explicit controller choice."""
    if isinstance(strategy, AdaptiveCFL):
        return {"strategy": "adaptive_cfl", "cfl": _number(strategy.cfl)}
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
