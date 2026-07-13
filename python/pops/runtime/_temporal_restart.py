"""Strict Uniform temporal state persisted for exact next-attempt restart."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from pops._manifest_protocol import strict_json_loads


_SCHEMA_VERSION = 1
_STATUSES = frozenset(("accepted", "rejected", "failed"))
_EVENT_KEYS = frozenset(("kind", "time", "cursor", "payload"))


def _clock(time: Any, macro_step: Any) -> tuple[str, int]:
    value = float(time)
    if not math.isfinite(value):
        raise ValueError("temporal restart time must be finite")
    step = int(macro_step)
    if step < 0:
        raise ValueError("temporal restart macro_step must be >= 0")
    return value.hex(), step


def _positive_finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a numeric scalar" % where)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("%s must be finite and > 0" % where)
    return result


def _validate_strategy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("strategy"), str):
        raise TypeError("Uniform temporal restart strategy must be a run-control mapping")
    kind = value["strategy"]
    expected = {
        "adaptive_cfl": {"strategy", "cfl", "max_dt"},
        "fixed_dt": {"strategy", "dt"},
        "error_controlled_dt": {"strategy", "rtol", "atol"},
        "external_time_grid": {"strategy", "grid_id"},
    }
    if kind not in expected or set(value) != expected[kind]:
        raise ValueError("Uniform temporal restart strategy has unknown or incomplete keys")
    if kind == "adaptive_cfl":
        _positive_finite(value["cfl"], where="AdaptiveCFL.cfl")
        if value["max_dt"] is not None:
            _positive_finite(value["max_dt"], where="AdaptiveCFL.max_dt")
    elif kind == "fixed_dt":
        if value["dt"] is None:
            raise ValueError("checkpointed FixedDt requires an authored dt")
        _positive_finite(value["dt"], where="FixedDt.dt")
    elif kind == "error_controlled_dt":
        _positive_finite(value["rtol"], where="ErrorControlledDt.rtol")
        _positive_finite(value["atol"], where="ErrorControlledDt.atol")
    elif not isinstance(value["grid_id"], str) or not value["grid_id"]:
        raise ValueError("ExternalTimeGrid.grid_id must be a non-empty string")
    return dict(value)


def _validate_controller_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"last_accepted_dt"}:
        raise ValueError("Uniform temporal controller state has incomplete keys")
    last_dt = value["last_accepted_dt"]
    if last_dt is not None:
        if not isinstance(last_dt, str):
            raise TypeError("last_accepted_dt must be a hexadecimal float string or null")
        try:
            dt = float.fromhex(last_dt)
        except ValueError:
            raise ValueError("last_accepted_dt is not a hexadecimal float") from None
        if not math.isfinite(dt) or dt <= 0.0 or dt.hex() != last_dt:
            raise ValueError("last_accepted_dt must be a canonical finite positive hexadecimal float")
    return dict(value)


def _validate_event_queue(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Uniform temporal event queue must be a list")
    result = []
    for index, event in enumerate(value):
        if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
            raise ValueError("Uniform temporal event %d has incomplete keys" % index)
        if not isinstance(event["kind"], str) or not event["kind"]:
            raise ValueError("Uniform temporal event kind must be a non-empty string")
        if not isinstance(event["time"], str):
            raise ValueError("Uniform temporal event time must be a canonical hexadecimal float")
        try:
            event_value = float.fromhex(event["time"])
        except ValueError:
            raise ValueError("Uniform temporal event time is not hexadecimal") from None
        if not math.isfinite(event_value) or event_value.hex() != event["time"]:
            raise ValueError("Uniform temporal event time must be a canonical hexadecimal float")
        cursor = int(event["cursor"])
        if isinstance(event["cursor"], bool) or cursor != event["cursor"] or cursor < 0:
            raise ValueError("Uniform temporal event cursor must be a non-negative integer")
        if not isinstance(event["payload"], dict):
            raise TypeError("Uniform temporal event payload must be a mapping")
        result.append({"kind": event["kind"], "time": event["time"], "cursor": cursor,
                       "payload": dict(event["payload"])})
    return result


@dataclass
class TemporalRestartState:
    """Runtime-owned state needed to reproduce the next Uniform step attempt.

    Native fields, history rings and held values stay in their existing stores.  This
    object owns the controller/cursor envelope around them and records only committed
    attempt boundaries.  A rejected/failed attempt deliberately makes checkpointing
    ineligible until another attempt is accepted.
    """

    strategy: dict[str, Any] | None = None
    time_hex: str = float(0).hex()
    macro_step: int = 0
    schedule_cursors: dict[str, int] = field(default_factory=lambda: {"macro_step": 0})
    controller_state: dict[str, Any] = field(
        default_factory=lambda: {"last_accepted_dt": None})
    event_queue: list[dict[str, Any]] = field(default_factory=list)
    transaction_stats: dict[str, int] = field(
        default_factory=lambda: {"accepted": 0, "rejected": 0, "failed": 0})
    status: str = "accepted"
    synchronized: bool = True
    _restored_pending: bool = field(default=False, repr=False)

    def begin_run(self, strategy: dict[str, Any], *, time: Any, macro_step: Any) -> None:
        """Bind the controller for this run, enforcing the first post-restart attempt."""
        now, step = _clock(time, macro_step)
        if self.strategy is None and not self._restored_pending:
            self.time_hex = now
            self.macro_step = step
            self.schedule_cursors = {"macro_step": step}
        self._require_live_clock(now, step)
        candidate = _validate_strategy(strategy)
        if self._restored_pending and candidate != self.strategy:
            raise RuntimeError(
                "restart requires the checkpointed step strategy for the exact next attempt")
        if not self._restored_pending:
            self.strategy = candidate

    def before_attempt(self, *, time: Any, macro_step: Any) -> None:
        now, step = _clock(time, macro_step)
        self._require_live_clock(now, step)

    def accept(self, *, before_time: Any, before_step: Any,
               time: Any, macro_step: Any) -> None:
        before, old_step = _clock(before_time, before_step)
        now, step = _clock(time, macro_step)
        if step != old_step + 1 or float.fromhex(now) <= float.fromhex(before):
            raise RuntimeError(
                "accepted Uniform attempt must advance time and macro_step exactly once")
        self.time_hex = now
        self.macro_step = step
        self.schedule_cursors = {"macro_step": step}
        self.controller_state = {
            "last_accepted_dt": (float.fromhex(now) - float.fromhex(before)).hex(),
        }
        self.transaction_stats["accepted"] += 1
        self.status = "accepted"
        self.synchronized = True
        self._restored_pending = False

    def reject(self, *, time: Any, macro_step: Any) -> None:
        self._record_unsuccessful("rejected", time=time, macro_step=macro_step)

    def fail(self, *, time: Any, macro_step: Any) -> None:
        self._record_unsuccessful("failed", time=time, macro_step=macro_step)

    def checkpoint_json(self, *, time: Any, macro_step: Any) -> str:
        """Return the strict payload, refusing an uncommitted or unsynchronized point."""
        now, step = _clock(time, macro_step)
        self._require_live_clock(now, step)
        if self.status != "accepted" or not self.synchronized:
            raise RuntimeError(
                "checkpoint requires an accepted synchronized step boundary; "
                "the last attempt was %s" % self.status)
        if self.strategy is None:
            raise RuntimeError("checkpoint requires a declared step strategy")
        _validate_strategy(self.strategy)
        _validate_controller_state(self.controller_state)
        _validate_event_queue(self.event_queue)
        if self.schedule_cursors != {"macro_step": step}:
            raise RuntimeError("checkpoint schedule cursor is not synchronized with macro_step")
        if (set(self.transaction_stats) != _STATUSES
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in self.transaction_stats.values())):
            raise RuntimeError("checkpoint transaction statistics are invalid")
        return json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "strategy": self.strategy,
            "clock": {"time": self.time_hex, "macro_step": self.macro_step},
            "schedule_cursors": dict(self.schedule_cursors),
            "controller_state": dict(self.controller_state),
            "event_queue": list(self.event_queue),
            "transaction_stats": dict(self.transaction_stats),
            "status": self.status,
            "synchronized": self.synchronized,
        }

    @classmethod
    def from_json(cls, payload: Any, *, time: Any, macro_step: Any) -> TemporalRestartState:
        data = strict_json_loads(str(payload), where="Uniform temporal restart state")
        expected = {
            "schema_version", "strategy", "clock", "schedule_cursors", "controller_state",
            "event_queue", "transaction_stats", "status", "synchronized",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Uniform temporal restart state has an incomplete strict manifest")
        if (isinstance(data["schema_version"], bool)
                or not isinstance(data["schema_version"], int)
                or data["schema_version"] != _SCHEMA_VERSION):
            raise ValueError("unsupported Uniform temporal restart schema_version")
        strategy = _validate_strategy(data["strategy"])
        clock = data["clock"]
        if not isinstance(clock, dict) or set(clock) != {"time", "macro_step"}:
            raise ValueError("Uniform temporal restart clock is incomplete")
        now, step = _clock(time, macro_step)
        if (not isinstance(clock["time"], str)
                or isinstance(clock["macro_step"], bool)
                or not isinstance(clock["macro_step"], int)):
            raise TypeError("Uniform temporal restart clock has invalid field types")
        if clock != {"time": now, "macro_step": step}:
            raise ValueError("Uniform temporal restart clock differs from the checkpoint clock")
        cursors = data["schedule_cursors"]
        if (not isinstance(cursors, dict) or set(cursors) != {"macro_step"}
                or isinstance(cursors["macro_step"], bool)
                or not isinstance(cursors["macro_step"], int)
                or cursors["macro_step"] != step):
            raise ValueError("Uniform schedule cursor is not synchronized with macro_step")
        stats = data["transaction_stats"]
        if (not isinstance(stats, dict) or set(stats) != _STATUSES
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in stats.values())):
            raise ValueError("Uniform temporal transaction statistics are invalid")
        if data["status"] != "accepted" or data["synchronized"] is not True:
            raise ValueError("checkpoint temporal state is not an accepted synchronized point")
        controller = _validate_controller_state(data["controller_state"])
        events = _validate_event_queue(data["event_queue"])
        out = cls(
            strategy=strategy, time_hex=now, macro_step=step,
            schedule_cursors=dict(cursors), controller_state=controller,
            event_queue=events, transaction_stats=dict(stats),
            status="accepted", synchronized=True,
        )
        out._restored_pending = True
        return out

    def _record_unsuccessful(self, status: str, *, time: Any, macro_step: Any) -> None:
        now, step = _clock(time, macro_step)
        self._require_live_clock(now, step)
        self.transaction_stats[status] += 1
        self.status = status
        self.synchronized = False
        self._restored_pending = False

    def _require_live_clock(self, time_hex: str, macro_step: int) -> None:
        if time_hex != self.time_hex or macro_step != self.macro_step:
            raise RuntimeError(
                "Uniform temporal state is desynchronized from the native runtime clock")


def is_rejected_attempt(error: BaseException) -> bool:
    """Recognize only the stable exception type exported by the native transaction layer."""
    from pops._bootstrap import StepAttemptRejected
    return isinstance(error, StepAttemptRejected)


__all__ = ["TemporalRestartState", "is_rejected_attempt"]
