"""Strict accepted-boundary temporal state for exact next-attempt restart.

The native checkpoint owns field values, history buffers and held cache values.  This module owns
the data-only temporal envelope around them: the installed logical-clock schedule, accepted clock
and schedule cursors, controller state and event queue.  Only an accepted, fully synchronized
boundary is serializable; an older schema is an offline-migration input, never a runtime fallback.
"""
from __future__ import annotations

import json
import math
import operator
from dataclasses import dataclass, field
from typing import Any

from pops._manifest_protocol import strict_json_loads
from pops.time._step.strategy import validate_step_strategy_manifest


_SCHEMA_VERSION = 2
_STATUSES = frozenset(("accepted", "rejected", "failed"))
_EVENT_KEYS = frozenset(("kind", "time", "cursor", "payload"))
_PROGRAM_KEYS = frozenset((
    "schema_version", "kind", "primary_clock", "clocks", "subcycles",
    "synchronizations", "schedules", "histories",
))
_UNSPECIFIED = object()


def _clock(time: Any, macro_step: Any) -> tuple[str, int]:
    value = float(time)
    if not math.isfinite(value):
        raise ValueError("temporal restart time must be finite")
    try:
        step = operator.index(macro_step)
    except TypeError:
        raise ValueError("temporal restart macro_step must be a non-negative integer") from None
    if isinstance(macro_step, bool) or step < 0:
        raise ValueError("temporal restart macro_step must be a non-negative integer")
    return value.hex(), step


def _json_copy(value: Any, *, where: str) -> Any:
    """Detach one manifest through the exact JSON value model and reject NaN/opaque values."""
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError("%s must contain canonical JSON values" % where) from error
    return strict_json_loads(payload, where=where)


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % where)
    return value


def _node_id(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % where)
    return value


def _clock_identity(descriptor: Any, *, where: str) -> str:
    from pops.model.ownership import OwnerPath
    from pops.time.points import Clock

    if not isinstance(descriptor, dict) or set(descriptor) != {
            "schema_version", "name", "owner"}:
        raise ValueError("%s has an incomplete clock descriptor" % where)
    if descriptor["schema_version"] != 1:
        raise ValueError("%s has an unsupported clock schema_version" % where)
    if not isinstance(descriptor["name"], str) or not descriptor["name"]:
        raise ValueError("%s clock name must be non-empty text" % where)
    owner = None if descriptor["owner"] is None else OwnerPath.from_data(descriptor["owner"])
    clock = Clock(descriptor["name"], owner=owner)
    if clock.to_data() != descriptor:
        raise ValueError("%s clock descriptor is not canonical" % where)
    return clock.qualified_id


def _schedule_clock_identity(schedule: Any, *, where: str) -> str:
    if not isinstance(schedule, dict) or set(schedule) != {
            "schema_version", "domain", "trigger", "off"}:
        raise ValueError("%s has an incomplete typed schedule" % where)
    if schedule["schema_version"] != 1 or not isinstance(schedule["domain"], dict):
        raise ValueError("%s has an unsupported typed schedule" % where)
    domain = schedule["domain"]
    required = {"type", "clock", "at"}
    if not required <= set(domain) or not isinstance(domain["type"], str):
        raise ValueError("%s schedule domain is incomplete" % where)
    return _clock_identity(domain["clock"], where=where + ".domain")


def _validate_program_schedule(value: Any) -> dict[str, Any]:
    """Validate the generic temporal-program protocol and its clock topology."""
    data = _json_copy(value, where="temporal program schedule")
    if not isinstance(data, dict) or set(data) != _PROGRAM_KEYS:
        raise ValueError("temporal program schedule has an incomplete strict manifest")
    if data["schema_version"] != 1 or data["kind"] != "pops.temporal-program-schedule":
        raise ValueError("unsupported temporal program schedule schema or kind")
    if not isinstance(data["primary_clock"], str) or not data["primary_clock"]:
        raise ValueError("temporal program schedule primary_clock must be qualified text")

    if not isinstance(data["clocks"], list) or not data["clocks"]:
        raise ValueError("temporal program schedule must declare at least one clock")
    clocks: dict[str, int] = {}
    for index, row in enumerate(data["clocks"]):
        if not isinstance(row, dict) or set(row) != {"id", "descriptor", "ticks_per_macro"}:
            raise ValueError("temporal program clock %d has incomplete keys" % index)
        identity = _clock_identity(row["descriptor"], where="temporal clock %d" % index)
        if row["id"] != identity:
            raise ValueError("temporal program clock %d identity is not canonical" % index)
        if identity in clocks:
            raise ValueError("temporal program schedule declares clock %s twice" % identity)
        clocks[identity] = _positive_int(
            row["ticks_per_macro"], where="temporal clock ticks_per_macro")
    primary = data["primary_clock"]
    if clocks.get(primary) != 1:
        raise ValueError("primary temporal clock must exist with ticks_per_macro=1")

    if not isinstance(data["subcycles"], list):
        raise TypeError("temporal program subcycles must be a list")
    child_relations: dict[str, tuple[str, int]] = {}
    subcycle_nodes: set[int] = set()
    for index, row in enumerate(data["subcycles"]):
        if not isinstance(row, dict) or set(row) != {
                "node_id", "parent_clock", "child_clock", "count"}:
            raise ValueError("temporal subcycle %d has incomplete keys" % index)
        node = _node_id(row["node_id"], where="temporal subcycle node_id")
        if node in subcycle_nodes:
            raise ValueError("temporal subcycle node_id %d is duplicated" % node)
        subcycle_nodes.add(node)
        parent, child = row["parent_clock"], row["child_clock"]
        count = _positive_int(row["count"], where="temporal subcycle count")
        if parent not in clocks or child not in clocks or parent == child:
            raise ValueError("temporal subcycle references invalid qualified clocks")
        relation = (parent, count)
        if child in child_relations and child_relations[child] != relation:
            raise ValueError("temporal child clock has conflicting subcycle parents")
        child_relations[child] = relation
        if clocks[child] != clocks[parent] * count:
            raise ValueError("temporal subcycle ticks_per_macro is inconsistent with count")
    if set(clocks) - {primary} != set(child_relations):
        raise ValueError("every non-primary temporal clock requires one subcycle parent")

    if not isinstance(data["synchronizations"], list):
        raise TypeError("temporal program synchronizations must be a list")
    sync_nodes: set[int] = set()
    for index, row in enumerate(data["synchronizations"]):
        if not isinstance(row, dict) or set(row) != {
                "node_id", "source_clock", "target_clock", "relation", "point"}:
            raise ValueError("temporal synchronization %d has incomplete keys" % index)
        node = _node_id(row["node_id"], where="temporal synchronization node_id")
        if node in sync_nodes:
            raise ValueError("temporal synchronization node_id %d is duplicated" % node)
        sync_nodes.add(node)
        if (row["source_clock"] not in clocks or row["target_clock"] not in clocks
                or row["source_clock"] == row["target_clock"]):
            raise ValueError("temporal synchronization references invalid qualified clocks")
        relation = row["relation"]
        if not isinstance(relation, dict) or not isinstance(relation.get("kind"), str) \
                or not relation["kind"]:
            raise ValueError("temporal synchronization relation is not a typed provider")
        if not isinstance(row["point"], dict):
            raise TypeError("temporal synchronization point must be typed data")

    if not isinstance(data["schedules"], list):
        raise TypeError("temporal program schedules must be a list")
    schedule_nodes: set[int] = set()
    for index, row in enumerate(data["schedules"]):
        if not isinstance(row, dict) or set(row) != {
                "node_id", "schedule", "cache_required"}:
            raise ValueError("temporal schedule %d has incomplete keys" % index)
        node = _node_id(row["node_id"], where="temporal schedule node_id")
        if node in schedule_nodes:
            raise ValueError("temporal schedule node_id %d is duplicated" % node)
        schedule_nodes.add(node)
        clock_id = _schedule_clock_identity(
            row["schedule"], where="temporal schedule %d" % index)
        if clock_id not in clocks:
            raise ValueError("temporal schedule references an undeclared clock")
        if type(row["cache_required"]) is not bool:
            raise TypeError("temporal schedule cache_required must be bool")

    if not isinstance(data["histories"], list):
        raise TypeError("temporal program histories must be a list")
    history_names: set[str] = set()
    for index, row in enumerate(data["histories"]):
        expected = {
            "name", "owner", "state", "space", "clock", "depth", "ring_slots", "ncomp",
            "validity", "interpolation", "checkpoint_policy",
        }
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("temporal history %d has incomplete keys" % index)
        name = row["name"]
        if not isinstance(name, str) or not name or name in history_names:
            raise ValueError("temporal history names must be unique non-empty text")
        history_names.add(name)
        if row["clock"] not in clocks:
            raise ValueError("temporal history %r references an undeclared clock" % name)
        depth = _positive_int(row["depth"], where="temporal history depth")
        if row["ring_slots"] != depth + 1:
            raise ValueError("temporal history ring_slots must equal depth + 1")
        ncomp = row["ncomp"]
        if ncomp is not None:
            _positive_int(ncomp, where="temporal history ncomp")
        if not isinstance(row["state"], dict) or not isinstance(row["space"], dict):
            raise TypeError("temporal history state/space must be typed data")
        if row["owner"] is not None and not isinstance(row["owner"], dict):
            raise TypeError("temporal history owner must be typed data or null")
        if not isinstance(row["validity"], dict) or not isinstance(row["interpolation"], dict):
            raise TypeError("temporal history validity/interpolation must be typed data")
        if row["checkpoint_policy"] is not None \
                and not isinstance(row["checkpoint_policy"], dict):
            raise TypeError("temporal history checkpoint policy must be typed data or null")
    return data


def _validate_controller_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"last_accepted_dt"}:
        raise ValueError("temporal controller state has incomplete keys")
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
        raise TypeError("temporal event queue must be a list")
    result = []
    for index, event in enumerate(value):
        if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
            raise ValueError("temporal event %d has incomplete keys" % index)
        if not isinstance(event["kind"], str) or not event["kind"]:
            raise ValueError("temporal event kind must be a non-empty string")
        if not isinstance(event["time"], str):
            raise ValueError("temporal event time must be a canonical hexadecimal float")
        try:
            event_value = float.fromhex(event["time"])
        except ValueError:
            raise ValueError("temporal event time is not hexadecimal") from None
        if not math.isfinite(event_value) or event_value.hex() != event["time"]:
            raise ValueError("temporal event time must be a canonical hexadecimal float")
        cursor = event["cursor"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("temporal event cursor must be a non-negative integer")
        if not isinstance(event["payload"], dict):
            raise TypeError("temporal event payload must be a mapping")
        result.append({
            "kind": event["kind"], "time": event["time"], "cursor": cursor,
            "payload": _json_copy(event["payload"], where="temporal event payload"),
        })
    return result


def _boundary_cursors(
    schedule: dict[str, Any] | None, *, time_hex: str, macro_step: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive every accepted-boundary cursor from the immutable program schedule."""
    schedule_cursors: dict[str, Any] = {
        "macro_step": {"macro_step": macro_step, "phase": "accepted"},
    }
    if schedule is None:
        return {}, schedule_cursors, {}, {}, {}
    clock_ticks = {
        row["id"]: macro_step * row["ticks_per_macro"] for row in schedule["clocks"]
    }
    clock_cursors = {
        identity: {"time": time_hex, "tick": tick, "phase": "accepted"}
        for identity, tick in clock_ticks.items()
    }
    for row in schedule["subcycles"]:
        schedule_cursors["subcycle:%d" % row["node_id"]] = {
            "macro_step": macro_step, "next_iteration": 0, "phase": "accepted",
        }
    for row in schedule["schedules"]:
        clock_id = _schedule_clock_identity(
            row["schedule"], where="temporal schedule cursor")
        schedule_cursors["schedule:%d" % row["node_id"]] = {
            "macro_step": macro_step, "clock_tick": clock_ticks[clock_id],
            "phase": "accepted",
        }
    synchronization_cursors = {
        str(row["node_id"]): {
            "macro_step": macro_step,
            "source_tick": clock_ticks[row["source_clock"]],
            "target_tick": clock_ticks[row["target_clock"]],
            "phase": "accepted",
        }
        for row in schedule["synchronizations"]
    }
    history_cursors = {
        row["name"]: {
            "clock": row["clock"], "newest_tick": clock_ticks[row["clock"]],
            "oldest_tick": max(0, clock_ticks[row["clock"]] - row["depth"]),
            "valid_lags": row["depth"] if macro_step > 0 else 0,
            "cold_start_extended": (
                macro_step > 0 and clock_ticks[row["clock"]] < row["depth"]),
            "initialized": macro_step > 0,
        }
        for row in schedule["histories"]
    }
    cache_cursors = {}
    for row in schedule["schedules"]:
        if not row["cache_required"]:
            continue
        clock_id = _schedule_clock_identity(row["schedule"], where="temporal cache cursor")
        cache_cursors[str(row["node_id"])] = {
            "clock": clock_id, "valid_through_tick": clock_ticks[clock_id],
            "initialized": macro_step > 0,
        }
    return (
        clock_cursors, schedule_cursors, synchronization_cursors,
        history_cursors, cache_cursors,
    )


@dataclass
class TemporalRestartState:
    """Runtime-owned temporal envelope committed only at accepted synchronized boundaries."""

    strategy: dict[str, Any] | None = None
    program_schedule: dict[str, Any] | None = None
    time_hex: str = float(0).hex()
    macro_step: int = 0
    clock_cursors: dict[str, Any] = field(default_factory=dict)
    schedule_cursors: dict[str, Any] = field(default_factory=lambda: {
        "macro_step": {"macro_step": 0, "phase": "accepted"},
    })
    synchronization_cursors: dict[str, Any] = field(default_factory=dict)
    history_cursors: dict[str, Any] = field(default_factory=dict)
    cache_cursors: dict[str, Any] = field(default_factory=dict)
    controller_state: dict[str, Any] = field(
        default_factory=lambda: {"last_accepted_dt": None})
    event_queue: list[dict[str, Any]] = field(default_factory=list)
    transaction_stats: dict[str, int] = field(
        default_factory=lambda: {"accepted": 0, "rejected": 0, "failed": 0})
    status: str = "accepted"
    synchronized: bool = True
    _restored_pending: bool = field(default=False, repr=False)

    def configure_program(self, schedule: Any, *, time: Any, macro_step: Any) -> None:
        """Bind one immutable nested-clock contract before execution or restart."""
        now, step = _clock(time, macro_step)
        self._require_live_clock(now, step)
        candidate = _validate_program_schedule(schedule)
        if self.program_schedule is not None and candidate != self.program_schedule:
            raise RuntimeError("installed temporal program schedule differs from restored schedule")
        self.program_schedule = candidate
        cursors = _boundary_cursors(candidate, time_hex=now, macro_step=step)
        if self._restored_pending:
            existing = (
                self.clock_cursors, self.schedule_cursors, self.synchronization_cursors,
                self.history_cursors, self.cache_cursors,
            )
            if existing != cursors:
                raise RuntimeError("restored temporal cursors differ from the installed schedule")
        else:
            (self.clock_cursors, self.schedule_cursors, self.synchronization_cursors,
             self.history_cursors, self.cache_cursors) = cursors

    def begin_run(self, strategy: dict[str, Any], *, time: Any, macro_step: Any) -> None:
        """Bind the controller for this run, enforcing the first post-restart attempt."""
        now, step = _clock(time, macro_step)
        if self.strategy is None and not self._restored_pending:
            self.time_hex = now
            self.macro_step = step
            (self.clock_cursors, self.schedule_cursors, self.synchronization_cursors,
             self.history_cursors, self.cache_cursors) = _boundary_cursors(
                 self.program_schedule, time_hex=now, macro_step=step)
        self._require_live_clock(now, step)
        candidate = validate_step_strategy_manifest(strategy)
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
                "accepted temporal attempt must advance time and macro_step exactly once")
        self.time_hex = now
        self.macro_step = step
        (self.clock_cursors, self.schedule_cursors, self.synchronization_cursors,
         self.history_cursors, self.cache_cursors) = _boundary_cursors(
             self.program_schedule, time_hex=now, macro_step=step)
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

    def cursor_for_clock(self, clock: Any) -> dict[str, Any]:
        """Return one accepted qualified cursor; never infer it from the macro clock."""
        from pops.time.points import Clock

        if type(clock) is not Clock:
            raise TypeError("temporal cursor lookup requires an exact Clock")
        if self.status != "accepted" or not self.synchronized:
            raise RuntimeError("temporal cursor lookup requires an accepted synchronized boundary")
        identity = clock.qualified_id
        try:
            cursor = self.clock_cursors[identity]
        except KeyError:
            raise RuntimeError(
                "accepted temporal state has no cursor for qualified clock %s" % identity) from None
        if cursor.get("phase") != "accepted" or cursor.get("time") != self.time_hex:
            raise RuntimeError("qualified temporal clock cursor is not at the accepted boundary")
        return dict(cursor)

    def checkpoint_json(self, *, time: Any, macro_step: Any) -> str:
        """Return schema v2, refusing an uncommitted or unsynchronized point."""
        now, step = _clock(time, macro_step)
        self._require_live_clock(now, step)
        if self.status != "accepted" or not self.synchronized:
            raise RuntimeError(
                "checkpoint requires an accepted synchronized step boundary; "
                "the last attempt was %s" % self.status)
        if self.strategy is None:
            raise RuntimeError("checkpoint requires a declared step strategy")
        validate_step_strategy_manifest(self.strategy)
        if self.program_schedule is not None:
            _validate_program_schedule(self.program_schedule)
        expected = _boundary_cursors(self.program_schedule, time_hex=now, macro_step=step)
        actual = (
            self.clock_cursors, self.schedule_cursors, self.synchronization_cursors,
            self.history_cursors, self.cache_cursors,
        )
        if actual != expected:
            raise RuntimeError("checkpoint temporal cursors are not at one synchronized boundary")
        _validate_controller_state(self.controller_state)
        _validate_event_queue(self.event_queue)
        if (set(self.transaction_stats) != _STATUSES
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in self.transaction_stats.values())):
            raise RuntimeError("checkpoint transaction statistics are invalid")
        return json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "strategy": self.strategy,
            "program_schedule": self.program_schedule,
            "clock": {"time": self.time_hex, "macro_step": self.macro_step},
            "clock_cursors": _json_copy(self.clock_cursors, where="clock cursors"),
            "schedule_cursors": _json_copy(self.schedule_cursors, where="schedule cursors"),
            "synchronization_cursors": _json_copy(
                self.synchronization_cursors, where="synchronization cursors"),
            "history_cursors": _json_copy(self.history_cursors, where="history cursors"),
            "cache_cursors": _json_copy(self.cache_cursors, where="cache cursors"),
            "controller_state": dict(self.controller_state),
            "event_queue": _json_copy(self.event_queue, where="event queue"),
            "transaction_stats": dict(self.transaction_stats),
            "status": self.status,
            "synchronized": self.synchronized,
        }

    @classmethod
    def from_json(
        cls, payload: Any, *, time: Any, macro_step: Any,
        program_schedule: Any = _UNSPECIFIED,
    ) -> TemporalRestartState:
        data = strict_json_loads(str(payload), where="temporal restart state")
        expected = {
            "schema_version", "strategy", "program_schedule", "clock", "clock_cursors",
            "schedule_cursors", "synchronization_cursors", "history_cursors", "cache_cursors",
            "controller_state", "event_queue", "transaction_stats", "status", "synchronized",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("temporal restart state has an incomplete strict manifest")
        if (isinstance(data["schema_version"], bool)
                or not isinstance(data["schema_version"], int)
                or data["schema_version"] != _SCHEMA_VERSION):
            raise ValueError(
                "unsupported temporal restart schema_version; historical payloads require "
                "offline migration")
        strategy = validate_step_strategy_manifest(data["strategy"])
        schedule = (None if data["program_schedule"] is None
                    else _validate_program_schedule(data["program_schedule"]))
        if program_schedule is not _UNSPECIFIED:
            installed = (None if program_schedule is None
                         else _validate_program_schedule(program_schedule))
            if schedule != installed:
                raise ValueError(
                    "checkpoint temporal program schedule differs from installed program")
        clock = data["clock"]
        if not isinstance(clock, dict) or set(clock) != {"time", "macro_step"}:
            raise ValueError("temporal restart clock is incomplete")
        now, step = _clock(time, macro_step)
        if (not isinstance(clock["time"], str)
                or isinstance(clock["macro_step"], bool)
                or not isinstance(clock["macro_step"], int)):
            raise TypeError("temporal restart clock has invalid field types")
        if clock != {"time": now, "macro_step": step}:
            raise ValueError("temporal restart clock differs from the checkpoint clock")
        cursor_sections = (
            data["clock_cursors"], data["schedule_cursors"],
            data["synchronization_cursors"], data["history_cursors"], data["cache_cursors"],
        )
        if any(not isinstance(section, dict) for section in cursor_sections):
            raise TypeError("temporal cursor sections must be mappings")
        if cursor_sections != _boundary_cursors(schedule, time_hex=now, macro_step=step):
            raise ValueError("temporal restart cursors are not at one synchronized boundary")
        stats = data["transaction_stats"]
        if (not isinstance(stats, dict) or set(stats) != _STATUSES
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in stats.values())):
            raise ValueError("temporal transaction statistics are invalid")
        if data["status"] != "accepted" or data["synchronized"] is not True:
            raise ValueError("checkpoint temporal state is not an accepted synchronized point")
        controller = _validate_controller_state(data["controller_state"])
        events = _validate_event_queue(data["event_queue"])
        out = cls(
            strategy=strategy, program_schedule=schedule, time_hex=now, macro_step=step,
            clock_cursors=dict(data["clock_cursors"]),
            schedule_cursors=dict(data["schedule_cursors"]),
            synchronization_cursors=dict(data["synchronization_cursors"]),
            history_cursors=dict(data["history_cursors"]),
            cache_cursors=dict(data["cache_cursors"]),
            controller_state=controller, event_queue=events,
            transaction_stats=dict(stats), status="accepted", synchronized=True,
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
                "temporal state is desynchronized from the native runtime clock")


def is_rejected_attempt(error: BaseException) -> bool:
    """Recognize only the stable exception type exported by the native transaction layer."""
    from pops._bootstrap import StepAttemptRejected
    return isinstance(error, StepAttemptRejected)


__all__ = ["TemporalRestartState", "is_rejected_attempt"]
