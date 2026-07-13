"""Strict backend-neutral IR for native schedule extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ScheduleTimeline(Enum):
    """Native timeline primitives understood by the current runtime."""

    ACCEPTED_STEP = "accepted_step"


class ScheduleDueKind(Enum):
    """Closed native due-test primitives composable by schedule extensions."""

    ALWAYS = "always"
    CACHE_PERIOD = "cache_period"
    MACRO_STEP_ZERO = "macro_step_zero"
    AT_END = "at_end"
    PROGRAM_PREDICATE = "program_predicate"


class ScheduleAction(Enum):
    """Closed effects available to an off-cadence schedule policy."""

    EFFECTIVE_DT = "effective_dt"
    STORE = "store"
    ZERO = "zero"
    ACCUMULATE_DT = "accumulate_dt"
    RESTORE = "restore"
    ERROR = "error"


class ScheduleComment(Enum):
    """Validated annotations emitted by a schedule policy."""

    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class ScheduleDomainIR:
    """Strict native projection of a schedule domain."""

    timeline: ScheduleTimeline
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.timeline) is not ScheduleTimeline:
            raise TypeError("ScheduleDomainIR timeline must be an exact ScheduleTimeline")


@dataclass(frozen=True, slots=True)
class ScheduleDueIR:
    """Strict due-test IR returned by a native Trigger implementation."""

    kind: ScheduleDueKind
    period: int | None = None
    predicate: Any = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.kind) is not ScheduleDueKind:
            raise TypeError("ScheduleDueIR kind must be an exact ScheduleDueKind")
        if self.kind is ScheduleDueKind.CACHE_PERIOD:
            if (
                isinstance(self.period, bool)
                or not isinstance(self.period, int)
                or self.period <= 0
            ):
                raise ValueError("CACHE_PERIOD requires a positive integer period")
            if self.predicate is not None:
                raise ValueError("CACHE_PERIOD does not accept a predicate")
            return
        if self.period is not None:
            raise ValueError("%s does not accept a period" % self.kind.value)
        if self.kind is ScheduleDueKind.PROGRAM_PREDICATE:
            if self.predicate is None:
                raise ValueError("PROGRAM_PREDICATE requires a predicate")
        elif self.predicate is not None:
            raise ValueError("%s does not accept a predicate" % self.kind.value)


@dataclass(frozen=True, slots=True)
class ScheduleOffIR:
    """Strict, backend-neutral action plan for due and off-cadence branches."""

    before_due: tuple[ScheduleAction, ...] = ()
    after_due: tuple[ScheduleAction, ...] = ()
    off_cadence: tuple[ScheduleAction, ...] = ()
    comment: ScheduleComment | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        action_groups = (
            ("before_due", self.before_due),
            ("after_due", self.after_due),
            ("off_cadence", self.off_cadence),
        )
        for where, actions in action_groups:
            if type(actions) is not tuple:
                raise TypeError("ScheduleOffIR %s must be an exact tuple" % where)
            if any(type(action) is not ScheduleAction for action in actions):
                raise TypeError(
                    "ScheduleOffIR %s entries must be exact ScheduleAction values" % where
                )
        if self.comment is not None and type(self.comment) is not ScheduleComment:
            raise TypeError("ScheduleOffIR comment must be an exact ScheduleComment or None")
        allowed = {
            "before_due": frozenset({ScheduleAction.EFFECTIVE_DT}),
            "after_due": frozenset({ScheduleAction.STORE}),
            "off_cadence": frozenset(
                {
                    ScheduleAction.ZERO,
                    ScheduleAction.ACCUMULATE_DT,
                    ScheduleAction.RESTORE,
                    ScheduleAction.ERROR,
                }
            ),
        }
        for where, actions in action_groups:
            if len(actions) != len(set(actions)):
                raise ValueError("ScheduleOffIR %s must not contain duplicate actions" % where)
            if not set(actions).issubset(allowed[where]):
                raise ValueError("ScheduleOffIR contains an action invalid in %s" % where)
        if self.comment is ScheduleComment.SKIP and any(actions for _, actions in action_groups):
            raise ValueError("the SKIP annotation requires an empty action plan")
        if ScheduleAction.ERROR in self.off_cadence and len(self.off_cadence) != 1:
            raise ValueError("ERROR must be the only off-cadence action")
        if ScheduleAction.ZERO in self.off_cadence and len(self.off_cadence) != 1:
            raise ValueError("ZERO must be the only off-cadence action")
        accumulates = ScheduleAction.ACCUMULATE_DT in self.off_cadence
        effective = ScheduleAction.EFFECTIVE_DT in self.before_due
        if accumulates != effective:
            raise ValueError("ACCUMULATE_DT and EFFECTIVE_DT must be used together")


@dataclass(frozen=True, slots=True)
class ScheduleLoweringIR:
    """Exact return contract implemented by schedules lowerable by the native backend."""

    domain: ScheduleDomainIR
    due: ScheduleDueIR
    off: ScheduleOffIR
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.domain) is not ScheduleDomainIR:
            raise TypeError("ScheduleLoweringIR domain must be an exact ScheduleDomainIR")
        if type(self.due) is not ScheduleDueIR:
            raise TypeError("ScheduleLoweringIR due must be an exact ScheduleDueIR")
        if type(self.off) is not ScheduleOffIR:
            raise TypeError("ScheduleLoweringIR off must be an exact ScheduleOffIR")
        if self.due.kind is ScheduleDueKind.ALWAYS and (
            self.off.before_due
            or self.off.after_due
            or self.off.off_cadence
            or self.off.comment is not None
        ):
            raise ValueError("an ALWAYS due-test cannot have an off-cadence action plan")


__all__ = [
    "ScheduleTimeline",
    "ScheduleDueKind",
    "ScheduleAction",
    "ScheduleComment",
    "ScheduleDomainIR",
    "ScheduleDueIR",
    "ScheduleOffIR",
    "ScheduleLoweringIR",
]
