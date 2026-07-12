"""Explicit consumed outcome contract for Program solve nodes."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from pops.time.program_transaction import authoring_transaction
from pops.time.residual_common import CanonicalDescriptor, residual_names

SOLVE_STATUSES = (
    "singular", "breakdown", "iteration_limit",
    "invalid_evaluation", "capability_failure", "invalid_input",
)
_STATUS_SET = frozenset(SOLVE_STATUSES)


def _statuses(values: Iterable[Any]) -> tuple[str, ...]:
    statuses = residual_names(values, "SolveAction statuses", nonempty=True)
    unknown = tuple(status for status in statuses if status not in _STATUS_SET)
    if unknown:
        raise ValueError("unknown solve status(es): %s" % ", ".join(unknown))
    return statuses


@dataclass(frozen=True, slots=True)
class SolveAction(CanonicalDescriptor):
    """Base class for explicit runtime disposition of a non-solved solve result."""

    statuses: tuple[str, ...] = SOLVE_STATUSES
    kind: ClassVar[str] = "solve_action"

    def __post_init__(self) -> None:
        object.__setattr__(self, "statuses", _statuses(self.statuses))


@dataclass(frozen=True, slots=True)
class FailRun(SolveAction):
    """Abort the run when a solve reports any configured non-solved status."""

    kind: ClassVar[str] = "fail_run"


@dataclass(frozen=True, slots=True)
class RejectAttempt(SolveAction):
    """Reject the current attempt/step when a solve reports a configured status."""

    kind: ClassVar[str] = "reject_attempt"


@dataclass(frozen=True, slots=True)
class ResidualSolution:
    """Ordered, typed projections of one consumed residual solve outcome."""

    values: tuple[Any, ...]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        return self.values[index]


class SolveOutcome:
    """Non-readable Program solve token; call ``consume(action=...)`` to project values."""

    __slots__ = ("_program", "_token", "_factory", "_name", "_result")

    def __init__(self, program: Any, token: Any, factory: Any, name: Any) -> None:
        self._program = program
        self._token = token
        self._factory = factory
        self._name = name
        self._result = None

    @property
    def token(self) -> Any:
        raise TypeError(
            "SolveOutcome is not readable; call outcome.consume(action=FailRun(...) or "
            "RejectAttempt(...)) before using solved values")

    def consume(self, *, action: SolveAction) -> Any:
        if not isinstance(action, SolveAction):
            raise TypeError("SolveOutcome.consume requires action=FailRun(...) or RejectAttempt(...)")
        if self._result is not None:
            raise RuntimeError("SolveOutcome has already been consumed")
        with authoring_transaction(self._program):
            node = self._program._new(
                "solve_outcome", "solve_outcome", (self._token,),
                {"action": action}, "%s_outcome" % self._name, self._token.block,
                space=self._token.space, point=self._token.point)
            result = self._factory(node)
        self._result = result
        return result

    def __iter__(self):
        raise TypeError("SolveOutcome is not iterable; call consume(action=...) first")

    def __len__(self) -> int:
        raise TypeError("SolveOutcome has no length; call consume(action=...) first")

    def __getitem__(self, _index: int) -> Any:
        raise TypeError("SolveOutcome is not indexable; call consume(action=...) first")

    def __repr__(self) -> str:
        return "<SolveOutcome %s: unconsumed>" % self._name


__all__ = [
    "FailRun", "RejectAttempt", "ResidualSolution", "SOLVE_STATUSES",
    "SolveAction", "SolveOutcome",
]
