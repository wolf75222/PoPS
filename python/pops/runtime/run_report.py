"""Immutable public evidence returned by :func:`pops.run`."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pops._frozen_data import freeze_data, thaw_data
from pops.identity import Identity


class RunStopReason(str, Enum):
    """Successful terminal conditions implemented by the public run transition."""

    TARGET_TIME_REACHED = "target_time_reached"


def _count(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % where)
    if value < 0:
        raise ValueError("%s must be non-negative" % where)
    return value


def _identity(value: Any, *, domain: str, where: str) -> Identity:
    if type(value) is not Identity or value.domain != domain:
        raise TypeError("%s must be a domain-%r Identity" % (where, domain))
    return Identity.from_data(value.to_data())


@dataclass(frozen=True, slots=True)
class RunReport:
    """Observed result of one successful public ``pops.run`` invocation.

    Counts are local to this invocation. ``final_macro_step`` is the cumulative native clock after
    the invocation, while ``rejected_steps`` counts rejected native attempts retried before an
    accepted macro-step. A failed run raises and therefore never produces a successful report.
    """

    accepted_steps: int
    rejected_steps: int
    final_time: float
    final_macro_step: int
    stop_reason: RunStopReason
    run_identity: Identity
    bind_identity: Identity
    execution_identity: Identity
    artifact_identity: Identity
    field_providers: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "accepted_steps", _count(self.accepted_steps, where="accepted_steps")
        )
        object.__setattr__(
            self, "rejected_steps", _count(self.rejected_steps, where="rejected_steps")
        )
        if isinstance(self.final_time, bool) or not isinstance(self.final_time, (int, float)):
            raise TypeError("final_time must be a real number")
        final_time = float(self.final_time)
        if not math.isfinite(final_time):
            raise ValueError("final_time must be finite")
        object.__setattr__(self, "final_time", final_time)
        object.__setattr__(
            self, "final_macro_step", _count(self.final_macro_step, where="final_macro_step")
        )
        if type(self.stop_reason) is not RunStopReason:
            raise TypeError("stop_reason must be a RunStopReason")
        for name, domain in (
            ("run_identity", "run"),
            ("bind_identity", "bind"),
            ("execution_identity", "execution-context"),
            ("artifact_identity", "artifact"),
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), domain=domain, where=name),
            )
        if not isinstance(self.field_providers, (list, tuple)):
            raise TypeError("field_providers must be an ordered report sequence")
        object.__setattr__(
            self,
            "field_providers",
            freeze_data(self.field_providers, "field_providers"),
        )

    def to_data(self) -> dict[str, Any]:
        """Return detached, canonical report data suitable for inspection or serialization."""
        return {
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "final_time": self.final_time,
            "final_macro_step": self.final_macro_step,
            "stop_reason": self.stop_reason.value,
            "run_identity": self.run_identity.to_data(),
            "bind_identity": self.bind_identity.to_data(),
            "execution_identity": self.execution_identity.to_data(),
            "artifact_identity": self.artifact_identity.to_data(),
            "field_providers": thaw_data(self.field_providers),
        }

    def __bool__(self) -> bool:
        raise TypeError(
            "RunReport has no implicit truth value; inspect accepted_steps or stop_reason"
        )


__all__ = ["RunReport", "RunStopReason"]
