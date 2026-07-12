"""Step transaction reports and authoring plans.

`StepStrategy` lives in :mod:`pops.time.step_strategy`.  This module carries the
transaction-specific contract: which effects are staged and what a native
attempt report may claim.  It is deliberately data-only at the Python layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pops.time.step_strategy import StepStrategy


_PHASES = frozenset((
    "prepare", "stage", "solve", "synchronize", "guard", "effect", "commit",
))
_STATUSES = frozenset(("accepted", "rejected", "failed"))


@dataclass(frozen=True)
class StepTransactionReport:
    """Serializable report shape emitted by a native attempt controller."""

    status: str
    phase: str
    action: str
    attempts: int = 1
    staged_effects: tuple[str, ...] = ()
    committed_effects: tuple[str, ...] = ()
    rolled_back_effects: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError("StepTransactionReport.status must be one of %s" % sorted(_STATUSES))
        if self.phase not in _PHASES:
            raise ValueError("StepTransactionReport.phase must be one of %s" % sorted(_PHASES))
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts <= 0:
            raise ValueError("StepTransactionReport.attempts must be a positive integer")
        if self.status == "accepted" and self.rolled_back_effects:
            raise ValueError("accepted transaction reports cannot include rolled_back_effects")
        if self.status != "accepted" and self.committed_effects:
            raise ValueError("rejected/failed transaction reports cannot include committed_effects")

    def to_data(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "action": self.action,
            "attempts": self.attempts,
            "staged_effects": list(self.staged_effects),
            "committed_effects": list(self.committed_effects),
            "rolled_back_effects": list(self.rolled_back_effects),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class StepTransactionPlan:
    """Frozen authoring contract attached to one Program before lowering."""

    strategy: StepStrategy
    staged_effects: tuple[str, ...] = field(default_factory=tuple)
    guards: tuple[str, ...] = field(default_factory=tuple)
    projections: tuple[str, ...] = field(default_factory=tuple)

    def to_data(self) -> dict[str, Any]:
        return {
            "strategy": repr(self.strategy),
            "staged_effects": list(self.staged_effects),
            "guards": list(self.guards),
            "projections": list(self.projections),
        }


def ensure_step_strategy(strategy: Any) -> StepStrategy:
    if not isinstance(strategy, StepStrategy):
        raise TypeError(
            "step_strategy expects FixedDt(), AdaptiveCFL(), ErrorControlledDt(), "
            "or ExternalTimeGrid()")
    return strategy


__all__ = ["StepTransactionPlan", "StepTransactionReport", "ensure_step_strategy"]
