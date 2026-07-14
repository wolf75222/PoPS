"""Atomic step authoring contracts and reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pops.time.solve_outcome import FailRun, RejectAttempt, SolveAction
from pops.time._step.strategy import StepStrategy


class ProvisionalStore(str, Enum):
    """Implemented runtime-owned stores made provisional until macro-step acceptance.

    This enumeration is a capability manifest, not a wishlist.  In particular, PoPS does not yet
    expose a stochastic Program primitive or a core-owned, counter-based RNG stream, so no ``rng``
    store is advertised.  Adding one requires native snapshot/restart ownership and a deterministic
    stream-allocation contract; storing an opaque process-global generator here would be misleading.
    """

    STATES = "states"
    FIELDS = "fields"
    TOPOLOGY = "topology"
    FLUX_LEDGERS = "flux_ledgers"
    CACHES = "caches"
    SOLVER_WARM_STARTS = "solver_warm_starts"
    HISTORIES = "histories"
    CLOCKS = "clocks"
    SCHEDULES = "schedules"
    CONSUMERS = "consumers"
    DIAGNOSTICS = "diagnostics"
    EXTERNAL_EFFECTS = "external_effects"


ALL_PROVISIONAL_STORES = tuple(ProvisionalStore)


class GuardRole(str, Enum):
    INVARIANT = "invariant"
    ERROR_ESTIMATE = "error_estimate"


@dataclass(frozen=True, slots=True)
class BlockProjection:
    """Apply the exact native pointwise projection declared by the value's owning block."""

    kind: str = field(default="block_projection", init=False)
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True, slots=True)
class ProjectAndRecheck:
    """Project the candidate lazily, re-evaluate its guard, then apply a terminal action."""

    projection: BlockProjection
    on_failure: SolveAction = field(default_factory=RejectAttempt)
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.projection) is not BlockProjection:
            raise TypeError("ProjectAndRecheck.projection must be BlockProjection()")
        if not isinstance(self.on_failure, (RejectAttempt, FailRun)):
            raise TypeError("ProjectAndRecheck.on_failure must be RejectAttempt() or FailRun()")

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "project_and_recheck",
            "projection": self.projection.to_data(),
            "on_failure": self.on_failure.to_data(),
        }


@dataclass(frozen=True, slots=True)
class AcceptanceGuard:
    """Detached metadata for one actually lowered Program acceptance guard."""

    name: str
    role: GuardRole
    action: SolveAction | ProjectAndRecheck
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("AcceptanceGuard.name must be a non-empty string")
        if type(self.role) is not GuardRole:
            raise TypeError("AcceptanceGuard.role must be a GuardRole")
        if not isinstance(self.action, (RejectAttempt, FailRun, ProjectAndRecheck)):
            raise TypeError(
                "AcceptanceGuard.action must be RejectAttempt(), FailRun(), or ProjectAndRecheck()")

    def to_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "action": self.action.to_data(),
        }


_PHASES = frozenset((
    "prepare", "stage", "solve", "synchronize", "guard", "effect", "commit",
))
_STATUSES = frozenset(("accepted", "rejected", "failed"))


@dataclass(frozen=True, slots=True)
class StepTransactionReport:
    """Serializable observed result of one complete macro-step transaction."""

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


@dataclass(frozen=True, slots=True)
class StepTransactionPlan:
    """Frozen, identity-bearing contract attached to one compiled Program."""

    strategy: StepStrategy
    stores: tuple[ProvisionalStore, ...] = ALL_PROVISIONAL_STORES
    guards: tuple[AcceptanceGuard, ...] = ()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        ensure_step_strategy(self.strategy)
        if not self.stores or any(type(store) is not ProvisionalStore for store in self.stores):
            raise TypeError("StepTransactionPlan.stores must contain typed ProvisionalStore values")
        if len(set(self.stores)) != len(self.stores):
            raise ValueError("StepTransactionPlan.stores cannot contain duplicates")
        if any(type(guard) is not AcceptanceGuard for guard in self.guards):
            raise TypeError("StepTransactionPlan.guards must contain AcceptanceGuard values")
        if len({guard.name for guard in self.guards}) != len(self.guards):
            raise ValueError("StepTransactionPlan guard names must be unique")

    @property
    def projections(self) -> tuple[BlockProjection, ...]:
        return tuple(
            guard.action.projection for guard in self.guards
            if type(guard.action) is ProjectAndRecheck)

    def to_data(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.to_data(),
            "stores": [store.value for store in self.stores],
            "guards": [guard.to_data() for guard in self.guards],
            "projections": [projection.to_data() for projection in self.projections],
        }


def ensure_step_strategy(strategy: Any) -> StepStrategy:
    from pops.time._step.strategy import registered_step_strategy_type

    if not isinstance(strategy, StepStrategy):
        raise TypeError("step_strategy expects a registered StepStrategy provider")
    provider = registered_step_strategy_type(getattr(strategy, "kind", None))
    if provider is not type(strategy):
        raise TypeError("step_strategy provider is not registered for its exact kind")
    required = (
        "from_data", "restore_runtime_controls", "runtime_controls_data", "to_data",
        "validate_runtime_controls",
    )
    if any(not callable(getattr(strategy, name, None)) for name in required):
        raise TypeError("step_strategy provider does not implement the complete protocol")
    descriptor = strategy.to_data()
    if not isinstance(descriptor, dict) or descriptor.get("kind") != strategy.kind:
        raise TypeError("step_strategy provider returned an invalid canonical descriptor")
    return strategy


__all__ = [
    "ALL_PROVISIONAL_STORES", "AcceptanceGuard", "BlockProjection", "GuardRole",
    "ProjectAndRecheck", "ProvisionalStore", "StepTransactionPlan", "StepTransactionReport",
    "ensure_step_strategy",
]
