"""Transactional regrid requests and owner-qualified hierarchy lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.identity import Identity

from ._contracts import canonical_handle, event_data, schedule_data, time_point_data, transaction_data
from .hierarchy import FrozenHierarchy, HierarchyPhaseError, RegridSchedule
from .hierarchy_resolution import ResolvedHierarchy


def _events(values: Any, *, where: str, kind: str) -> tuple[Any, ...]:
    result = tuple(values)
    for value in result:
        canonical_handle(value, where=where, kinds=kind)
    if len(result) != len(set(result)):
        raise ValueError("%s event handles must be unique" % where)
    return result


@dataclass(frozen=True, slots=True)
class HierarchyLifecycleEvents:
    """Exact patch create/destroy/rebalance effects produced by one regrid."""

    create: tuple[Any, ...] = ()
    destroy: tuple[Any, ...] = ()
    rebalance: tuple[Any, ...] = ()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "create",
            _events(self.create, where="HierarchyLifecycleEvents.create", kind="amr_patch_create"),
        )
        object.__setattr__(
            self,
            "destroy",
            _events(
                self.destroy, where="HierarchyLifecycleEvents.destroy", kind="amr_patch_destroy"
            ),
        )
        object.__setattr__(
            self,
            "rebalance",
            _events(
                self.rebalance,
                where="HierarchyLifecycleEvents.rebalance",
                kind="amr_patch_rebalance",
            ),
        )
        if not (self.create or self.destroy or self.rebalance):
            raise ValueError("HierarchyLifecycleEvents requires at least one lifecycle effect")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "create": [value.canonical_identity() for value in self.create],
            "destroy": [value.canonical_identity() for value in self.destroy],
            "rebalance": [value.canonical_identity() for value in self.rebalance],
        }


@dataclass(frozen=True, slots=True)
class RegridDueToken:
    """Program-produced proof that one exact regrid schedule is due at an accepted cycle."""

    event: Any
    schedule_identity: Identity
    point: Any
    accepted_cycle: int
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        event_data(self.event, where="RegridDueToken.event")
        if (
            not isinstance(self.schedule_identity, Identity)
            or self.schedule_identity.domain != "amr-regrid-schedule"
        ):
            raise TypeError("RegridDueToken.schedule_identity must identify an AMR regrid schedule")
        time_point_data(self.point, where="RegridDueToken.point")
        if (
            isinstance(self.accepted_cycle, bool)
            or not isinstance(self.accepted_cycle, int)
            or self.accepted_cycle < 0
        ):
            raise ValueError("RegridDueToken.accepted_cycle must be a non-negative int")
        if self.point.step != self.accepted_cycle:
            raise ValueError("RegridDueToken point and accepted cycle must exactly agree")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "event": self.event.to_data(),
            "schedule_identity": self.schedule_identity.token,
            "point": self.point.to_data(),
            "accepted_cycle": self.accepted_cycle,
        }


@dataclass(frozen=True, slots=True)
class RegridRequest:
    """Lifecycle change attached to an authenticated Program-produced due token."""

    due: RegridDueToken
    lifecycle: HierarchyLifecycleEvents
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.due) is not RegridDueToken:
            raise TypeError("RegridRequest.due must be a RegridDueToken")
        if type(self.lifecycle) is not HierarchyLifecycleEvents:
            raise TypeError("RegridRequest.lifecycle must be HierarchyLifecycleEvents")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "due": self.due.canonical_identity(),
            "lifecycle": self.lifecycle.canonical_identity(),
        }


@dataclass(frozen=True, slots=True)
class RegridTransactionDecision:
    """Atomic decision: rejected attempts contain neither a plan nor a commit."""

    planned: HierarchyLifecycleEvents | None
    committed: HierarchyLifecycleEvents | None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if self.planned is not None and type(self.planned) is not HierarchyLifecycleEvents:
            raise TypeError("planned must be HierarchyLifecycleEvents or None")
        if self.committed is not None and type(self.committed) is not HierarchyLifecycleEvents:
            raise TypeError("committed must be HierarchyLifecycleEvents or None")
        if self.committed is not None and self.planned != self.committed:
            raise ValueError("a committed regrid must exactly match its transactional plan")

    @property
    def planned_regrid(self) -> bool:
        return self.planned is not None

    @property
    def committed_regrid(self) -> bool:
        return self.committed is not None

    def to_data(self) -> dict[str, Any]:
        return {
            "planned": None if self.planned is None else self.planned.canonical_identity(),
            "committed": None if self.committed is None else self.committed.canonical_identity(),
        }


@dataclass(frozen=True, slots=True)
class RegridTransactionGate:
    hierarchy: ResolvedHierarchy
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.hierarchy) is not ResolvedHierarchy:
            raise TypeError("RegridTransactionGate.hierarchy must be ResolvedHierarchy")

    def evaluate(
        self,
        request: RegridRequest | None,
        transaction: Any,
        *,
        at: Any,
    ) -> RegridTransactionDecision:
        transaction_data(transaction, where="regrid gate transaction")
        # This guard intentionally precedes request inspection: rejected/failed attempts cannot
        # even plan a regrid, including when their provisional request is malformed or stale.
        if transaction.status != "accepted":
            return RegridTransactionDecision(None, None)
        if transaction.phase != "commit":
            raise HierarchyPhaseError("accepted regrid evaluation requires commit phase")
        time_point_data(at, where="accepted regrid evaluation")
        schedule = self.hierarchy.plan.regrid
        if type(schedule) is FrozenHierarchy:
            if request is not None:
                raise ValueError("a frozen hierarchy cannot accept a regrid request")
            return RegridTransactionDecision(None, None)
        if type(schedule) is not RegridSchedule:
            raise TypeError("resolved hierarchy contains an unsupported regrid contract")
        if request is None:
            return RegridTransactionDecision(None, None)
        if type(request) is not RegridRequest:
            raise TypeError("regrid gate request must be RegridRequest or None")
        due = request.due
        if due.event != schedule.due_event:
            raise ValueError("RegridDueToken belongs to a different Program event")
        if due.schedule_identity != schedule.identity:
            raise ValueError("RegridRequest belongs to a different regrid schedule")
        if due.point.clock != schedule.schedule.clock:
            raise ValueError("RegridRequest point is not synchronized with the regrid clock")
        if due.point != at:
            raise ValueError("RegridDueToken is stale for the current commit point")
        trigger = schedule_data(schedule.schedule, where="resolved regrid schedule")["trigger"]
        decision = RegridTransactionDecision(request.lifecycle, request.lifecycle)
        if trigger["type"] == "always":
            return decision
        if trigger["type"] == "every" and at.step % trigger["n"] != 0:
            raise ValueError("RegridDueToken is not due at the current Every cadence")
        return decision


__all__ = [
    "HierarchyLifecycleEvents",
    "RegridDueToken",
    "RegridRequest",
    "RegridTransactionDecision",
    "RegridTransactionGate",
]
