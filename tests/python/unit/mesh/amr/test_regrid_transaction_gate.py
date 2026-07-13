from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pops.mesh.amr import (
    CanonicalOptions,
    ClusteringPolicy,
    DerivedNestingRequirements,
    FrozenHierarchy,
    HierarchyLifecycleEvents,
    HierarchyPhaseError,
    HierarchyPlan,
    HierarchyProviderCapabilities,
    HierarchyResolutionContext,
    LevelTransition,
    LoadBalancePolicy,
    NestingRequirementSource,
    PatchGenerationPolicy,
    RegridDueToken,
    RegridRequest,
    RegridSchedule,
    RegridTransactionGate,
    resolve_hierarchy,
)
from pops.model import Handle, OwnerPath
from pops.time import Clock, EventHandle, StepTransactionReport, TimePoint, every


OWNER = OwnerPath.shared("amr-regrid-gate-tests")


def _handle(name: str, kind: str) -> Handle:
    return Handle(name, kind=kind, owner=OWNER)


def _clock(name: str = "main") -> Clock:
    return Clock(name, owner=OWNER)


def _due_event(name: str = "regrid_due") -> EventHandle:
    return EventHandle(OWNER, name)


def _source(role: str) -> NestingRequirementSource:
    return NestingRequirementSource(_handle(role, f"amr_{role}_requirement"), (1, 1), 1)


def _plan(
    regrid: RegridSchedule | FrozenHierarchy | None = None,
) -> HierarchyPlan:
    return HierarchyPlan(
        transitions=(LevelTransition(0, 1, (2, 2), (1, 1), 1),),
        nesting=DerivedNestingRequirements(
            stencil=_source("stencil"),
            transfer=_source("transfer"),
            reflux=_source("reflux"),
            boundary=_source("boundary"),
        ),
        clustering=ClusteringPolicy(
            _handle("cluster", "amr_clustering_provider"), CanonicalOptions()
        ),
        patch_generation=PatchGenerationPolicy(
            _handle("patches", "amr_patch_generation_provider"), CanonicalOptions()
        ),
        load_balance=LoadBalancePolicy(
            _handle("balance", "amr_load_balance_provider"), CanonicalOptions()
        ),
        regrid=regrid or RegridSchedule(every(4, clock=_clock()), _due_event()),
    )


def _gate(regrid: RegridSchedule | FrozenHierarchy | None = None) -> RegridTransactionGate:
    plan = _plan(regrid)
    provider = HierarchyProviderCapabilities(
        _handle("native", "amr_hierarchy_provider"),
        (2,),
        False,
        2,
        type(plan.regrid) is RegridSchedule,
        type(plan.regrid) is RegridSchedule,
    )
    clock = plan.regrid.schedule.clock if type(plan.regrid) is RegridSchedule else _clock()
    return RegridTransactionGate(
        resolve_hierarchy(plan, provider, HierarchyResolutionContext(clock))
    )


def _events() -> HierarchyLifecycleEvents:
    return HierarchyLifecycleEvents(
        create=(_handle("fine-new", "amr_patch_create"),),
        destroy=(_handle("fine-old", "amr_patch_destroy"),),
        rebalance=(_handle("fine-moved", "amr_patch_rebalance"),),
    )


def _request(gate: RegridTransactionGate) -> RegridRequest:
    schedule = gate.hierarchy.plan.regrid
    assert type(schedule) is RegridSchedule
    due = RegridDueToken(
        schedule.due_event,
        schedule.identity,
        TimePoint(schedule.schedule.clock, step=4),
        4,
    )
    return RegridRequest(due, _events())


def _transaction(status: str = "accepted", phase: str = "commit") -> StepTransactionReport:
    return StepTransactionReport(status=status, phase=phase, action="regrid")


def test_rejected_attempt_neither_plans_nor_commits_regrid() -> None:
    gate = _gate()

    request = _request(gate)
    decision = gate.evaluate(request, _transaction("rejected", "guard"), at=request.due.point)

    assert not decision.planned_regrid
    assert not decision.committed_regrid
    assert decision.to_data() == {"planned": None, "committed": None}


def test_accepted_commit_atomically_plans_and_commits_lifecycle_events() -> None:
    gate = _gate()
    request = _request(gate)

    decision = gate.evaluate(request, _transaction(), at=request.due.point)

    assert decision.planned == request.lifecycle
    assert decision.committed == request.lifecycle
    assert decision.to_data()["committed"]["rebalance"][0]["local_id"] == "fine-moved"


def test_gate_authenticates_program_event_schedule_clock_and_commit_phase() -> None:
    gate = _gate()
    request = _request(gate)
    schedule = gate.hierarchy.plan.regrid
    assert type(schedule) is RegridSchedule

    wrong_event = RegridRequest(
        RegridDueToken(_due_event("other"), schedule.identity, request.due.point, 4),
        request.lifecycle,
    )
    with pytest.raises(ValueError, match="different Program event"):
        gate.evaluate(wrong_event, _transaction(), at=wrong_event.due.point)

    other_schedule = RegridSchedule(every(5, clock=schedule.schedule.clock), schedule.due_event)
    wrong_schedule = RegridRequest(
        RegridDueToken(schedule.due_event, other_schedule.identity, request.due.point, 4),
        request.lifecycle,
    )
    with pytest.raises(ValueError, match="different regrid schedule"):
        gate.evaluate(wrong_schedule, _transaction(), at=wrong_schedule.due.point)

    wrong_clock = RegridRequest(
        RegridDueToken(
            schedule.due_event,
            schedule.identity,
            TimePoint(_clock("other"), step=4),
            4,
        ),
        request.lifecycle,
    )
    with pytest.raises(ValueError, match="not synchronized"):
        gate.evaluate(wrong_clock, _transaction(), at=wrong_clock.due.point)

    with pytest.raises(HierarchyPhaseError, match="commit phase"):
        gate.evaluate(request, _transaction("accepted", "stage"), at=request.due.point)
    with pytest.raises(ValueError, match="exactly agree"):
        RegridDueToken(schedule.due_event, schedule.identity, request.due.point, 5)

    with pytest.raises(ValueError, match="stale"):
        gate.evaluate(
            request,
            _transaction(),
            at=TimePoint(schedule.schedule.clock, step=8),
        )

    off_cadence = RegridRequest(
        RegridDueToken(
            schedule.due_event,
            schedule.identity,
            TimePoint(schedule.schedule.clock, step=5),
            5,
        ),
        request.lifecycle,
    )
    with pytest.raises(ValueError, match="Every cadence"):
        gate.evaluate(off_cadence, _transaction(), at=off_cadence.due.point)


def test_frozen_hierarchy_has_no_runtime_request_or_transactional_capability_need() -> None:
    frozen_gate = _gate(FrozenHierarchy())

    assert frozen_gate.evaluate(None, _transaction(), at=TimePoint(_clock(), step=4)).to_data() == {
        "planned": None,
        "committed": None,
    }
    with pytest.raises(ValueError, match="frozen hierarchy"):
        request = _request(_gate())
        frozen_gate.evaluate(request, _transaction(), at=request.due.point)


def test_lifecycle_and_due_contracts_are_immutable_and_kind_checked() -> None:
    gate = _gate()
    events = _events()
    request = _request(gate)

    with pytest.raises(FrozenInstanceError):
        events.create = ()
    with pytest.raises(FrozenInstanceError):
        request.due.accepted_cycle = 8
    with pytest.raises(TypeError, match="amr_patch_create"):
        HierarchyLifecycleEvents(create=(_handle("bad", "amr_patch_destroy"),))
    with pytest.raises(ValueError, match="at least one"):
        HierarchyLifecycleEvents()
