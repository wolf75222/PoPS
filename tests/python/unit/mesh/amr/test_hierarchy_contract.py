from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect

import pytest

from pops.amr import (
    PreparedHierarchyNativeLowering,
    PreparedHierarchyNativeProvider,
    register_prepared_hierarchy_native_provider,
)
from pops.mesh._amr import (
    CanonicalOptions,
    ClusteringPolicy,
    DerivedNestingRequirements,
    FrozenHierarchy,
    HierarchyCapabilityError,
    HierarchyPlan,
    HierarchyProviderCapabilities,
    HierarchyResolutionContext,
    LevelTransition,
    LoadBalancePolicy,
    NestingRequirementSource,
    PatchGenerationPolicy,
    RegridSchedule,
    ResolvedHierarchy,
    lower_native_hierarchy,
    resolve_hierarchy,
)
from pops.model import Handle, OwnerPath
from pops.time import Attempt, Clock, EventHandle, Every, Schedule, every, when


OWNER = OwnerPath.shared("amr-contract-tests")


def _handle(name: str, kind: str) -> Handle:
    return Handle(name, kind=kind, owner=OWNER)


def _clock(name: str = "main") -> Clock:
    return Clock(name, owner=OWNER)


def _due_event(name: str = "regrid_due") -> EventHandle:
    return EventHandle(OWNER, name)


def _source(role: str, buffer: tuple[int, ...], lookahead: int) -> NestingRequirementSource:
    return NestingRequirementSource(_handle(role, "amr_%s_requirement" % role), buffer, lookahead)


def _nesting(dimension: int = 2, *, extra: int = 0) -> DerivedNestingRequirements:
    base = tuple(1 + extra for _ in range(dimension))
    transfer = tuple(2 + extra for _ in range(dimension))
    return DerivedNestingRequirements(
        stencil=_source("stencil", base, 1),
        transfer=_source("transfer", transfer, 2),
        reflux=_source("reflux", base, 1),
        boundary=_source("boundary", base, 0),
    )


def _clustering(name: str = "berger_rigoutsos", efficiency: float = 0.7) -> ClusteringPolicy:
    return ClusteringPolicy(
        _handle(name, "amr_clustering_provider"),
        CanonicalOptions({"minimum_efficiency": efficiency}),
    )


def _patches(name: str = "box_generator") -> PatchGenerationPolicy:
    return PatchGenerationPolicy(
        _handle(name, "amr_patch_generation_provider"),
        CanonicalOptions({"maximum_extent": [32, 32]}),
    )


def _balance(name: str = "space_filling_curve") -> LoadBalancePolicy:
    return LoadBalancePolicy(
        _handle(name, "amr_load_balance_provider"),
        CanonicalOptions({"weighted": True}),
    )


def _plan(
    *,
    transitions: tuple[LevelTransition, ...] | None = None,
    nesting: DerivedNestingRequirements | None = None,
    clock: Clock | None = None,
    regrid: RegridSchedule | FrozenHierarchy | None = None,
) -> HierarchyPlan:
    if transitions is None:
        transitions = (
            LevelTransition(0, 1, (2, 2), (2, 2), 2),
            LevelTransition(1, 2, (2, 4), (2, 2), 2),
        )
    return HierarchyPlan(
        transitions=transitions,
        nesting=nesting or _nesting(len(transitions[0].ratio)),
        clustering=_clustering(),
        patch_generation=_patches(),
        load_balance=_balance(),
        regrid=regrid or RegridSchedule(every(4, clock=clock or _clock()), _due_event()),
    )


def _provider(
    *,
    dimensions: tuple[int, ...] = (2,),
    anisotropic: bool = True,
    levels: int = 3,
    transactional: bool = True,
    lifecycle: bool = True,
) -> HierarchyProviderCapabilities:
    return HierarchyProviderCapabilities(
        _handle("native_amr", "amr_hierarchy_provider"),
        dimensions,
        anisotropic,
        levels,
        transactional,
        lifecycle,
    )


def _context(clock: Clock) -> HierarchyResolutionContext:
    return HierarchyResolutionContext(clock)


def test_three_level_hierarchy_is_explicit_and_level_count_is_strictly_derived() -> None:
    plan = _plan()

    assert plan.level_count == 3
    assert plan.dimension == 2
    assert plan.transitions[1].anisotropic
    assert plan.nesting.minimum_buffer == (2, 2)
    assert plan.nesting.minimum_lookahead == 2
    assert plan.identity.domain == "amr-hierarchy-plan"
    assert plan.inspect()["level_count"] == 3

    parameters = inspect.signature(HierarchyPlan).parameters
    assert "nesting" in parameters
    for forbidden in ("level_count", "nlevels", "max_levels", "order", "ghost"):
        assert forbidden not in parameters


def test_native_hierarchy_lowering_dispatches_to_an_opaque_provider_route() -> None:
    clock = _clock("external-native")
    route = "tests.hierarchy-native.external-graph@3"
    observed: list[ResolvedHierarchy] = []
    def lower_external(
        hierarchy: ResolvedHierarchy, authority: dict[str, object]
    ) -> PreparedHierarchyNativeLowering:
        observed.append(hierarchy)
        return PreparedHierarchyNativeLowering(
            authority,
            hierarchy.plan.level_count,
            nesting_buffer=2,
            nesting_lookahead=2,
        )

    provider = register_prepared_hierarchy_native_provider(
        PreparedHierarchyNativeProvider(route, 3, lower_external)
    )
    capabilities = replace(
        _provider(),
        options=CanonicalOptions({
            "native_route": route,
            "native_provider": provider.authority(),
            "radius": 2,
        }),
    )
    resolved = resolve_hierarchy(_plan(clock=clock), capabilities, _context(clock))

    lowered = lower_native_hierarchy(resolved)

    assert observed == [resolved, resolved]
    assert lowered.provider == provider.authority()
    assert lowered.level_count == resolved.plan.level_count


def test_transition_fields_change_identity_or_are_rejected() -> None:
    baseline = _plan()
    first, second = baseline.transitions

    variants = (
        replace(first, ratio=(3, 3)),
        replace(first, buffer=(3, 2)),
        replace(first, lookahead=3),
    )
    for variant in variants:
        changed = replace(baseline, transitions=(variant, second))
        assert changed.identity != baseline.identity

    with pytest.raises(ValueError, match="fine_level"):
        replace(first, fine_level=2)
    with pytest.raises(ValueError, match="contiguous"):
        replace(baseline, transitions=(LevelTransition(1, 2, (2, 2), (2, 2), 2),))


def test_every_component_and_requirement_source_changes_plan_identity() -> None:
    baseline = _plan()
    candidates = (
        replace(baseline, clustering=_clustering("other_cluster")),
        replace(baseline, clustering=_clustering(efficiency=0.8)),
        replace(baseline, patch_generation=_patches("other_boxes")),
        replace(baseline, load_balance=_balance("other_balance")),
        replace(
            baseline,
            regrid=RegridSchedule(
                every(5, clock=baseline.regrid.schedule.clock), baseline.regrid.due_event
            ),
        ),
        replace(baseline, regrid=RegridSchedule(baseline.regrid.schedule, _due_event("other_due"))),
        replace(baseline, regrid=FrozenHierarchy()),
        replace(baseline, nesting=_nesting(extra=1)),
    )
    assert all(candidate.identity != baseline.identity for candidate in candidates)

    nesting = baseline.nesting
    for field in ("stencil", "transfer", "reflux", "boundary"):
        source = getattr(nesting, field)
        changed_source = replace(source, minimum_lookahead=source.minimum_lookahead + 1)
        changed = replace(baseline, nesting=replace(nesting, **{field: changed_source}))
        assert changed.identity != baseline.identity


def test_authoring_contracts_are_deeply_immutable_and_data_only() -> None:
    plan = _plan()

    with pytest.raises(FrozenInstanceError):
        plan.transitions = ()
    with pytest.raises(FrozenInstanceError):
        plan.transitions[0].ratio = (3, 3)
    with pytest.raises(FrozenInstanceError):
        plan.clustering.options._data = ()
    with pytest.raises(TypeError, match="callbacks/objects are forbidden"):
        CanonicalOptions({"algorithm": lambda: None})


def test_resolution_is_pre_runtime_and_requires_a_synchronized_dynamic_schedule() -> None:
    plan = _plan()

    resolved = resolve_hierarchy(plan, _provider(), _context(plan.regrid.schedule.clock))
    assert resolved.plan is plan
    with pytest.raises(HierarchyCapabilityError, match="not synchronized") as mismatch:
        resolve_hierarchy(plan, _provider(), _context(_clock("other")))
    assert mismatch.value.evidence["schedule_clock"] != mismatch.value.evidence["resolution_clock"]

    attempt_schedule = Schedule(Every(Attempt(_clock()), 4))
    with pytest.raises(ValueError, match="AcceptedStep"):
        RegridSchedule(attempt_schedule, _due_event())
    with pytest.raises(ValueError, match="Always or Every"):
        RegridSchedule(when(True, clock=_clock()), _due_event())
    with pytest.raises(ValueError, match="one Program owner"):
        RegridSchedule(
            every(4, clock=_clock()), EventHandle(OwnerPath.shared("other-owner"), "due")
        )


def test_frozen_hierarchy_resolves_without_runtime_regrid_capabilities() -> None:
    plan = _plan(regrid=FrozenHierarchy())
    provider = _provider(transactional=False, lifecycle=False)

    resolved = resolve_hierarchy(plan, provider, _context(_clock("unrelated-runtime-clock")))

    assert type(resolved.plan.regrid) is FrozenHierarchy
    assert resolved.plan.regrid.identity.domain == "amr-frozen-hierarchy"
    with pytest.raises(TypeError, match="ResolvedHierarchy.plan"):
        ResolvedHierarchy("not-a-plan", provider)


def test_provider_refuses_anisotropy_3d_and_three_levels_with_capability_evidence() -> None:
    plan = _plan()
    context = _context(plan.regrid.schedule.clock)

    with pytest.raises(HierarchyCapabilityError, match="anisotropic") as anisotropic:
        resolve_hierarchy(plan, _provider(anisotropic=False), context)
    assert anisotropic.value.evidence["supports_anisotropic_ratio"] is False

    with pytest.raises(HierarchyCapabilityError, match="level count") as levels:
        resolve_hierarchy(plan, _provider(levels=2), context)
    assert levels.value.evidence == {"requested_level_count": 3, "supported_level_count": 2}

    transitions_3d = (
        LevelTransition(0, 1, (2, 2, 2), (2, 2, 2), 2),
        LevelTransition(1, 2, (2, 2, 2), (2, 2, 2), 2),
    )
    plan_3d = _plan(transitions=transitions_3d, nesting=_nesting(3))
    with pytest.raises(HierarchyCapabilityError, match="dimension") as dimension:
        resolve_hierarchy(
            plan_3d, _provider(dimensions=(2,)), _context(plan_3d.regrid.schedule.clock)
        )
    assert dimension.value.evidence["requested_dimension"] == 3


def test_nesting_and_dynamic_provider_requirements_fail_before_artifact() -> None:
    inadequate = _plan(
        transitions=(
            LevelTransition(0, 1, (2, 2), (1, 2), 1),
            LevelTransition(1, 2, (2, 4), (2, 2), 2),
        )
    )
    context = _context(inadequate.regrid.schedule.clock)
    with pytest.raises(HierarchyCapabilityError, match="nesting") as nesting:
        resolve_hierarchy(inadequate, _provider(), context)
    assert nesting.value.evidence["insufficient_axes"] == [0]

    for field in ("supports_transactional_regrid", "supports_lifecycle_events"):
        provider = _provider(
            transactional=field != "supports_transactional_regrid",
            lifecycle=field != "supports_lifecycle_events",
        )
        with pytest.raises(HierarchyCapabilityError) as error:
            resolve_hierarchy(_plan(), provider, _context(_clock()))
        assert error.value.evidence[field] is False


def test_provider_capability_fields_change_identity() -> None:
    baseline = _provider()
    candidates = (
        _provider(dimensions=(1, 2)),
        _provider(anisotropic=False),
        _provider(levels=4),
        _provider(transactional=False),
        _provider(lifecycle=False),
        replace(baseline, provider=_handle("alternate", "amr_hierarchy_provider")),
    )
    assert all(candidate.identity != baseline.identity for candidate in candidates)
