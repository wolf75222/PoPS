from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from pops.mesh.layout_plan import (
    LayoutHandle,
    LayoutMappingOperation,
    LayoutMappingRequirement,
    LayoutPlan,
    LayoutRepresentation,
    LayoutSynchronization,
    LayoutPlanBuilder,
    NormalizedGeometry,
    ResolvedLayoutMapping,
    normalize_layout_plan,
)
from pops.layouts import Uniform
from pops.model import Handle, OwnerKind, OwnerPath
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout


@dataclass(frozen=True)
class Provider:
    qualified_id: str
    routes: frozenset[tuple[str, str]]

    def canonical_identity(self):
        return {"qualified_id": self.qualified_id, "routes": sorted(self.routes)}

    def supports_layout_mapping(self, requirement: LayoutMappingRequirement) -> bool:
        return (requirement.source_layout.qualified_id,
                requirement.target_layout.qualified_id) in self.routes


def _refs():
    return (
        Handle("U", kind="state", owner=OwnerPath.model("fluid")),
        Handle("phi", kind="field", owner=OwnerPath.case("main")),
        Handle("fluid", kind="block", owner=OwnerPath.case("main")),
    )


def _complete_builder(reverse: bool = True):
    state, field, block = _refs()
    builder = LayoutPlanBuilder(OwnerPath.case("main"))
    uniform = builder.layout("base", Uniform(cartesian_grid(n=16)))
    adaptive = builder.layout(
        "adaptive", final_amr_layout(cartesian_grid(n=16), max_levels=2, ratio=2))
    builder.assign_state(state, adaptive)
    builder.assign_field(field, uniform)
    builder.assign_block(block, adaptive)
    builder.require_mapping(
        adaptive, uniform, source=state, target=field,
        operation=LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1,
        synchronization=LayoutSynchronization.BEFORE_STEP_V1,
        source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
        target_representation=LayoutRepresentation.CELL_AVERAGE_V1)
    if reverse:
        builder.require_mapping(
            uniform, adaptive, source=field, target=state,
            operation=LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1,
            synchronization=LayoutSynchronization.BEFORE_STEP_V1,
            source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            target_representation=LayoutRepresentation.CELL_AVERAGE_V1)
    return builder, uniform, adaptive, state, field, block


def test_layout_handle_is_immutable_hashable_and_owner_qualified():
    handle = LayoutHandle("fluid", owner=OwnerPath.case("main"))
    assert "::layout::fluid" in handle.qualified_id
    assert {handle: "ok"}[LayoutHandle("fluid", owner=OwnerPath.case("main"))] == "ok"
    with pytest.raises(AttributeError):
        handle.local_id = "other"
    assert LayoutHandle("fluid", owner=OwnerPath.case("a")) != \
        LayoutHandle("fluid", owner=OwnerPath.case("b"))
    assert LayoutHandle.from_canonical_identity(handle.canonical_identity()) == handle


def test_uniform_and_amr_share_one_level_plan_representation():
    builder, uniform, adaptive, state, field, block = _complete_builder(reverse=False)
    forward = Provider("provider/down", frozenset(((adaptive.qualified_id, uniform.qualified_id),)))
    plan = builder.resolve(states=[state], fields=[field], blocks=[block], providers=[forward])
    by_id = {row.handle.qualified_id: row for row in plan.layouts}
    assert [level.refinement for level in by_id[uniform.qualified_id].levels] == [1]
    assert by_id[uniform.qualified_id].transition_ratios == ()
    assert by_id[adaptive.qualified_id].transition_ratios == (2,)
    assert [level.refinement for level in by_id[adaptive.qualified_id].levels] == [1, 2]
    assert type(by_id[uniform.qualified_id]) is type(by_id[adaptive.qualified_id])


def test_assignments_are_exact_and_lookup_is_kind_qualified():
    builder, uniform, adaptive, state, field, block = _complete_builder(reverse=False)
    provider = Provider("provider/down", frozenset(((adaptive.qualified_id, uniform.qualified_id),)))
    plan = builder.resolve(states=[state], fields=[field], blocks=[block], providers=[provider])
    assert plan.layout_for(state) == adaptive
    assert plan.layout_for(field) == uniform
    with pytest.raises(TypeError, match="kind='field'"):
        builder.assign_field(state, uniform)


def test_unassigned_double_and_unexpected_assignments_fail_loud():
    state, field, block = _refs()
    builder = LayoutPlanBuilder(OwnerPath.case("main"))
    layout = builder.layout("base", Uniform(cartesian_grid(n=8)))
    builder.assign_state(state, layout)
    with pytest.raises(ValueError, match="double layout assignment"):
        builder.assign_state(state, layout)
    with pytest.raises(ValueError, match="unassigned layout subjects"):
        builder.resolve(states=[state], fields=[field])
    with pytest.raises(ValueError, match="unexpected subjects"):
        builder.resolve()


def test_each_explicit_direction_requires_its_own_provider():
    builder, uniform, adaptive, state, field, block = _complete_builder(reverse=True)
    down = Provider("provider/down", frozenset(((adaptive.qualified_id, uniform.qualified_id),)))
    with pytest.raises(ValueError, match="missing mapping provider"):
        builder.resolve(states=[state], fields=[field], blocks=[block], providers=[down])

    up = Provider("provider/up", frozenset(((uniform.qualified_id, adaptive.qualified_id),)))
    plan = builder.resolve(states=[state], fields=[field], blocks=[block], providers=[up, down])
    assert {row.provider_id for row in plan.mappings} == {"provider/down", "provider/up"}
    assert len(plan.mappings) == 2


def test_mapping_provider_resolution_rejects_ambiguity_and_duplicate_identity():
    builder, uniform, adaptive, state, field, block = _complete_builder(reverse=False)
    route = frozenset(((adaptive.qualified_id, uniform.qualified_id),))
    first = Provider("provider/first", route)
    second = Provider("provider/second", route)
    with pytest.raises(ValueError, match="ambiguous mapping providers"):
        builder.resolve(states=[state], fields=[field], blocks=[block], providers=[first, second])
    with pytest.raises(ValueError, match="duplicate mapping provider identity"):
        builder.resolve(states=[state], fields=[field], blocks=[block], providers=[first, first])


def test_overwrite_mappings_reject_two_writers_to_one_target_at_one_sync():
    owner = OwnerPath.case("main")
    source_a = Handle("A", kind="state", owner=OwnerPath.model("source_a"))
    source_b = Handle("B", kind="state", owner=OwnerPath.model("source_b"))
    target = Handle("C", kind="state", owner=OwnerPath.model("target"))
    builder = LayoutPlanBuilder(owner)
    layout_a = builder.layout("a", Uniform(cartesian_grid(n=16)))
    layout_b = builder.layout("b", Uniform(cartesian_grid(n=16)))
    layout_c = builder.layout("c", Uniform(cartesian_grid(n=8)))
    for state, layout in (
            (source_a, layout_a), (source_b, layout_b), (target, layout_c)):
        builder.assign_state(state, layout)
    requirements = []
    for source, layout in ((source_a, layout_a), (source_b, layout_b)):
        requirements.extend(builder.require_mapping(
            layout, layout_c, source=source, target=target,
            operation=LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1,
            synchronization=LayoutSynchronization.BEFORE_STEP_V1,
            source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            target_representation=LayoutRepresentation.CELL_AVERAGE_V1,
        ))
    provider = Provider("provider/down", frozenset((
        (layout_a.qualified_id, layout_c.qualified_id),
        (layout_b.qualified_id, layout_c.qualified_id),
    )))

    with pytest.raises(ValueError, match="concurrent overwrite mappings"):
        builder.resolve(states=(source_a, source_b, target), providers=(provider,))

    forged_rows = tuple(
        ResolvedLayoutMapping(requirement, provider.qualified_id, provider.canonical_identity())
        for requirement in requirements)
    with pytest.raises(ValueError, match="concurrent overwrite mappings"):
        LayoutPlan(
            builder.owner,
            tuple(sorted(builder._layouts.values(), key=lambda row: row.handle.qualified_id)),
            tuple(sorted(builder._assignments.values(), key=lambda row: row.subject_id)),
            forged_rows,
            "0" * 64,
        )


def test_plan_identity_and_inspection_are_canonical_and_detached():
    builder, uniform, adaptive, state, field, block = _complete_builder(reverse=True)
    providers = [
        Provider("provider/up", frozenset(((uniform.qualified_id, adaptive.qualified_id),))),
        Provider("provider/down", frozenset(((adaptive.qualified_id, uniform.qualified_id),))),
    ]
    first = builder.resolve(states=[state], fields=[field], blocks=[block], providers=providers)
    second = builder.resolve(states=[state], fields=[field], blocks=[block],
                             providers=list(reversed(providers)))
    assert first == second
    assert first.qualified_id == second.qualified_id
    report = first.inspect()
    assert report["schema_version"] == 1
    assert report["report_type"] == "layout_plan"
    assert json.loads(json.dumps(report)) == report
    report["layouts"].clear()
    assert len(first.layouts) == 2

    richer_down = Provider("provider/down", frozenset((
        (adaptive.qualified_id, uniform.qualified_id),
        (uniform.qualified_id, uniform.qualified_id),
    )))
    changed_provider = builder.resolve(
        states=[state], fields=[field], blocks=[block],
        providers=[providers[0], richer_down])
    assert changed_provider.canonical_id != first.canonical_id

    with pytest.raises(ValueError, match="does not authenticate"):
        type(first)(first.owner, first.layouts, first.assignments, first.mappings, "0" * 64)


def test_foreign_layout_handles_and_algorithm_shaped_providers_are_rejected():
    state, _, _ = _refs()
    builder = LayoutPlanBuilder(OwnerPath.case("main"))
    local = builder.layout("base", Uniform(cartesian_grid(n=8)))
    foreign = LayoutHandle("base", owner=OwnerPath.case("other"))
    with pytest.raises(ValueError, match="declared by this builder"):
        builder.assign_state(state, foreign)
    with pytest.raises(ValueError, match="distinct layouts"):
        builder.require_mapping(
            local, local, source=state, target=state,
            operation=LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1,
            synchronization=LayoutSynchronization.BEFORE_STEP_V1,
            source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            target_representation=LayoutRepresentation.CELL_AVERAGE_V1)


def test_bare_string_authorities_subjects_and_providers_are_never_promoted():
    with pytest.raises(TypeError, match="never a string"):
        LayoutPlanBuilder("case/main")
    builder = LayoutPlanBuilder(OwnerPath.case("main"))
    layout = builder.layout("base", Uniform(cartesian_grid(n=8)))
    with pytest.raises(TypeError, match="canonical pops.model.Handle"):
        builder.assign_state("U", layout)
    state = Handle("U", kind="state", owner=OwnerPath.model("fluid"))
    builder.assign_state(state, layout)
    other = builder.layout("other", Uniform(cartesian_grid(n=8)))
    field = Handle("phi", kind="field", owner=OwnerPath.case("main"))
    builder.assign_field(field, other)
    builder.require_mapping(
        layout, other, source=state, target=field,
        operation=LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1,
        synchronization=LayoutSynchronization.BEFORE_STEP_V1,
        source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
        target_representation=LayoutRepresentation.CELL_AVERAGE_V1)
    with pytest.raises(TypeError, match="never a string"):
        builder.resolve(states=[state], fields=[field], providers=["provider"])


def test_authoring_owner_is_not_silently_collapsed_to_a_homonymous_canonical_owner():
    first = OwnerPath.fresh(OwnerKind.CASE, "main")
    second = OwnerPath.fresh(OwnerKind.CASE, "main")
    assert first != second
    with pytest.raises(TypeError, match="post-resolution contract"):
        LayoutPlanBuilder(first)
    with pytest.raises(TypeError, match="post-resolution contract"):
        LayoutHandle("base", owner=second)


def test_public_single_layout_normalization_returns_a_degenerate_plan():
    state, field, block = _refs()
    plan = normalize_layout_plan(
        Uniform(cartesian_grid(n=8)), owner=OwnerPath.case("main"),
        states=[state], fields=[field], blocks=[block])
    assert len(plan.layouts) == 1
    assert len(plan.mappings) == 0
    assert plan.layout_for(state) == plan.layouts[0].handle
    assert [level.refinement for level in plan.layouts[0].levels] == [1]


def test_normalized_geometry_is_exact_detached_and_delegated_by_uniform_and_amr():
    owner = OwnerPath.case("main")
    grid = cartesian_grid(n=8, L=2.5)
    uniform = normalize_layout_plan(Uniform(grid), owner=owner).layouts[0]
    adaptive = normalize_layout_plan(
        final_amr_layout(grid, max_levels=2), owner=owner).layouts[0]

    assert uniform.geometry == adaptive.geometry
    assert uniform.geometry.axis_names == ("x", "y")
    assert uniform.geometry.lower == (0.0, 0.0)
    assert uniform.geometry.upper == (2.5, 2.5)
    assert uniform.geometry.cells == (8, 8)
    assert uniform.to_data()["geometry"] == adaptive.to_data()["geometry"]
    with pytest.raises(AttributeError):
        uniform.geometry.cells = (16, 16)


def test_normalized_geometry_protocol_is_called_twice_and_must_be_deterministic():
    class FlakyLayout:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def validate(self):
            return True

        def capabilities(self):
            return {"levels": 1, "supports_amr": False, "transition_ratios": []}

        def options(self):
            return {}

        def requirements(self):
            return {}

        def normalized_geometry(self):
            self.calls += 1
            return NormalizedGeometry(
                "pops://coordinates/test-2d@1",
                "pops://cell-measures/test-area@1",
                ("x", "y"), (0.0, 0.0), (float(self.calls), 1.0), (4, 4),
            )

    descriptor = FlakyLayout()
    with pytest.raises(ValueError, match="must be deterministic"):
        normalize_layout_plan(descriptor, owner=OwnerPath.case("main"))
    assert descriptor.calls == 2


def test_descriptor_snapshot_accounts_for_structured_hierarchy_semantics():
    owner = OwnerPath.case("main")
    first = normalize_layout_plan(
        final_amr_layout(cartesian_grid(n=8), max_levels=2), owner=owner)
    second = normalize_layout_plan(
        final_amr_layout(cartesian_grid(n=8), max_levels=3), owner=owner)
    assert first.layouts[0].options != second.layouts[0].options
    assert first.layouts[0].descriptor_snapshot != second.layouts[0].descriptor_snapshot
    assert first.canonical_id != second.canonical_id
