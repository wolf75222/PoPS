from __future__ import annotations

from dataclasses import dataclass, replace
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
    NativeSpatialLayout,
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
    (forward,) = builder.require_mapping(
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
            target_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            reverse_of=forward)
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
    with pytest.raises(ValueError, match="missing reverse mapping provider"):
        builder.resolve(states=[state], fields=[field], blocks=[block], providers=[down])

    up = Provider("provider/up", frozenset(((uniform.qualified_id, adaptive.qualified_id),)))
    plan = builder.resolve(states=[state], fields=[field], blocks=[block], providers=[up, down])
    assert {row.provider_id for row in plan.mappings} == {"provider/down", "provider/up"}
    assert len(plan.mappings) == 2
    forward = next(row.requirement for row in plan.mappings if row.requirement.reverse_of is None)
    reverse = next(row.requirement for row in plan.mappings if row.requirement.reverse_of is not None)
    assert reverse.reverse_of == forward.qualified_id

    forged = tuple(
        replace(row, requirement=replace(row.requirement, reverse_of="missing-forward"))
        if row.requirement == reverse else row
        for row in plan.mappings
    )
    with pytest.raises(ValueError, match="reverse mapping references a missing requirement"):
        LayoutPlan(plan.owner, plan.layouts, plan.assignments, forged, plan.canonical_id)
    with pytest.raises(ValueError, match="duplicate mapping requirements"):
        LayoutPlan(
            plan.owner, plan.layouts, plan.assignments,
            plan.mappings + (plan.mappings[0],), plan.canonical_id)


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
    assert uniform.native_spatial_layout is not None
    assert adaptive.native_spatial_layout is not None
    assert uniform.native_spatial_layout.dimension == 2
    assert uniform.native_spatial_layout.shape == uniform.geometry.cells
    assert uniform.native_spatial_layout.periodicity == (True, True)
    assert uniform.native_spatial_layout.decomposition["kind"] == "single_box"
    assert adaptive.native_spatial_layout.decomposition["kind"] == "adaptive"
    with pytest.raises(AttributeError):
        uniform.geometry.cells = (16, 16)


def test_native_spatial_layout_round_trip_and_identity_cover_every_spatial_fact():
    row = normalize_layout_plan(
        Uniform(cartesian_grid(n=8)), owner=OwnerPath.case("native-spatial")).layouts[0]
    native = row.native_spatial_layout
    assert native is not None
    assert NativeSpatialLayout.from_data(native.to_data()) == native

    data = native.to_data()
    data["periodicity"][0] = False
    data.pop("identity")
    changed = NativeSpatialLayout(
        layout_id=data["layout_id"],
        coordinate_system=data["coordinate_system"],
        cell_measure=data["cell_measure"],
        axis_names=tuple(data["axis_names"]),
        shape=tuple(data["shape"]),
        lower=tuple(float.fromhex(value) for value in data["lower"]),
        upper=tuple(float.fromhex(value) for value in data["upper"]),
        periodicity=tuple(data["periodicity"]),
        centering=data["centering"],
        decomposition=data["decomposition"],
    )
    assert changed.identity != native.identity

    forged = native.to_data()
    forged["dimension"] = 3
    with pytest.raises(ValueError, match="dimension does not match shape"):
        NativeSpatialLayout.from_data(forged)


def test_native_dimension_refuses_structurally_before_artifact_creation():
    class ThreeDimensionalLayout:
        name = "three-dimensional"

        def validate(self):
            return True

        def capabilities(self):
            return {"levels": 1, "supports_amr": False, "transition_ratios": []}

        def options(self):
            return {}

        def requirements(self):
            return {}

        def normalized_geometry(self):
            return NormalizedGeometry(
                "pops://coordinates/test-3d@1",
                "pops://cell-measures/test-volume@1",
                ("x", "y", "z"),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (4, 5, 6),
            )

        def native_spatial_data(self):
            return {
                "schema_version": 1,
                "periodicity": [True, False, True],
                "centering": "cell",
                "decomposition": {
                    "schema_version": 1,
                    "kind": "single_box",
                    "boxes": [{"lower": [0, 0, 0], "upper_exclusive": [4, 5, 6]}],
                },
            }

    plan = normalize_layout_plan(
        ThreeDimensionalLayout(), owner=OwnerPath.case("three-dimensional"))
    assert plan.layouts[0].native_spatial_layout.dimension == 3

    from pops.codegen._native_spatial_layout import (
        NativeSpatialLayoutError,
        native_spatial_layouts,
    )

    with pytest.raises(NativeSpatialLayoutError) as error:
        native_spatial_layouts(plan, supported_dimensions=(2,))
    assert error.value.code == "native_dimension_unavailable"
    assert error.value.evidence == {
        "resolved_dimension": 3,
        "supported_dimensions": [2],
    }


def test_non_cartesian_native_provider_refuses_before_artifact_creation():
    from pops.codegen._layout_resolution import (
        LayoutCapabilityError,
        resolve_native_spatial_layouts,
    )
    from pops.mesh import PolarMesh

    plan = normalize_layout_plan(
        Uniform(PolarMesh(0.2, 1.0, 8, 16)),
        owner=OwnerPath.case("polar-native-refusal"),
    )
    assert plan.layouts[0].geometry.coordinate_system == \
        "pops://coordinates/polar-annulus-2d@1"

    with pytest.raises(LayoutCapabilityError) as error:
        resolve_native_spatial_layouts(plan)
    assert error.value.evidence["gate"] == "native_coordinate_system_unavailable"
    assert error.value.evidence["refusal"]["evidence"] == {
        "resolved_coordinate_system": "pops://coordinates/polar-annulus-2d@1",
        "required_coordinate_system": "pops://coordinates/cartesian-2d@1",
        "resolved_dimension": 2,
    }


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
