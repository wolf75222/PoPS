from __future__ import annotations

import pytest

import pops
from pops.analytic import angle, between, coordinates, param, radius, sin, where
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import (
    InitialCondition,
    InitialConditionAuthorities,
    InitialConditionPlanBuilder,
    InitialConditionSource,
)
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic, BindArray, Constant, Gaussian
from pops.layouts import Uniform
from pops.mesh import LayoutPlanBuilder
from pops.mesh import CartesianGrid
from pops.mesh._amr import (
    Above,
    AMRTransfer,
    CanonicalOptions,
    ClusteringPolicy,
    ConflictPolicy,
    DerivedNestingRequirements,
    EqualityPolicy,
    FrozenHierarchy,
    Hysteresis,
    HierarchyPlan,
    HierarchyProviderCapabilities,
    HierarchyResolutionContext,
    LevelTransition,
    LoadBalancePolicy,
    NestingRequirementSource,
    PatchGenerationPolicy,
    TaggingGraph,
    resolve_hierarchy,
)
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout
from pops.model import Handle
from pops.params import ConstParam, Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import Clock
from pops.codegen._plans import _canonicalize_initial_value_mapping


def _case(*, components=("u",)):
    domain = Rectangle("unit", (0.0, 0.0), (1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state(
        "U",
        components=components,
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    case = pops.Case("initial-test")
    block = case.block("tracer", model)
    threshold = case.param(RuntimeParam("refine", default=0.1, domain=Positive()))
    return case, frame, state, block[state], threshold


def _source(owner, role, buffer=(1, 1), lookahead=0):
    return NestingRequirementSource(
        Handle(role, kind="amr_%s_requirement" % role, owner=owner),
        buffer,
        lookahead,
    )


def _resolved_amr_authorities(case, state, threshold):
    owner = case.owner_path.canonical()
    state = case.resolve(state)
    threshold = case.resolve(threshold)

    builder = LayoutPlanBuilder(owner)
    layout = builder.layout(
        "adaptive", final_amr_layout(cartesian_grid(n=8), max_levels=2, ratio=2))
    builder.assign_state(state, layout)
    layout_plan = builder.resolve(states=(state,))

    transfer_authoring = AMRTransfer()
    transfer_authoring.state(state, StateTransfer())
    transfers = transfer_authoring.resolve(layout_plan)

    nesting = DerivedNestingRequirements(
        stencil=_source(owner, "stencil"),
        transfer=transfers.nesting_requirement,
        reflux=_source(owner, "reflux"),
        boundary=_source(owner, "boundary"),
    )
    hierarchy_plan = HierarchyPlan(
        transitions=(LevelTransition(
            0, 1, (2, 2), nesting.minimum_buffer, nesting.minimum_lookahead),),
        nesting=nesting,
        clustering=ClusteringPolicy(
            Handle("cluster", kind="amr_clustering_provider", owner=owner),
            CanonicalOptions(),
        ),
        patch_generation=PatchGenerationPolicy(
            Handle("patches", kind="amr_patch_generation_provider", owner=owner),
            CanonicalOptions(),
        ),
        load_balance=LoadBalancePolicy(
            Handle("balance", kind="amr_load_balance_provider", owner=owner),
            CanonicalOptions(),
        ),
        regrid=FrozenHierarchy(),
    )
    hierarchy = resolve_hierarchy(
        hierarchy_plan,
        HierarchyProviderCapabilities(
            Handle("hierarchy", kind="amr_hierarchy_provider", owner=owner),
            (2,),
            False,
            2,
            True,
            True,
        ),
        HierarchyResolutionContext(Clock("macro", owner=owner)),
    )
    tagging = TaggingGraph(
        refine=Above(state, threshold),
        coarsen=None,
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.ERROR,
    ).resolve()
    return layout_plan, transfers, hierarchy, tagging


def test_case_initials_is_unique_freezable_and_in_snapshot():
    case, _, state, qualified, _ = _case()
    initial = InitialCondition(
        state=qualified,
        value=Constant((0.25,)),
        projection=ConservativeCellAverage(),
    )

    assert case.initials.add(initial) is initial
    with pytest.raises(ValueError, match="already declared"):
        case.initials.add(initial)

    validated = pops.validate(case)
    assert validated is case and case.frozen
    assert case.options()["n_initials"] == 1
    assert pops.inspect(case)["initials"][0]["value"]["profile"] == "constant"
    semantic_initial, = case.snapshot.semantic_to_dict()["initials"]
    assert semantic_initial["state"] == case.resolve(qualified).canonical_identity()
    assert semantic_initial["value"]["profile"] == "constant"
    with pytest.raises(RuntimeError, match="frozen"):
        case.initials.add(initial)

    with pytest.raises(TypeError, match="block-qualified"):
        InitialCondition(
            state=state,
            value=Constant((0.25,)),
            projection=ConservativeCellAverage(),
        )


def test_initial_registry_derives_exact_amr_initial_and_bootstrap_plans():
    case, _, _, state, threshold = _case()
    case.initials.add(InitialCondition(
        state=state,
        value=Constant((0.25,)),
        projection=ConservativeCellAverage(),
    ))
    pops.validate(case)
    layout_plan, transfers, hierarchy, tagging = _resolved_amr_authorities(
        case, state, threshold)

    authorities = case.initials.resolve_amr(
        layout_plan=layout_plan,
        transfers=transfers,
        hierarchy=hierarchy,
        tagging=tagging,
    )

    assert type(authorities) is InitialConditionAuthorities
    binding, = authorities.initial_condition_plan.bindings
    assert binding.subject == case.resolve(state)
    options = binding.source.options.to_data()
    assert options["native_route"] == "constant_field"
    assert options["projection"]["formal_order"] == 2
    assert any(
        action.operation == "analytic_reprojection"
        for action in authorities.bootstrap_plan.actions
    )


def test_initial_registry_resolves_a_layout_generic_uniform_plan():
    case, _, _, state, _ = _case()
    case.initials.add(InitialCondition(
        state=state,
        value=Constant((0.25,)),
        projection=ConservativeCellAverage(),
    ))
    pops.validate(case)
    canonical = case.resolve(state)

    builder = LayoutPlanBuilder(case.owner_path.canonical())
    layout = builder.layout("uniform", Uniform(cartesian_grid(n=8)))
    builder.assign_state(canonical, layout)
    layout_plan = builder.resolve(states=(canonical,))

    initial_plan = case.initials.resolve_plan(
        layout_plan=layout_plan,
        expected_subjects=(canonical,),
    )

    binding, = initial_plan.bindings
    assert binding.subject == canonical
    assert binding.layout == layout
    assert initial_plan.identity.domain == "initial-condition-plan"
    assert not hasattr(initial_plan, "transfer_identity")


def test_initial_plan_rejects_an_explicit_layout_other_than_the_subject_assignment():
    case, _, _, state, _ = _case()
    initial = InitialCondition(
        state=state,
        value=Constant((0.25,)),
        projection=ConservativeCellAverage(),
    )
    case.initials.add(initial)
    pops.validate(case)
    canonical = case.resolve(state)

    layouts = LayoutPlanBuilder(case.owner_path.canonical())
    assigned = layouts.layout("assigned", Uniform(cartesian_grid(n=8)))
    other = layouts.layout("other", Uniform(cartesian_grid(n=16)))
    layouts.assign_state(canonical, assigned)
    layout_plan = layouts.resolve(states=(canonical,))

    initial_plan = InitialConditionPlanBuilder(layout_plan, (canonical,))
    source = initial.resolve_references(case.resolve).source(case.owner_path)
    with pytest.raises(ValueError, match="differs from the subject's LayoutPlan assignment"):
        initial_plan.add(canonical, source, layout=other)


def test_frame_bound_initial_refuses_a_layout_on_another_physical_frame():
    case, frame, _, state, _ = _case()
    profile = Analytic(frame=frame, components=(radius(frame),))
    case.initials.add(InitialCondition(
        state=state,
        value=profile,
        projection=ConservativeCellAverage(),
    ))
    pops.validate(case)
    canonical = case.resolve(state)

    other_frame = Rectangle("other-domain", (0.0, 0.0), (2.0, 1.0)).frame(Cartesian2D())
    builder = LayoutPlanBuilder(case.owner_path.canonical())
    layout = builder.layout(
        "uniform",
        Uniform(CartesianGrid(frame=other_frame, cells=(8, 8))),
    )
    builder.assign_state(canonical, layout)
    layout_plan = builder.resolve(states=(canonical,))

    with pytest.raises(ValueError, match="assigned layout frame"):
        case.initials.resolve_plan(
            layout_plan=layout_plan,
            expected_subjects=(canonical,),
        )


def test_bind_array_authors_full_level_zero_value_and_transfer_bootstrap():
    case, _, _, state, threshold = _case(components=("rho", "rho_u", "rho_v"))
    profile = BindArray()
    case.initials.add(InitialCondition(
        state=state,
        value=profile,
        projection=ConservativeCellAverage(),
    ))
    pops.validate(case)
    layout_plan, transfers, hierarchy, tagging = _resolved_amr_authorities(
        case, state, threshold)

    authorities = case.initials.resolve_amr(
        layout_plan=layout_plan,
        transfers=transfers,
        hierarchy=hierarchy,
        tagging=tagging,
    )

    binding, = authorities.initial_condition_plan.bindings
    selection, = authorities.bootstrap_plan.selections
    assert binding.source.options.to_data()["native_route"] == "bound_level_zero"
    assert selection.method.to_data() == {"method": "prolongation"}
    assert profile.to_data() == {"schema_version": 1, "profile": "bind_array"}

    canonical = case.resolve(state)
    authored_payload = object()
    assert _canonicalize_initial_value_mapping(
        authorities.initial_condition_plan, {state: authored_payload}
    ) == {canonical: authored_payload}
    canonical_payload = object()
    assert _canonicalize_initial_value_mapping(
        authorities.initial_condition_plan, {canonical: canonical_payload}
    ) == {canonical: canonical_payload}

    foreign_case, _, _, foreign_state, _ = _case(
        components=("rho", "rho_u", "rho_v"))
    assert foreign_case is not case
    with pytest.raises(KeyError, match="not an authenticated subject or authoring alias"):
        _canonicalize_initial_value_mapping(
            authorities.initial_condition_plan, {foreign_state: object()})
    with pytest.raises(ValueError, match="multiple aliases"):
        _canonicalize_initial_value_mapping(
            authorities.initial_condition_plan,
            {state: object(), canonical: object()},
        )


def test_gaussian_is_typed_frame_bound_and_scalar():
    case, frame, _, state, _ = _case()
    profile = Gaussian(
        frame=frame,
        center={frame.x: 0.3, frame.y: 0.35},
        background=0.05,
        amplitude=0.95,
        inverse_width=120.0,
    )
    initial = InitialCondition(
        state=state,
        value=profile,
        projection=ConservativeCellAverage(),
    )
    assert initial.value.initial_source_options()["native_route"] == "gaussian_field"
    assert initial.projection.formal_order == 2

    other_case, _, _, vector_state, _ = _case(components=("u", "v"))
    with pytest.raises(ValueError, match="one-component"):
        InitialCondition(
            state=vector_state,
            value=profile,
            projection=ConservativeCellAverage(),
        )
    assert other_case is not case


def test_analytic_profile_is_frame_bound_ordered_and_callback_free():
    case, frame, _, state, _ = _case()
    radial_coordinate = radius(frame)
    angular_coordinate = angle(frame)
    density = where(
        between(radial_coordinate, 0.35, 0.40),
        0.9 + 0.1 * sin(4.0 * angular_coordinate),
        1.0e-4,
    )
    profile = Analytic(frame=frame, components=(density,))
    initial = InitialCondition(
        state=state,
        value=profile,
        projection=ConservativeCellAverage(),
    )

    options = initial.value.initial_source_options()
    assert options["native_route"] == "analytic_expression"
    assert options["frame_id"] == frame.canonical_id
    assert options["components"][0]["expression_type"] == "scalar"
    assert profile.reprojectable is True

    other_frame = Rectangle("other", (-1.0, -1.0), (1.0, 1.0)).frame(Cartesian2D())
    other_x, _ = coordinates(other_frame)
    with pytest.raises(ValueError, match="another physical frame"):
        Analytic(frame=frame, components=(other_x,))
    with pytest.raises(TypeError, match="ScalarExpr"):
        Analytic(frame=frame, components=(lambda x: x,))


def test_analytic_profile_recaptures_resolved_parameter_references() -> None:
    case, frame, _, state, runtime_value = _case()
    offset = case.param(ConstParam("offset", 0.25))
    x_coord, _ = coordinates(frame)
    profile = Analytic(
        frame=frame,
        components=(param(runtime_value) + param(offset) * x_coord,),
    )
    initial = InitialCondition(
        state=state,
        value=profile,
        projection=ConservativeCellAverage(),
    )
    authored_reference = initial.inspect()["value"]["components"][0]["root"][
        "arguments"
    ][0]["reference"]
    assert authored_reference["ownership_phase"] == "authoring"

    resolved = initial.resolve_references(case.resolve)
    source = resolved.source(case.owner_path).options.to_data()
    repeated = initial.resolve_references(case.resolve)
    assert repeated.canonical_identity() == resolved.canonical_identity()
    assert repeated.source(case.owner_path).options.to_data() == source
    references = (
        source["components"][0]["root"]["arguments"][0]["reference"],
        source["components"][0]["root"]["arguments"][1]["arguments"][0]["reference"],
    )
    assert all("ownership_phase" not in reference for reference in references)
    assert {reference["param_kind"] for reference in references} == {"runtime", "const"}


def test_initial_condition_rejects_incomplete_or_unstable_extension_providers():
    _, _, _, state, _ = _case()

    class CompleteValue:
        reprojectable = True

        def validate_for(self, target):
            assert target is state
            return True

        def initial_source_options(self):
            return {"native_route": "test_field"}

        def to_data(self):
            return {"schema_version": 1, "profile": "test"}

        def canonical_identity(self):
            return self.to_data()

    class CompleteProjection:
        bootstrap_phases = ("transfer", "projection", "constraint")

        def validate_for(self, target, value):
            assert target is state
            assert isinstance(value, CompleteValue)
            return True

        def initial_projection_options(self):
            return {"projection": {"schema_version": 1, "provider": "test"}}

        def to_data(self):
            return {"schema_version": 1, "projection": "test"}

        def canonical_identity(self):
            return self.to_data()

    class MissingDataValue(CompleteValue):
        to_data = None

        def canonical_identity(self):
            return {"schema_version": 1, "profile": "test"}

    class NondeterministicValue(CompleteValue):
        def __init__(self):
            self.calls = 0

        def initial_source_options(self):
            self.calls += 1
            return {"native_route": "test_field", "call": self.calls}

    class MismatchedProjection(CompleteProjection):
        def canonical_identity(self):
            return {"schema_version": 1, "projection": "another-test"}

    class InvalidBootstrapProjection(CompleteProjection):
        bootstrap_phases = ("transfer", "projection")

    with pytest.raises(TypeError, match=r"value.*to_data\(\)"):
        InitialCondition(state, MissingDataValue(), CompleteProjection())
    with pytest.raises(TypeError, match=r"initial_source_options\(\).*deterministic"):
        InitialCondition(state, NondeterministicValue(), CompleteProjection())
    with pytest.raises(ValueError, match=r"projection canonical_identity\(\).*match"):
        InitialCondition(state, CompleteValue(), MismatchedProjection())
    with pytest.raises(ValueError, match="bootstrap_phases"):
        InitialCondition(state, CompleteValue(), InvalidBootstrapProjection())


def test_initial_condition_accepts_a_complete_structural_extension_provider():
    _, _, _, state, _ = _case()

    class Value:
        reprojectable = False

        def validate_for(self, target):
            assert target is state
            return True

        def initial_source_options(self):
            return {"native_route": "test_bound_field"}

        def to_data(self):
            return {"schema_version": 1, "profile": "test-bound"}

        canonical_identity = to_data

    class Projection:
        bootstrap_phases = ("transfer", "projection", "constraint")

        def validate_for(self, target, value):
            assert target is state
            assert isinstance(value, Value)
            return True

        def initial_projection_options(self):
            return {"projection": {"schema_version": 1, "provider": "test"}}

        def to_data(self):
            return {"schema_version": 1, "projection": "test"}

        canonical_identity = to_data

    initial = InitialCondition(state, Value(), Projection())

    assert initial.bootstrap_phases == ("transfer", "projection", "constraint")
    assert initial.bootstrap_method().to_data() == {"method": "prolongation"}


def test_initial_resolution_uses_only_the_captured_provider_snapshot() -> None:
    case, _, _, state, _ = _case()

    class MutableValue:
        reprojectable = True

        def __init__(self) -> None:
            self.marker = 1

        def validate_for(self, target):
            assert target is state
            return True

        def initial_source_options(self):
            return {"native_route": "captured-test", "marker": self.marker}

        def to_data(self):
            return {"schema_version": 1, "profile": "captured-test", "marker": self.marker}

        canonical_identity = to_data

    class MutableProjection:
        bootstrap_phases = ("transfer", "projection", "constraint")

        def __init__(self) -> None:
            self.marker = 10

        def validate_for(self, target, value):
            assert target is state
            assert isinstance(value, MutableValue)
            return True

        def initial_projection_options(self):
            return {"projection": {"schema_version": 1, "marker": self.marker}}

        def to_data(self):
            return {"schema_version": 1, "projection": "captured-test", "marker": self.marker}

        canonical_identity = to_data

    value = MutableValue()
    projection = MutableProjection()
    initial = InitialCondition(state, value, projection)
    value.marker = 2
    projection.marker = 20

    resolved = initial.resolve_references(case.resolve)

    assert initial.inspect()["value"]["marker"] == 1
    assert resolved.inspect()["value"]["marker"] == 1
    assert resolved.inspect()["projection"]["marker"] == 10
    assert resolved.source(case.owner_path).options.to_data()["marker"] == 1
    assert resolved.source(case.owner_path).options.to_data()["projection"]["marker"] == 10


def test_initial_provider_with_only_live_reference_resolution_is_refused() -> None:
    _, _, _, state, _ = _case()

    class UnsafeValue:
        reprojectable = True

        def validate_for(self, target):
            assert target is state
            return True

        def initial_source_options(self):
            return {"native_route": "unsafe-test"}

        def to_data(self):
            return {"schema_version": 1, "profile": "unsafe-test"}

        canonical_identity = to_data

        def resolve_references(self, resolver):
            del resolver
            return self

    class Projection:
        bootstrap_phases = ("transfer", "projection", "constraint")

        def validate_for(self, target, value):
            assert target is state
            assert isinstance(value, UnsafeValue)
            return True

        def initial_projection_options(self):
            return {"projection": {"schema_version": 1, "provider": "test"}}

        def to_data(self):
            return {"schema_version": 1, "projection": "test"}

        canonical_identity = to_data

    with pytest.raises(TypeError, match="unsafe after capture"):
        InitialCondition(state, UnsafeValue(), Projection())


def test_initial_source_rejects_a_duck_typed_fake_handle() -> None:
    class FakeHandle:
        is_resolved = True
        kind = "initial_condition_provider"
        qualified_id = "pops.fake-provider"

        def canonical_identity(self):
            return {"kind": self.kind, "qualified_id": self.qualified_id}

        def __hash__(self):
            return 1

    with pytest.raises(TypeError, match="canonical owner-qualified Handle"):
        InitialConditionSource(FakeHandle())
