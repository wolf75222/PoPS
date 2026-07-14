from __future__ import annotations

import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition, InitialConditionAuthorities
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant, Gaussian
from pops.mesh import LayoutPlanBuilder
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
from pops.params import Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import Clock


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
