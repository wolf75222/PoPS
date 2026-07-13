from dataclasses import FrozenInstanceError

import pytest

from pops.mesh import CartesianMesh, LayoutPlanBuilder
from pops.mesh.amr import (
    AnalyticReprojection,
    Above,
    BootstrapOrdering,
    BootstrapSelection,
    CanonicalOptions,
    ClusteringPolicy,
    ConflictPolicy,
    DerivedNestingRequirements,
    FrozenHierarchy,
    Hysteresis,
    HierarchyPlan,
    HierarchyProviderCapabilities,
    HierarchyResolutionContext,
    InitialConditionPlanBuilder,
    InitialConditionSource,
    LevelTransition,
    LoadBalancePolicy,
    NestingRequirementSource,
    EqualityPolicy,
    ProlongFromParent,
    PatchGenerationPolicy,
    TaggingGraph,
    resolve_bootstrap,
    resolve_hierarchy,
)
from pops.mesh.amr.transfer import (
    AccuracyRequirement,
    AMRTransfer,
    AMRTransferBuilder,
    ResolvedAMRTransfer,
    CACHE,
    CACHE_SPACE,
    CELL_CENTERED,
    CELL_SPACE,
    CONSERVATIVE_REPRESENTATION,
    COARSE_FINE_FILL,
    DENSE_STORAGE,
    DERIVED_FIELD,
    FACE_CENTERED,
    FACE_X_CENTERED,
    FACE_SPACE,
    FIELD_SPACE,
    InvalidateThenRebuild,
    MaterializationProvider,
    NODE_CENTERED,
    NODE_SPACE,
    PRIMITIVE_REPRESENTATION,
    PROLONGATION,
    RESTRICTION,
    Recompute,
    TransferCapabilities,
    TransferKey,
    TransferProvider,
    TransferProviderRoute,
)
from pops.lib.amr import (
    ConservativeLinear,
    EllipticRecompute,
    FaceTransfer,
    NodeTransfer,
    PatchTopologyRebuild,
    StateTransfer,
)
from pops.mesh.layouts import AMR
from pops.model import Handle, OwnerPath, ParamHandle
from pops.time import Clock
from pops.identity import make_identity
from pops.runtime._amr_bootstrap_execution import BootstrapReceipt, execute_bootstrap


OWNER = OwnerPath.case("transfer-bootstrap")


def _handle(name, kind):
    return Handle(name, kind=kind, owner=OWNER)


def _layout():
    state = Handle("U", kind="state", owner=OwnerPath.model("transport"))
    field = Handle("grad_u", kind="field", owner=OWNER)
    builder = LayoutPlanBuilder(OWNER)
    layout = builder.layout("adaptive", AMR(CartesianMesh(n=8), max_levels=2, ratio=2))
    builder.assign_state(state, layout)
    builder.assign_field(field, layout)
    plan = builder.resolve(states=(state,), fields=(field,))
    return plan, layout, state, field


def _key(operation=PROLONGATION, *, representation=CONSERVATIVE_REPRESENTATION):
    return TransferKey(
        CELL_SPACE,
        CELL_CENTERED,
        representation,
        DENSE_STORAGE,
        operation,
    )


def _provider(*keys, name="finite_volume_transfer", order=2, ghost=(2,), conservative=True):
    capabilities = TransferCapabilities(
        order=order,
        ghost_depth=ghost,
        dimensions=(2,),
        conservative=conservative,
    )
    return TransferProvider(
        _handle(name, "amr_transfer_provider"),
        tuple(TransferProviderRoute(key, capabilities) for key in keys),
    )


def _accuracy(*, conservative=False, order=1, ghost=(0,), temporal=False):
    return AccuracyRequirement(
        order=order,
        ghost_depth=ghost,
        dimension=2,
        refinement_ratio=(2, 2),
        conservative=conservative,
        temporal=temporal,
    )


def _resolved_transfer(*, include_field=True, include_cache=True):
    plan, layout, state, field = _layout()
    builder = AMRTransferBuilder(plan)
    key = _key()
    restriction_key = _key(RESTRICTION)
    builder.register(_provider(key, restriction_key))
    builder.require(state, key, accuracy=_accuracy(conservative=True))
    builder.require(
        state,
        restriction_key,
        accuracy=_accuracy(conservative=True),
    )
    if include_field:
        builder.require(
            field,
            TransferKey(
                FIELD_SPACE,
                CELL_CENTERED,
                PRIMITIVE_REPRESENTATION,
                DENSE_STORAGE,
                COARSE_FINE_FILL,
            ),
            materialization=DERIVED_FIELD,
            accuracy=_accuracy(),
            materializer=MaterializationProvider(
                _handle("elliptic", "field_operator"),
                DERIVED_FIELD,
                CanonicalOptions({"native_route": "elliptic_solve"}),
            ),
        )
    cache = _handle("reconstruction_cache", "cache")
    if include_cache:
        builder.require(
            cache,
            TransferKey(
                CACHE_SPACE,
                CELL_CENTERED,
                PRIMITIVE_REPRESENTATION,
                DENSE_STORAGE,
                COARSE_FINE_FILL,
            ),
            materialization=CACHE,
            accuracy=_accuracy(),
            layout=layout,
            materializer=MaterializationProvider(
                _handle("patch_topology", "cache_provider"),
                CACHE,
                CanonicalOptions({"native_route": "patch_topology"}),
            ),
        )
    return plan, layout, state, field, cache, builder.resolve()


def _source(role, buffer=(1, 1), lookahead=0):
    return NestingRequirementSource(
        _handle(role, "amr_%s_requirement" % role), buffer, lookahead
    )


def _hierarchy(transfer):
    nesting = DerivedNestingRequirements(
        stencil=_source("stencil"),
        transfer=transfer.nesting_requirement,
        reflux=_source("reflux"),
        boundary=_source("boundary"),
    )
    plan = HierarchyPlan(
        transitions=(LevelTransition(0, 1, (2, 2), nesting.minimum_buffer, 1),),
        nesting=nesting,
        clustering=ClusteringPolicy(
            _handle("cluster", "amr_clustering_provider"), CanonicalOptions()
        ),
        patch_generation=PatchGenerationPolicy(
            _handle("patches", "amr_patch_generation_provider"), CanonicalOptions()
        ),
        load_balance=LoadBalancePolicy(
            _handle("balance", "amr_load_balance_provider"), CanonicalOptions()
        ),
        regrid=FrozenHierarchy(),
    )
    provider = HierarchyProviderCapabilities(
        _handle("hierarchy", "amr_hierarchy_provider"),
        (2,),
        False,
        2,
        True,
        True,
    )
    return resolve_hierarchy(plan, provider, HierarchyResolutionContext(Clock("t", owner=OWNER)))


def _tagging(state):
    graph = TaggingGraph(
        refine=Above(
            state,
            ParamHandle("refine", owner=OWNER, param_kind="runtime"),
        ),
        coarsen=None,
        hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.ERROR,
    )
    return graph.resolve()


def test_exact_transfer_key_separates_representation_and_storage():
    conservative = _key()
    primitive = _key(representation=PRIMITIVE_REPRESENTATION)
    assert conservative.identity != primitive.identity
    assert conservative.to_data()["representation"]["name"] == "conservative"


def test_public_transfer_object_derives_all_state_routes_and_hides_internal_builders():
    plan, _, state, _ = _layout()
    authored = AMRTransfer()
    authored.state(state, StateTransfer())
    resolved = authored.resolve(plan)
    routes = {
        entry.key.operation.name:
            entry.action.route.options.to_data()["native_route"]
        for entry in resolved.entries
    }
    assert routes == {
        "prolongation": "conservative_linear",
        "restriction": "volume_average",
        "coarse_fine_fill": "conservative_coarse_fine",
        "temporal_interpolation": "linear_time_interpolation",
    }
    import pops.mesh.amr.transfer as module
    assert module.__all__ == ["AMRTransfer", "ResolvedAMRTransfer"]
    assert "AMRTransferBuilder" not in module.__all__


def test_builtin_policies_are_intrinsic_and_reject_duplicate_accuracy_knobs():
    assert ConservativeLinear().order == 2
    with pytest.raises(TypeError):
        ConservativeLinear(order=3)
    with pytest.raises(TypeError):
        StateTransfer(dimension=2)
    with pytest.raises(TypeError):
        FaceTransfer(ghost_depth=(2,))
    with pytest.raises(TypeError):
        NodeTransfer(conservative=True)


def test_public_provider_identity_is_stable_under_declaration_reordering():
    states = tuple(
        Handle(name, kind="state", owner=OwnerPath.model("stable-transfer"))
        for name in ("a", "b")
    )
    builder = LayoutPlanBuilder(OWNER)
    layout = builder.layout("stable", AMR(CartesianMesh(n=8), max_levels=2, ratio=2))
    for state in states:
        builder.assign_state(state, layout)
    plan = builder.resolve(states=states)

    forward = AMRTransfer()
    reverse = AMRTransfer()
    for state in states:
        forward.state(state, StateTransfer())
    for state in reversed(states):
        reverse.state(state, StateTransfer())
    resolved_forward = forward.resolve(plan)
    resolved_reverse = reverse.resolve(plan)
    assert resolved_forward.identity == resolved_reverse.identity
    assert {
        row.action.provider.qualified_id for row in resolved_forward.entries
    } == {
        row.action.provider.qualified_id for row in resolved_reverse.entries
    }


def test_field_and_cache_materializer_identities_are_owner_qualified_not_local_names():
    state = Handle("U", kind="state", owner=OwnerPath.model("materializer-state"))
    fields = (
        Handle("phi", kind="field", owner=OwnerPath.model("left")),
        Handle("phi", kind="field", owner=OwnerPath.model("right")),
    )
    caches = (
        Handle("topology", kind="cache", owner=OwnerPath.model("left")),
        Handle("topology", kind="cache", owner=OwnerPath.model("right")),
    )
    builder = LayoutPlanBuilder(OWNER)
    layout = builder.layout("qualified", AMR(CartesianMesh(n=8), max_levels=2, ratio=2))
    builder.assign_state(state, layout)
    for field in fields:
        builder.assign_field(field, layout)
    plan = builder.resolve(states=(state,), fields=fields)
    authored = AMRTransfer()
    authored.state(state, StateTransfer())
    for field in fields:
        authored.field(field, EllipticRecompute())
    for cache in caches:
        authored.cache(cache, PatchTopologyRebuild(), layout=layout)
    resolved = authored.resolve(plan)
    field_ids = {
        resolved.for_subject(field, COARSE_FINE_FILL).action.provider.qualified_id
        for field in fields
    }
    cache_ids = {
        resolved.for_subject(cache, COARSE_FINE_FILL).action.provider.qualified_id
        for cache in caches
    }
    assert len(field_ids) == 2 and len(cache_ids) == 2
    assert all(
        value.startswith("pops.handle.v1::%s::" % plan.owner)
        for value in field_ids | cache_ids
    )


def test_cell_face_node_states_use_distinct_providers_and_exact_initial_sources():
    states = tuple(
        Handle(name, kind="state", owner=OwnerPath.model("multi-space"))
        for name in ("cell_state", "face_state", "node_state")
    )
    layout_builder = LayoutPlanBuilder(OWNER)
    layout = layout_builder.layout(
        "multi_space", AMR(CartesianMesh(n=8), max_levels=2, ratio=2)
    )
    for state in states:
        layout_builder.assign_state(state, layout)
    plan = layout_builder.resolve(states=states)
    keys = (
        TransferKey(
            CELL_SPACE,
            CELL_CENTERED,
            CONSERVATIVE_REPRESENTATION,
            DENSE_STORAGE,
            PROLONGATION,
        ),
        TransferKey(
            FACE_SPACE,
            FACE_X_CENTERED,
            PRIMITIVE_REPRESENTATION,
            DENSE_STORAGE,
            PROLONGATION,
        ),
        TransferKey(
            NODE_SPACE,
            NODE_CENTERED,
            PRIMITIVE_REPRESENTATION,
            DENSE_STORAGE,
            PROLONGATION,
        ),
    )
    transfer_builder = AMRTransferBuilder(plan)
    for name, state, key in zip(("cell", "face", "node"), states, keys, strict=True):
        provider_keys = (key, _key(RESTRICTION)) if name == "cell" else (key,)
        transfer_builder.register(
            _provider(
                *provider_keys,
                name="%s_transfer" % name,
                conservative=name == "cell",
            )
        )
        transfer_builder.require(
            state,
            key,
            accuracy=_accuracy(conservative=name == "cell"),
            layout=layout,
        )
        if name == "cell":
            transfer_builder.require(
                state,
                _key(RESTRICTION),
                accuracy=_accuracy(conservative=True),
                layout=layout,
            )
    transfer = transfer_builder.resolve()
    assert {
        transfer.for_subject(state, PROLONGATION).action.provider.provider.local_id
        for state in states
    } == {"cell_transfer", "face_transfer", "node_transfer"}
    initial_builder = InitialConditionPlanBuilder(plan, transfer)
    for name, state in zip(("cell", "face", "node"), states, strict=True):
        initial_builder.add(
            state,
            InitialConditionSource(_handle("%s_ic" % name, "initial_condition_provider")),
            layout=layout,
        )
    initial = initial_builder.resolve()
    bootstrap = resolve_bootstrap(
        layout_plan=plan,
        hierarchy=_hierarchy(transfer),
        transfers=transfer,
        initial_conditions=initial,
        tagging=_tagging(states[0]),
        selections=tuple(BootstrapSelection(state, ProlongFromParent()) for state in states),
        ordering=BootstrapOrdering(("transfer", "projection", "constraint")),
    )
    assert sum(row.operation == "prolong_from_parent" for row in bootstrap.actions) == 3


def test_resolution_is_immutable_exact_and_derives_nesting():
    _, _, state, _, _, transfer = _resolved_transfer()
    assert isinstance(transfer, ResolvedAMRTransfer)
    assert transfer.nesting_requirement.minimum_buffer == (2, 2)
    assert transfer.nesting_requirement.minimum_lookahead == 1
    assert transfer.for_subject(state, PROLONGATION).action.capabilities.order == 2
    with pytest.raises(FrozenInstanceError):
        transfer.entries = ()
    provider_data = transfer.for_subject(state, PROLONGATION).action.to_data()["provider"]
    provider_data["routes"].clear()
    assert transfer.for_subject(state, PROLONGATION).action.to_data()["provider"]["routes"]


def test_registry_refuses_empty_missing_incompatible_ambiguous_and_unused():
    plan, _, state, _ = _layout()
    with pytest.raises(ValueError, match="requirement manifest"):
        AMRTransferBuilder(plan).resolve()
    missing = AMRTransferBuilder(plan)
    missing.require(state, _key(), accuracy=_accuracy(conservative=True))
    with pytest.raises(ValueError, match="missing AMR transfer provider"):
        missing.resolve()
    incompatible = AMRTransferBuilder(plan)
    incompatible.register(_provider(_key(), conservative=False))
    incompatible.require(state, _key(), accuracy=_accuracy(conservative=True))
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.resolve()
    ambiguous = AMRTransferBuilder(plan)
    ambiguous.register(_provider(_key()))
    other = TransferProvider(
        _handle("other", "amr_transfer_provider"), _provider(_key()).routes
    )
    ambiguous.register(other)
    ambiguous.require(state, _key(), accuracy=_accuracy(conservative=True))
    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous.resolve()
    unused = AMRTransferBuilder(plan)
    unused.register(_provider(_key()))
    unused.register(
        TransferProvider(
            _handle("unused", "amr_transfer_provider"),
            _provider(_key(representation=PRIMITIVE_REPRESENTATION)).routes,
        )
    )
    unused.require(state, _key(), accuracy=_accuracy(conservative=True))
    with pytest.raises(ValueError, match="unused AMR transfer provider"):
        unused.resolve()


def test_derived_fields_recompute_and_caches_invalidate_then_rebuild():
    _, _, _, field, cache, transfer = _resolved_transfer()
    assert isinstance(transfer.for_subject(field, COARSE_FINE_FILL).action, Recompute)
    assert isinstance(
        transfer.for_subject(cache, COARSE_FINE_FILL).action, InvalidateThenRebuild
    )


def test_initial_condition_manifest_exactly_covers_physical_subjects():
    plan, layout, state, field, _, transfer = _resolved_transfer()
    builder = InitialConditionPlanBuilder(plan, transfer)
    with pytest.raises(ValueError, match="physical state/particle"):
        builder.add(field, InitialConditionSource(_handle("field_ic", "initial_condition_provider")))
    with pytest.raises(ValueError, match="missing physical subjects"):
        builder.resolve()
    builder.add(
        state,
        InitialConditionSource(_handle("state_ic", "initial_condition_provider")),
        layout=layout,
    )
    initial = builder.resolve()
    assert initial.transfer_identity == transfer.identity


def test_bootstrap_orders_level_zero_and_recursive_materialization_explicitly():
    plan, layout, state, field, cache, transfer = _resolved_transfer()
    initial_builder = InitialConditionPlanBuilder(plan, transfer)
    initial_builder.add(
        state,
        InitialConditionSource(_handle("state_ic", "initial_condition_provider")),
        layout=layout,
    )
    initial = initial_builder.resolve()
    hierarchy = _hierarchy(transfer)
    bootstrap = resolve_bootstrap(
        layout_plan=plan,
        hierarchy=hierarchy,
        transfers=transfer,
        initial_conditions=initial,
        tagging=_tagging(state),
        selections=(BootstrapSelection(state, ProlongFromParent()),),
        ordering=BootstrapOrdering(("transfer", "projection", "constraint")),
    )
    operations = [(row.level, row.phase, row.operation, row.subject_id) for row in bootstrap.actions]
    assert operations[0][:3] == (0, "initial_condition", "initialize_level_zero")
    assert (1, "hierarchy", "tag_parent", None) in operations
    assert (1, "transfer", "prolong_from_parent", state.qualified_id) in operations
    assert (1, "projection", "recompute", field.qualified_id) in operations
    assert (1, "transfer", "invalidate_cache", cache.qualified_id) in operations
    assert (1, "projection", "rebuild_cache", cache.qualified_id) in operations
    changed = resolve_bootstrap(
        layout_plan=plan,
        hierarchy=hierarchy,
        transfers=transfer,
        initial_conditions=initial,
        tagging=_tagging(state),
        selections=(BootstrapSelection(state, AnalyticReprojection()),),
        ordering=BootstrapOrdering(("projection", "transfer", "constraint")),
    )
    assert changed.identity != bootstrap.identity


def test_three_level_bootstrap_is_one_explicit_recursive_plan():
    state = Handle("U", kind="state", owner=OwnerPath.model("three-level"))
    layout_builder = LayoutPlanBuilder(OWNER)
    layout = layout_builder.layout(
        "three_levels", AMR(CartesianMesh(n=8), max_levels=3, ratio=2)
    )
    layout_builder.assign_state(state, layout)
    layout_plan = layout_builder.resolve(states=(state,))
    authored = AMRTransfer()
    authored.state(state, StateTransfer())
    transfer = authored.resolve(layout_plan)
    nesting = DerivedNestingRequirements(
        stencil=_source("stencil"), transfer=transfer.nesting_requirement,
        reflux=_source("reflux"), boundary=_source("boundary"),
    )
    hierarchy_plan = HierarchyPlan(
        transitions=(
            LevelTransition(0, 1, (2, 2), nesting.minimum_buffer, 1),
            LevelTransition(1, 2, (2, 2), nesting.minimum_buffer, 1),
        ),
        nesting=nesting,
        clustering=ClusteringPolicy(
            _handle("cluster3", "amr_clustering_provider"), CanonicalOptions()
        ),
        patch_generation=PatchGenerationPolicy(
            _handle("patches3", "amr_patch_generation_provider"), CanonicalOptions()
        ),
        load_balance=LoadBalancePolicy(
            _handle("balance3", "amr_load_balance_provider"), CanonicalOptions()
        ),
        regrid=FrozenHierarchy(),
    )
    hierarchy = resolve_hierarchy(
        hierarchy_plan,
        HierarchyProviderCapabilities(
            _handle("hierarchy3", "amr_hierarchy_provider"), (2,), False, 3, True, True
        ),
        HierarchyResolutionContext(Clock("t3", owner=OWNER)),
    )
    initial_builder = InitialConditionPlanBuilder(layout_plan, transfer)
    initial_builder.add(
        state, InitialConditionSource(_handle("state3_ic", "initial_condition_provider")),
        layout=layout,
    )
    bootstrap = resolve_bootstrap(
        layout_plan=layout_plan,
        hierarchy=hierarchy,
        transfers=transfer,
        initial_conditions=initial_builder.resolve(),
        tagging=_tagging(state),
        selections=(BootstrapSelection(state, ProlongFromParent()),),
        ordering=BootstrapOrdering(("transfer", "projection", "constraint")),
    )
    assert [row.level for row in bootstrap.actions if row.operation == "create_level"] == [1, 2]
    assert [row.level for row in bootstrap.actions if row.operation == "prolong_from_parent"] == [1, 2]
    assert [row.level for row in bootstrap.actions if row.operation == "synchronize_covered_cells"] == [1, 2]

    failure_indices = (
        0,
        next(i for i, row in enumerate(bootstrap.actions)
             if row.operation == "create_level" and row.level == 1) + 1,
        next(i for i, row in enumerate(bootstrap.actions)
             if row.operation == "create_level" and row.level == 2) + 1,
    )
    for failure_index in failure_indices:
        class FailingConsumer:
            bootstrap_consumer_identity = make_identity(
                "three-level-rollback-consumer", {"failure_index": failure_index}
            )

            def __init__(self):
                self.state = {"levels": [0], "events": []}
                self.snapshot = {"levels": [0], "events": []}

            def consume_bootstrap_action(self, action):
                self.state["events"].append(action.identity.token)
                if action.operation == "create_level":
                    self.state["levels"].append(action.level)
                if len(self.state["events"]) - 1 == failure_index:
                    raise RuntimeError("injected late bootstrap failure")
                return BootstrapReceipt(
                    action.identity, self.bootstrap_consumer_identity,
                    {"operation": action.operation},
                )

            def abort_bootstrap(self):
                self.state = {
                    "levels": list(self.snapshot["levels"]),
                    "events": list(self.snapshot["events"]),
                }

        consumer = FailingConsumer()
        with pytest.raises(RuntimeError, match="injected late bootstrap failure"):
            execute_bootstrap(bootstrap, consumer)
        assert consumer.state == consumer.snapshot


def test_runtime_executor_consumes_every_transfer_projection_and_cache_action():
    plan, layout, state, field, cache, transfer = _resolved_transfer()
    initial_builder = InitialConditionPlanBuilder(plan, transfer)
    initial_builder.add(
        state,
        InitialConditionSource(_handle("state_ic", "initial_condition_provider")),
        layout=layout,
    )
    bootstrap = resolve_bootstrap(
        layout_plan=plan,
        hierarchy=_hierarchy(transfer),
        transfers=transfer,
        initial_conditions=initial_builder.resolve(),
        tagging=_tagging(state),
        selections=(BootstrapSelection(state, ProlongFromParent()),),
        ordering=BootstrapOrdering(("transfer", "projection", "constraint")),
    )

    class Consumer:
        bootstrap_consumer_identity = make_identity("test-bootstrap-consumer", {"v": 1})

        def __init__(self):
            self.levels = {0}
            self.transferred = set()
            self.fields = set()
            self.caches = {}

        def consume_bootstrap_action(self, action):
            if action.operation == "create_level":
                self.levels.add(action.level)
            elif action.operation in {"prolong_from_parent", "apply_transfer_provider"}:
                self.transferred.add((action.level, action.subject_id))
            elif action.operation == "recompute":
                self.fields.add((action.level, action.subject_id))
            elif action.operation == "invalidate_cache":
                self.caches[(action.level, action.subject_id)] = False
            elif action.operation in {"rebuild_cache", "invalidate_then_rebuild"}:
                self.caches[(action.level, action.subject_id)] = True
            return BootstrapReceipt(
                action.identity,
                self.bootstrap_consumer_identity,
                {"operation": action.operation, "level": action.level},
            )

    consumer = Consumer()
    execution = execute_bootstrap(bootstrap, consumer)
    assert len(execution.receipts) == len(bootstrap.actions)
    assert consumer.levels == {0, 1}
    assert (1, state.qualified_id) in consumer.transferred
    assert (1, field.qualified_id) in consumer.fields
    assert consumer.caches[(1, cache.qualified_id)] is True


def test_bootstrap_rejects_hierarchy_not_derived_from_transfer_registry():
    plan, layout, state, _, _, transfer = _resolved_transfer()
    initial_builder = InitialConditionPlanBuilder(plan, transfer)
    initial_builder.add(
        state,
        InitialConditionSource(_handle("state_ic", "initial_condition_provider")),
        layout=layout,
    )
    hierarchy = _hierarchy(transfer)
    changed_nesting = DerivedNestingRequirements(
        stencil=hierarchy.plan.nesting.stencil,
        transfer=_source("transfer", (2, 2), 1),
        reflux=hierarchy.plan.nesting.reflux,
        boundary=hierarchy.plan.nesting.boundary,
    )
    changed_plan = HierarchyPlan(
        hierarchy.plan.transitions,
        changed_nesting,
        hierarchy.plan.clustering,
        hierarchy.plan.patch_generation,
        hierarchy.plan.load_balance,
        hierarchy.plan.regrid,
    )
    changed = resolve_hierarchy(
        changed_plan,
        hierarchy.provider,
        HierarchyResolutionContext(Clock("t", owner=OWNER)),
    )
    with pytest.raises(ValueError, match="derived from the AMRTransfer"):
        resolve_bootstrap(
            layout_plan=plan,
            hierarchy=changed,
            transfers=transfer,
            initial_conditions=initial_builder.resolve(),
            tagging=_tagging(state),
            selections=(BootstrapSelection(state, ProlongFromParent()),),
            ordering=BootstrapOrdering(("transfer", "projection", "constraint")),
        )
