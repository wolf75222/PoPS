"""Exact AMR authority cross-checks and current native lowering boundary."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _validated_native_materialization(entry: Any) -> Any:
    """Return the closed native IR of an open AMR action, never its Python class."""

    from pops.mesh._amr.transfer import NativeAMRMaterializationDescriptor

    native = getattr(entry, "native_materialization", None)
    if type(native) is not NativeAMRMaterializationDescriptor:
        raise TypeError(
            "resolved AMR transfer entry must carry an exact prepared native descriptor"
        )
    if native.transfer_key_identity != entry.key.identity \
            or native.operation != entry.key.operation:
        raise ValueError("prepared native AMR descriptor authenticates another transfer key")
    if {row.materialization for row in entry.requirements} != {
        native.materialization.value
    }:
        raise ValueError(
            "prepared native AMR descriptor disagrees with requirement materialization"
        )
    return native


def validate_amr_authorities(plan: Any) -> None:
    from pops.initial import InitialConditionPlan

    initial_plan = plan.initial_condition_plan
    if initial_plan is not None:
        if type(initial_plan) is not InitialConditionPlan:
            raise TypeError(
                "ResolvedSimulationPlan contains a non-exact InitialConditionPlan")
        if initial_plan.layout_plan_id != plan.layout_plan.qualified_id:
            raise ValueError(
                "ResolvedSimulationPlan initial conditions reference another LayoutPlan")

    amr_authorities = (
        plan.resolved_hierarchy,
        plan.amr_transfer,
        plan.bootstrap_plan,
        plan.amr_execution,
    )
    if not any(value is not None for value in amr_authorities):
        if plan.amr_providers:
            raise ValueError("non-AMR plan cannot carry AMR provider bindings")
        return
    if plan.target != "amr_system" \
            or any(value is None for value in amr_authorities) \
            or initial_plan is None:
        raise ValueError(
            "AMR hierarchy, transfer, initial-condition, bootstrap, and execution authorities "
            "must be supplied together on an AMR target"
        )
    from pops.mesh._amr import (
        AnalyticReprojection,
        BootstrapPlan,
        ResolvedHierarchy,
    )
    from pops.mesh._amr.transfer import ResolvedAMRTransfer

    from pops.amr import AMRExecution
    authorities = (*amr_authorities[:2], initial_plan, *amr_authorities[2:])
    expected = (
        ResolvedHierarchy, ResolvedAMRTransfer, InitialConditionPlan,
        BootstrapPlan, AMRExecution,
    )
    if any(type(value) is not kind
           for value, kind in zip(authorities, expected, strict=True)):
        raise TypeError("ResolvedSimulationPlan contains a non-exact AMR authority")
    if plan.amr_transfer.layout_plan_id != plan.layout_plan.qualified_id \
            or plan.initial_condition_plan.layout_plan_id != plan.layout_plan.qualified_id \
            or plan.bootstrap_plan.layout_plan_id != plan.layout_plan.qualified_id:
        raise ValueError("ResolvedSimulationPlan AMR authorities reference another LayoutPlan")
    if plan.bootstrap_plan.hierarchy_identity != plan.resolved_hierarchy.identity \
            or plan.bootstrap_plan.transfer_identity != plan.amr_transfer.identity \
            or plan.bootstrap_plan.initial_identity != plan.initial_condition_plan.identity:
        raise ValueError("ResolvedSimulationPlan bootstrap does not authenticate AMR authorities")
    providers = plan.amr_providers
    if tuple(providers) != ("clustering", "tagger"):
        raise ValueError("AMR plan requires exact clustering and tagger provider bindings")
    # Component inputs deliberately admit both source authorities and already-compiled
    # artifacts.  Their representations differ, but both expose the same authenticated
    # projection protocol.  Index that projection instead of reaching through the source-only
    # ``ComponentManifest`` shape: unrelated compiled consumers (for example a Writer) must not
    # make an otherwise builtin AMR plan impossible to validate.
    component_inputs = {}
    for component in plan.component_inputs:
        projection = getattr(component, "to_data", None)
        if not callable(projection):
            raise TypeError("AMR component input lacks its authenticated data projection")
        component_data = projection()
        if not isinstance(component_data, Mapping):
            raise TypeError("AMR component input projection must be a canonical mapping")
        component_id = component_data.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise TypeError("AMR component input projection has no canonical component_id")
        if component_id in component_inputs:
            raise ValueError("AMR component inputs contain a duplicate component authority")
        component_inputs[component_id] = dict(component_data)
    from pops.amr.providers import validate_amr_provider_binding

    for role, binding in providers.items():
        validate_amr_provider_binding(
            role=role,
            frozen_binding=binding,
            layout_identity=plan.layout_plan.qualified_id,
            component_inputs=component_inputs,
            resolved_tagging_identity=plan.bootstrap_plan.tagging.qualified_id,
        )
    hierarchy = plan.resolved_hierarchy
    transitions = hierarchy.plan.transitions
    execution = plan.amr_execution
    if execution.mode == "subcycled":
        expected_children = tuple(range(1, len(transitions) + 1))
        actual_children = tuple(sorted(row.child_level for row in execution.relations))
        if actual_children != expected_children:
            raise ValueError(
                "subcycled AMRExecution requires one explicit temporal relation for every "
                "coarse/fine transition; temporal ratios are never inferred from spatial ratios")
    elif execution.relations:
        raise ValueError("synchronous AMRExecution must not carry temporal relations")
    from pops.mesh._amr.hierarchy_native import validate_native_hierarchy

    validate_native_hierarchy(hierarchy)
    cluster_options = hierarchy.plan.clustering.options.to_data()
    from pops.identity.semantic import semantic_value

    expected_clustering = semantic_value(
        dict(providers["clustering"]), where="AMR clustering provider")
    if cluster_options != {"provider": expected_clustering}:
        raise ValueError("resolved hierarchy clustering differs from the AMR provider authority")
    patch_options = hierarchy.plan.patch_generation.options.to_data()
    expected_patch_options = {
        "native_route", "distribute_coarse", "coarse_max_grid",
    }
    if set(patch_options) != expected_patch_options \
            or type(patch_options.get("distribute_coarse")) is not bool:
        raise TypeError("native AMR patch generation requires the exact box_array option schema")
    coarse_max_grid = patch_options["coarse_max_grid"]
    if coarse_max_grid is not None:
        if type(coarse_max_grid) is not int:
            raise TypeError("native AMR coarse_max_grid must be None or an exact integer")
        if coarse_max_grid < 1:
            raise ValueError("native AMR coarse_max_grid must be positive when provided")
    if patch_options["native_route"] != "box_array":
        raise NotImplementedError(
            "native AMR patch generation requires native_route='box_array'"
        )
    balance_options = hierarchy.plan.load_balance.options.to_data()
    if type(balance_options) is not dict or set(balance_options) != {"provider"}:
        raise TypeError(
            "resolved AMR load balance must preserve one exact provider authority")
    from pops.amr._load_balance_contract import validate_load_balance_provider_data

    validate_load_balance_provider_data(balance_options["provider"])
    state_blocks = []
    for binding in plan.initial_condition_plan.bindings:
        subject = binding.subject
        if subject.kind == "particle":
            raise NotImplementedError(
                "particle/hybrid particle-grid is outside the final native AMR target"
            )
        if subject.kind != "state":
            raise NotImplementedError(
                "native AMR initial conditions require state Handles"
            )
        if subject.block_ref is not None:
            state_blocks.append(subject.block_ref.qualified_id)
    if len(state_blocks) != len(set(state_blocks)):
        raise NotImplementedError(
            "native AMR currently exposes one conservative state space per block; "
            "multiple state Handles are refused before artifact creation"
        )
    from pops.mesh._amr.transfer import (
        CACHE,
        CELL_CENTERED,
        CELL_SPACE,
        CONSERVATIVE_REPRESENTATION,
        DENSE_STORAGE,
        DERIVED_FIELD,
        PHYSICAL,
        PROLONGATION,
        RESTRICTION,
        COARSE_FINE_FILL,
        TEMPORAL_INTERPOLATION,
        PRIMITIVE_REPRESENTATION,
        FACE_SPACE,
        FACE_X_CENTERED,
        FACE_Y_CENTERED,
        NODE_SPACE,
        NODE_CENTERED,
        NativeAMRMaterializationKind,
    )

    tagging_provider = getattr(plan.bootstrap_plan.tagging, "runtime_tagging_data", None)
    if not callable(tagging_provider):
        raise TypeError("resolved AMR tagging must implement runtime_tagging_data(params)")
    tagging_manifest = tagging_provider()
    if type(tagging_manifest) is not dict \
            or tagging_manifest.get("graph_type") != "amr_tagging_runtime" \
            or not tagging_manifest.get("lowerings"):
        raise TypeError("resolved AMR tagging returned an incomplete runtime provider manifest")
    initial_ids = {row.subject.qualified_id for row in plan.initial_condition_plan.bindings}
    for constraint in plan.bootstrap_plan.constraints:
        options = constraint.options.to_data()
        if constraint.subject.qualified_id not in initial_ids \
                or constraint.subject.block_ref is None \
                or options.get("native_route") != "component_floor" \
                or set(options) != {"native_route", "component", "floor"} \
                or isinstance(options.get("component"), bool) \
                or not isinstance(options.get("component"), int) \
                or options["component"] < 0:
            raise NotImplementedError(
                "native AMR constraints require an exact cell-state component_floor provider"
            )
    selections = {
        row.subject.qualified_id: row.method for row in plan.bootstrap_plan.selections
    }
    for binding in plan.initial_condition_plan.bindings:
        options = binding.source.options.to_data()
        analytic = type(selections[binding.subject.qualified_id]) is AnalyticReprojection
        if analytic and (
            binding.subject.kind == "particle"
            or not isinstance(options.get("native_route"), str)
            or not options.get("native_route")
        ):
            raise NotImplementedError(
                "native analytic AMR bootstrap requires a registered data-only source provider"
            )
        if not analytic and options.get("native_route") != "bound_level_zero":
            raise NotImplementedError(
                "native AMR initial source requires native_route='bound_level_zero'"
            )
    for entry in plan.amr_transfer.entries:
        native = _validated_native_materialization(entry)
        for requirement in entry.requirements:
            if requirement.materialization != native.materialization.value:
                raise ValueError(
                    "native AMR action descriptor disagrees with requirement materialization"
                )
            if requirement.materialization == PHYSICAL:
                key = requirement.key.to_data()
                axis = (key["space"]["qualified_id"], key["centering"]["qualified_id"])
                supported_axis = axis in {
                    (CELL_SPACE.qualified_id, CELL_CENTERED.qualified_id),
                    (FACE_SPACE.qualified_id, FACE_X_CENTERED.qualified_id),
                    (FACE_SPACE.qualified_id, FACE_Y_CENTERED.qualified_id),
                    (NODE_SPACE.qualified_id, NODE_CENTERED.qualified_id),
                }
                expected_representation = (
                    PRIMITIVE_REPRESENTATION.qualified_id
                    if axis in {
                        (NODE_SPACE.qualified_id, NODE_CENTERED.qualified_id),
                    }
                    else CONSERVATIVE_REPRESENTATION.qualified_id
                )
                expected_storage = DENSE_STORAGE.qualified_id
                supported_key = (
                    supported_axis
                    and key["representation"]["qualified_id"] == expected_representation
                    and key["storage"]["qualified_id"] == expected_storage
                    and requirement.key.operation in {
                        PROLONGATION,
                        RESTRICTION,
                        COARSE_FINE_FILL,
                        TEMPORAL_INTERPOLATION,
                    }
                )
                if requirement.key.operation in {
                    RESTRICTION,
                    COARSE_FINE_FILL,
                    TEMPORAL_INTERPOLATION,
                } and axis != (
                    CELL_SPACE.qualified_id,
                    CELL_CENTERED.qualified_id,
                ):
                    supported_key = False
                if axis == (CELL_SPACE.qualified_id, CELL_CENTERED.qualified_id) \
                        and requirement.subject.block_ref is None:
                    supported_key = False
                if requirement.subject.qualified_id not in initial_ids or not supported_key:
                    raise NotImplementedError(
                        "native AMR bootstrap supports initialized dense conservative "
                        "cell/face_x/face_y/node states"
                    )
                prolong_contract = {
                    (CELL_SPACE.qualified_id, CELL_CENTERED.qualified_id):
                        ("conservative_linear", 2, (1,)),
                    (FACE_SPACE.qualified_id, FACE_X_CENTERED.qualified_id):
                        ("face_divergence_preserving", 2, (1,)),
                    (FACE_SPACE.qualified_id, FACE_Y_CENTERED.qualified_id):
                        ("face_divergence_preserving", 2, (1,)),
                    (NODE_SPACE.qualified_id, NODE_CENTERED.qualified_id):
                        ("node_bilinear", 2, (1,)),
                }.get(axis)
                if requirement.key.operation == RESTRICTION:
                    route_contract = ("volume_average", 1, (0,))
                elif requirement.key.operation == COARSE_FINE_FILL:
                    # Coarse/fine providers form an open capability family.  The resolved action
                    # already proves that its exact route supports this requirement; the native
                    # registry authenticates and prepares the named implementation at bind time.
                    route_contract = None
                elif requirement.key.operation == TEMPORAL_INTERPOLATION:
                    route_contract = ("linear_time_interpolation", 2, (0,))
                else:
                    route_contract = prolong_contract
                if native.materialization is not NativeAMRMaterializationKind.PHYSICAL:
                    raise NotImplementedError(
                        "native AMR physical requirements need a physical transfer descriptor"
                    )
                capabilities = native.capabilities.transfer
                if capabilities is None:
                    raise NotImplementedError(
                        "native AMR physical descriptor omitted transfer capabilities"
                    )
                if requirement.key.operation == COARSE_FINE_FILL:
                    if not native.native_route or not capabilities.conservative \
                            or capabilities.temporal:
                        raise NotImplementedError(
                            "native AMR coarse/fine provider lacks conservative spatial "
                            "capabilities")
                elif route_contract is None \
                        or (
                            native.native_route,
                            capabilities.order,
                            capabilities.ghost_depth,
                        ) != route_contract:
                    raise NotImplementedError(
                        "native AMR prolongation provider does not match the exact builtin "
                        "cell/face/node kernel contract"
                    )
            elif requirement.materialization == DERIVED_FIELD:
                if native.materialization is not NativeAMRMaterializationKind.DERIVED_FIELD \
                        or native.native_route != "elliptic_solve":
                    raise NotImplementedError(
                        "native AMR bootstrap requires an owner-qualified field and exact "
                        "elliptic_solve materializer"
                    )
            elif requirement.materialization == CACHE:
                if native.materialization is not NativeAMRMaterializationKind.CACHE \
                        or native.native_route != "patch_topology":
                    raise NotImplementedError(
                        "native AMR cache bootstrap requires the patch_topology materializer"
                    )
            else:
                raise NotImplementedError("native AMR bootstrap has an unknown materialization")

    # Reconstruction and hierarchy transfer are separate authorities. Bind them by qualified state
    # identity and refuse a lower-order or shallower coarse/fine provider before artifact creation;
    # otherwise a WENO/MUSCL block could silently execute with a first-order interface injection.
    coarse_fine_capabilities = {}
    for entry in plan.amr_transfer.entries:
        if entry.key.operation != COARSE_FINE_FILL:
            continue
        capabilities = entry.native_materialization.capabilities.transfer
        if capabilities is None:
            raise TypeError("physical coarse/fine transfer omitted its capabilities")
        ghost = tuple(capabilities.ghost_depth)
        if not ghost or len(ghost) not in (1, 2) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in ghost):
            raise TypeError(
                "physical coarse/fine transfer requires one isotropic or two axis ghost depths"
            )
        for requirement in entry.requirements:
            subject = requirement.subject.qualified_id
            previous = coarse_fine_capabilities.get(subject)
            if previous is not None and previous != capabilities:
                raise ValueError(
                    "AMR state %s has conflicting coarse/fine transfer capabilities" % subject
                )
            coarse_fine_capabilities[subject] = capabilities
    for block in plan.blocks:
        formal_order = getattr(block.spatial, "formal_order", None)
        ghost_depth = getattr(block.spatial, "ghost_depth", None)
        if isinstance(formal_order, bool) or not isinstance(formal_order, int) \
                or isinstance(ghost_depth, bool) or not isinstance(ghost_depth, int):
            raise TypeError("AMR spatial provider lacks exact reconstruction order/halo metadata")
        for subject in block.state_identities:
            capabilities = coarse_fine_capabilities.get(subject)
            if capabilities is None:
                raise ValueError(
                    "AMR state %s has no resolved coarse/fine transfer authority" % subject)
            available_ghost = tuple(capabilities.ghost_depth)
            if len(available_ghost) == 1:
                available_ghost *= 2
            if capabilities.order < formal_order or any(
                    value < ghost_depth for value in available_ghost):
                raise NotImplementedError(
                    "AMR state %s uses reconstruction order %d with ghost depth %d, but its "
                    "coarse/fine provider certifies only order %d and ghost depth %r"
                    % (subject, formal_order, ghost_depth, capabilities.order,
                       tuple(capabilities.ghost_depth)))


__all__ = ["validate_amr_authorities"]
