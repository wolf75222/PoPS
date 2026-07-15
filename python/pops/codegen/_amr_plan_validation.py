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
    authorities = (
        plan.resolved_hierarchy,
        plan.amr_transfer,
        plan.initial_condition_plan,
        plan.bootstrap_plan,
        plan.amr_execution,
    )
    if not any(value is not None for value in authorities):
        if plan.amr_providers:
            raise ValueError("non-AMR plan cannot carry AMR provider bindings")
        return
    if plan.target != "amr_system" or any(value is None for value in authorities):
        raise ValueError(
            "AMR hierarchy, transfer, initial-condition, bootstrap, and execution authorities "
            "must be supplied together on an AMR target"
        )
    from pops.mesh._amr import (
        AnalyticReprojection,
        BootstrapPlan,
        InitialConditionPlan,
        ResolvedHierarchy,
    )
    from pops.mesh._amr.transfer import ResolvedAMRTransfer

    from pops.amr import AMRExecution
    expected = (
        ResolvedHierarchy,
        ResolvedAMRTransfer,
        InitialConditionPlan,
        BootstrapPlan,
        AMRExecution,
    )
    if any(type(value) is not kind for value, kind in zip(authorities, expected, strict=True)):
        raise TypeError("ResolvedSimulationPlan contains a non-exact AMR authority")
    if plan.amr_transfer.layout_plan_id != plan.layout_plan.qualified_id \
            or plan.initial_condition_plan.layout_plan_id != plan.layout_plan.qualified_id \
            or plan.bootstrap_plan.layout_plan_id != plan.layout_plan.qualified_id:
        raise ValueError("ResolvedSimulationPlan AMR authorities reference another LayoutPlan")
    if plan.bootstrap_plan.hierarchy_identity != plan.resolved_hierarchy.identity \
            or plan.bootstrap_plan.transfer_identity != plan.amr_transfer.identity \
            or plan.bootstrap_plan.initial_identity != plan.initial_condition_plan.identity:
        raise ValueError("ResolvedSimulationPlan bootstrap does not authenticate AMR authorities")
    if plan.initial_condition_plan.transfer_identity != plan.amr_transfer.identity:
        raise ValueError("ResolvedSimulationPlan initial conditions authenticate another transfer")
    providers = plan.amr_providers
    if tuple(providers) != ("clustering", "tagger"):
        raise ValueError("AMR plan requires exact clustering and tagger provider bindings")
    from pops import interfaces

    expected_interfaces = {
        "clustering": interfaces.Clustering.to_data(),
        "tagger": interfaces.Tagger.to_data(),
    }
    component_inputs = {
        component.component_manifest.component_id: component.to_data()
        for component in plan.component_inputs
    }
    for slot, binding in providers.items():
        if not isinstance(binding, Mapping) or binding.get("schema_version") != 1 \
                or not isinstance(binding.get("provider_identity"), str) \
                or binding.get("layout_identity") != plan.layout_plan.qualified_id \
                or binding.get("native_interface") != expected_interfaces[slot]:
            raise TypeError("AMR %s provider binding is incomplete or unauthenticated" % slot)
        external = binding.get("component_id") is not None
        expected_type = "external_amr_%s" % slot if external else "builtin_amr_%s" % slot
        if binding.get("provider_type") != expected_type:
            raise ValueError("AMR %s provider kind disagrees with its component authority" % slot)
        if external and (
                not isinstance(binding.get("component_manifest_identity"), str)
                or binding.get("interface_version") != 1
                or not isinstance(binding.get("component"), Mapping)
                or component_inputs.get(binding.get("component_id"))
                != dict(binding["component"])):
            raise TypeError("external AMR %s provider lost exact component identity" % slot)
        if not external:
            expected_id = {
                "clustering": "pops.lib.amr::berger_rigoutsos",
                "tagger": "pops.lib.amr::symbolic_tagger",
            }[slot]
            if binding.get("provider_id") != expected_id:
                raise ValueError("builtin AMR %s provider is not canonical" % slot)
    tagger = providers["tagger"]
    from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI

    capability = tagger.get("tagging_capability")
    expected_capability_keys = {
        "schema_version", "capability_type", "leaf_opcodes", "leaf_opcode_ids",
        "logical_opcodes", "logical_opcode_ids", "candidate_outputs",
        "indicator_stencil_routes", "maximum_stencil_terms",
        "maximum_instruction_count", "non_finite_policy", "persistent_hysteresis",
    }
    maximum_stencil_terms = (
        capability.get("maximum_stencil_terms")
        if isinstance(capability, Mapping)
        else None
    )
    if not isinstance(capability, Mapping) or set(capability) != expected_capability_keys \
            or capability.get("schema_version") != 1 \
            or capability.get("capability_type") != "amr_tagging_program" \
            or tuple(capability.get("candidate_outputs", ())) != tuple(
                NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"]) \
            or not set(capability.get("indicator_stencil_routes", ())) <= set(
                NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]) \
            or not capability.get("indicator_stencil_routes") \
            or isinstance(maximum_stencil_terms, bool) \
            or not isinstance(maximum_stencil_terms, int) \
            or maximum_stencil_terms < 1 \
            or maximum_stencil_terms \
            > NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"] \
            or capability.get("non_finite_policy") \
            != NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"] \
            or capability.get("persistent_hysteresis") is not \
            NATIVE_TAGGING_PROGRAM_ABI["persistent_hysteresis"]:
        raise ValueError("AMR tagger lacks the exact candidate-program protocol")
    if tagger.get("tagging_graph_identity") != plan.bootstrap_plan.tagging.qualified_id:
        raise ValueError("AMR tagger authenticates another resolved tagging graph")
    if not isinstance(tagger.get("clock_identity"), str):
        raise ValueError("AMR tagger lost its exact logical clock identity")
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
    if hierarchy.provider.options.to_data().get("native_route") != "shared_n_level":
        raise NotImplementedError(
            "native AMR hierarchy requires native_route='shared_n_level'"
        )
    if any(row.dimension != 2 or row.ratio != (2, 2) for row in transitions):
        raise NotImplementedError(
            "native AMR hierarchy supports only exact two-dimensional ratio-(2,2) transitions"
        )
    buffers = {row.buffer for row in transitions}
    lookaheads = {row.lookahead for row in transitions}
    if len(buffers) != 1 or len(lookaheads) != 1 \
            or any(len(set(row)) != 1 for row in buffers):
        raise NotImplementedError(
            "native AMR requires one isotropic buffer and one lookahead across transitions"
        )
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
    if hierarchy.plan.load_balance.options.to_data() != {"native_route": "round_robin"}:
        raise NotImplementedError(
            "native AMR load balance requires native_route='round_robin'"
        )
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
                    route_contract = ("conservative_coarse_fine", 1, (1,))
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
                if route_contract is None \
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


__all__ = ["validate_amr_authorities"]
