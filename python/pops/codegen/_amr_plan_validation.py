"""Exact AMR authority cross-checks and current native lowering boundary."""
from __future__ import annotations

from typing import Any


def validate_amr_authorities(plan: Any) -> None:
    authorities = (
        plan.resolved_hierarchy,
        plan.amr_transfer,
        plan.initial_condition_plan,
        plan.bootstrap_plan,
    )
    if not any(value is not None for value in authorities):
        return
    if plan.target != "amr_system" or any(value is None for value in authorities):
        raise ValueError(
            "AMR hierarchy, transfer, initial-condition, and bootstrap authorities "
            "must be supplied together on an AMR target"
        )
    from pops.mesh.amr import (
        AnalyticReprojection,
        BootstrapPlan,
        InitialConditionPlan,
        ResolvedHierarchy,
    )
    from pops.mesh.amr.transfer import ResolvedAMRTransfer

    expected = (ResolvedHierarchy, ResolvedAMRTransfer, InitialConditionPlan, BootstrapPlan)
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
    hierarchy = plan.resolved_hierarchy
    transitions = hierarchy.plan.transitions
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
    policy_routes = (
        (hierarchy.plan.clustering.options, "berger_rigoutsos", "clustering"),
        (hierarchy.plan.patch_generation.options, "box_array", "patch generation"),
        (hierarchy.plan.load_balance.options, "round_robin", "load balance"),
    )
    for options, route, label in policy_routes:
        if options.to_data().get("native_route") != route:
            raise NotImplementedError(
                "native AMR %s requires native_route=%r" % (label, route)
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
    from pops.mesh.amr import (
        Above,
    )
    from pops.mesh.amr.transfer import (
        ApplyTransferProvider,
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
        Recompute,
        InvalidateThenRebuild,
    )

    graph = plan.bootstrap_plan.tagging.graph
    if type(graph.refine) is not Above or graph.coarsen is not None \
            or graph.refine.indicator.kind != "state" \
            or graph.refine.indicator.block_ref is None:
        raise NotImplementedError(
            "native AMR artifact bootstrap lowers only an owner-qualified block-state "
            "Above(indicator, threshold) without a coarsen root; field indicators require "
            "a backend TagIndicator provider"
        )
    from pops.mesh.amr import TagNodeRegistry
    builtin_above = TagNodeRegistry.builtins().registration_for(graph.refine)
    resolved_above = tuple(
        row for row in plan.bootstrap_plan.tagging.registrations
        if row.node_type == graph.refine.node_type
    )
    if len(resolved_above) != 1 \
            or resolved_above[0].canonical_identity() != builtin_above.canonical_identity():
        raise NotImplementedError(
            "native AMR tagging has no prepared kernel manifest for the selected "
            "owner-qualified TagIndicator provider"
        )
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
            or options.get("native_route") != "constant_field"
            or not options.get("components")
        ):
            raise NotImplementedError(
                "native analytic AMR bootstrap requires a cell/face/node constant_field source"
            )
        if not analytic and options.get("native_route") != "bound_level_zero":
            raise NotImplementedError(
                "native AMR initial source requires native_route='bound_level_zero'"
            )
    for entry in plan.amr_transfer.entries:
        for requirement in entry.requirements:
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
                if type(entry.action) is not ApplyTransferProvider:
                    raise NotImplementedError(
                        "native AMR physical requirements need an exact transfer provider"
                    )
                capabilities = entry.action.capabilities
                if route_contract is None \
                        or (
                            entry.action.route.options.to_data().get("native_route"),
                            capabilities.order,
                            capabilities.ghost_depth,
                        ) != route_contract:
                    raise NotImplementedError(
                        "native AMR prolongation provider does not match the exact builtin "
                        "cell/face/node kernel contract"
                    )
            elif requirement.materialization == DERIVED_FIELD:
                if type(entry.action) is not Recompute \
                        or entry.action.provider.options.to_data().get("native_route") \
                        != "elliptic_solve":
                    raise NotImplementedError(
                        "native AMR bootstrap requires an owner-qualified field and exact "
                        "elliptic_solve materializer"
                    )
            elif requirement.materialization == CACHE:
                if type(entry.action) is not InvalidateThenRebuild \
                        or entry.action.provider.options.to_data().get("native_route") \
                        != "patch_topology":
                    raise NotImplementedError(
                        "native AMR cache bootstrap requires the patch_topology materializer"
                    )
            else:
                raise NotImplementedError("native AMR bootstrap has an unknown materialization")


__all__ = ["validate_amr_authorities"]
