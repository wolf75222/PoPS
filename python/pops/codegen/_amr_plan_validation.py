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


def _ranked_ghost_depth(
    value: Any, *, dimension: int, where: str
) -> tuple[int, ...]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise TypeError("%s must be an ordered axis sequence" % where) from exc
    if len(raw) not in (1, dimension) or any(
        type(item) is not int or item < 0 for item in raw
    ):
        raise TypeError(
            "%s must contain one or %d non-negative integers" % (where, dimension)
        )
    if len(raw) == 1:
        return tuple(raw[0] for _ in range(dimension))
    return raw


def _validate_builtin_coarse_fine_capabilities(
    *, native_route: str, capabilities: Any, dimension: int
) -> None:
    """Authenticate the intrinsic capability tuple of each builtin coarse/fine route.

    A resolved coarse/fine requirement intentionally carries no parent finite-volume stencil
    halo.  It therefore cannot, by itself, prove that a descriptor which claims one of the
    builtin native kernels has retained that kernel's intrinsic interpolation radius.
    """
    builtin = {
        "conservative_coarse_fine": (2, 1, False, False),
        "conservative_polynomial5_coarse_fine": (5, 3, False, False),
        "conservative_injection": (1, 1, True, False),
    }.get(native_route)
    if builtin is None:
        return
    order, ghost, conservative, temporal = builtin
    actual = (
        capabilities.order,
        _ranked_ghost_depth(
            capabilities.ghost_depth,
            dimension=dimension,
            where="builtin coarse/fine transfer ghost_depth",
        ),
        capabilities.conservative,
        capabilities.temporal,
    )
    expected = (order, (ghost,) * dimension, conservative, temporal)
    if actual != expected:
        raise NotImplementedError(
            "native AMR coarse/fine provider does not match the exact builtin kernel contract"
        )


def _physical_axis_contract(
    plan: Any, requirement: Any
) -> tuple[tuple[str, str], str | None, int]:
    """Classify one physical transfer key against its exact layout axis names."""
    from pops.mesh._amr.transfer import (
        BuiltinTransferAxis,
        CELL_CENTERED,
        CELL_SPACE,
        FACE_SPACE,
        NODE_CENTERED,
        NODE_SPACE,
    )

    normalized = plan.layout_plan.normalized(requirement.layout)
    axis_names = tuple(normalized.geometry.axis_names)
    dimension = len(axis_names)
    if requirement.accuracy.dimension != dimension:
        raise ValueError(
            "AMR transfer requirement dimension differs from its normalized layout geometry"
        )
    key = requirement.key.to_data()
    axis = (key["space"]["qualified_id"], key["centering"]["qualified_id"])
    cell_axis = (CELL_SPACE.qualified_id, CELL_CENTERED.qualified_id)
    node_axis = (NODE_SPACE.qualified_id, NODE_CENTERED.qualified_id)
    face_axes = {
        (
            FACE_SPACE.qualified_id,
            BuiltinTransferAxis("centering", "face_%s" % axis_name).qualified_id,
        )
        for axis_name in axis_names
    }
    axis_kind = {
        cell_axis: "cell",
        node_axis: "node",
        **{face_axis: "face" for face_axis in face_axes},
    }.get(axis)
    return axis, axis_kind, dimension


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
    if tuple(providers) != ("clustering", "tagger", "reflux"):
        raise ValueError(
            "AMR plan requires exact clustering, tagger and reflux provider bindings")
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
        CONSERVATIVE_REPRESENTATION,
        DENSE_STORAGE,
        DERIVED_FIELD,
        PHYSICAL,
        PROLONGATION,
        RESTRICTION,
        COARSE_FINE_FILL,
        TEMPORAL_INTERPOLATION,
        PRIMITIVE_REPRESENTATION,
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
                _, axis_kind, dimension = _physical_axis_contract(plan, requirement)
                supported_axis = axis_kind is not None
                expected_representation = (
                    PRIMITIVE_REPRESENTATION.qualified_id
                    if axis_kind == "node"
                    else CONSERVATIVE_REPRESENTATION.qualified_id
                )
                expected_storage = DENSE_STORAGE.qualified_id
                supported_key = (
                    supported_axis
                    and key["representation"]["qualified_id"] == expected_representation
                    and key["storage"]["qualified_id"] == expected_storage
                    and (
                        (axis_kind == "cell" and requirement.key.operation in {
                            PROLONGATION,
                            RESTRICTION,
                            COARSE_FINE_FILL,
                            TEMPORAL_INTERPOLATION,
                        })
                        or (axis_kind in {"face", "node"}
                            and requirement.key.operation == PROLONGATION)
                    )
                )
                if axis_kind == "cell" and requirement.subject.block_ref is None:
                    supported_key = False
                if requirement.subject.qualified_id not in initial_ids or not supported_key:
                    raise NotImplementedError(
                        "native AMR bootstrap physical transfer key has no exact prepared provider"
                    )
                if axis_kind == "face":
                    route_contract = (
                        "divergence_preserving_face",
                        2,
                        tuple(1 for _ in range(dimension)),
                    )
                elif axis_kind == "node":
                    route_contract = (
                        "node_multilinear",
                        2,
                        tuple(0 for _ in range(dimension)),
                    )
                else:
                    if requirement.key.operation == RESTRICTION:
                        route_contract = (
                            "volume_average",
                            1,
                            tuple(0 for _ in range(dimension)),
                        )
                    elif requirement.key.operation == COARSE_FINE_FILL:
                        # Coarse/fine providers form an open capability family.  The resolved
                        # action proves that its exact route supports this requirement.
                        route_contract = None
                    elif requirement.key.operation == TEMPORAL_INTERPOLATION:
                        route_contract = (
                            "linear_time_interpolation",
                            2,
                            tuple(0 for _ in range(dimension)),
                        )
                    elif native.native_route == "conservative_injection":
                        route_contract = (
                            "conservative_injection",
                            1,
                            tuple(0 for _ in range(dimension)),
                        )
                    else:
                        route_contract = (
                            "conservative_linear",
                            2,
                            tuple(1 for _ in range(dimension)),
                        )
                if native.materialization is not NativeAMRMaterializationKind.PHYSICAL:
                    raise NotImplementedError(
                        "native AMR physical requirements need a physical transfer descriptor"
                    )
                capabilities = native.capabilities.transfer
                if capabilities is None:
                    raise NotImplementedError(
                        "native AMR physical descriptor omitted transfer capabilities"
                    )
                if capabilities.temporal != (
                    requirement.key.operation == TEMPORAL_INTERPOLATION
                ):
                    raise NotImplementedError(
                        "native AMR transfer temporal capability disagrees with its operation"
                    )
                if requirement.key.operation == COARSE_FINE_FILL:
                    if not native.native_route or capabilities.temporal:
                        raise NotImplementedError(
                            "native AMR coarse/fine provider lacks spatial capabilities")
                    _validate_builtin_coarse_fine_capabilities(
                        native_route=native.native_route,
                        capabilities=capabilities,
                        dimension=dimension,
                    )
                elif route_contract is None \
                        or (
                            native.native_route,
                            capabilities.order,
                            _ranked_ghost_depth(
                                capabilities.ghost_depth,
                                dimension=dimension,
                                where="native AMR transfer capability ghost_depth",
                            ),
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
    # identity and refuse a lower-order coarse/fine provider before artifact creation; otherwise a
    # WENO/MUSCL block could silently execute with a first-order interface interpolation.  The FV
    # block halo is an input to the parent spatial stencil, not an interface-transfer requirement:
    # validate the latter only against the exact resolved coarse/fine requirement.
    coarse_fine_capabilities = {}
    for entry in plan.amr_transfer.entries:
        native = _validated_native_materialization(entry)
        if entry.key.operation != COARSE_FINE_FILL \
                or native.materialization is not NativeAMRMaterializationKind.PHYSICAL:
            continue
        capabilities = native.capabilities.transfer
        if capabilities is None:
            raise TypeError("physical coarse/fine transfer omitted its capabilities")
        for requirement in entry.requirements:
            subject = requirement.subject.qualified_id
            dimension = requirement.accuracy.dimension
            required_ghost = _ranked_ghost_depth(
                requirement.accuracy.ghost_depth,
                dimension=dimension,
                where="physical coarse/fine requirement ghost_depth",
            )
            _ranked_ghost_depth(
                capabilities.ghost_depth,
                dimension=dimension,
                where="physical coarse/fine transfer ghost_depth",
            )
            previous = coarse_fine_capabilities.get(subject)
            selected = (capabilities, dimension, required_ghost)
            if previous is not None and previous != selected:
                raise ValueError(
                    "AMR state %s has conflicting coarse/fine transfer capabilities" % subject
                )
            coarse_fine_capabilities[subject] = selected
    for block in plan.blocks:
        formal_order = getattr(block.spatial, "formal_order", None)
        if isinstance(formal_order, bool) or not isinstance(formal_order, int) \
                or formal_order < 1:
            raise TypeError("AMR spatial provider lacks exact reconstruction order/halo metadata")
        for subject in block.state_identities:
            selected = coarse_fine_capabilities.get(subject)
            if selected is None:
                raise ValueError(
                    "AMR state %s has no resolved coarse/fine transfer authority" % subject)
            capabilities, dimension, required_ghost = selected
            available_ghost = _ranked_ghost_depth(
                capabilities.ghost_depth,
                dimension=dimension,
                where="physical coarse/fine transfer ghost_depth",
            )
            if capabilities.order < formal_order:
                raise NotImplementedError(
                    "AMR state %s uses reconstruction order %d, but its coarse/fine provider "
                    "certifies only order %d"
                    % (subject, formal_order, capabilities.order))
            if any(
                    available < required
                    for available, required in zip(
                        available_ghost, required_ghost, strict=True
                    )):
                raise NotImplementedError(
                    "AMR state %s requires coarse/fine ghost depth %r, but its provider "
                    "certifies only ghost depth %r"
                    % (subject, required_ghost, tuple(capabilities.ghost_depth)))


__all__ = ["validate_amr_authorities"]
