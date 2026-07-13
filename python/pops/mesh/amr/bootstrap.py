"""Resolve deterministic level-zero initialization and recursive AMR bootstrap."""
# ruff: noqa: F405
from __future__ import annotations

from typing import Any

from .._layout_plan_contracts import LayoutPlan
from .hierarchy import LevelTransition
from .hierarchy_resolution import ResolvedHierarchy
from .tagging_resolution import ResolvedTaggingGraph
from .transfer import (
    ResolvedAMRTransfer, ApplyTransferProvider, CACHE, CELL_SPACE, DERIVED_FIELD,
    InvalidateThenRebuild, PHYSICAL, PROLONGATION, RESTRICTION, Recompute,
    TEMPORAL_INTERPOLATION,
)
from ._bootstrap_contracts import *  # noqa: F403
from ._bootstrap_contracts import _action, _handle


def resolve_bootstrap(
    *,
    layout_plan: LayoutPlan,
    hierarchy: ResolvedHierarchy,
    transfers: ResolvedAMRTransfer,
    initial_conditions: InitialConditionPlan,
    tagging: ResolvedTaggingGraph,
    selections: Any,
    ordering: BootstrapOrdering,
    constraints: Any = (),
) -> BootstrapPlan:
    """Build level zero, then tag/cluster/create and materialize each transition recursively."""
    if type(layout_plan) is not LayoutPlan:
        raise TypeError("resolve_bootstrap layout_plan must be LayoutPlan")
    if type(hierarchy) is not ResolvedHierarchy:
        raise TypeError("resolve_bootstrap hierarchy must be ResolvedHierarchy")
    if type(transfers) is not ResolvedAMRTransfer:
        raise TypeError("resolve_bootstrap transfers must be AMRTransfer")
    if type(initial_conditions) is not InitialConditionPlan:
        raise TypeError("resolve_bootstrap initial_conditions must be InitialConditionPlan")
    if type(tagging) is not ResolvedTaggingGraph:
        raise TypeError("resolve_bootstrap tagging must be ResolvedTaggingGraph")
    if type(ordering) is not BootstrapOrdering:
        raise TypeError("resolve_bootstrap ordering must be BootstrapOrdering")
    if transfers.layout_plan_id != layout_plan.qualified_id or \
            initial_conditions.layout_plan_id != layout_plan.qualified_id:
        raise ValueError("bootstrap authorities belong to different LayoutPlan identities")
    if initial_conditions.transfer_identity != transfers.identity:
        raise ValueError("initial-condition manifest belongs to a different AMRTransfer")
    if transfers.nesting_requirement != hierarchy.plan.nesting.transfer:
        raise ValueError(
            "ResolvedHierarchy transfer nesting must be derived from the AMRTransfer registry"
        )
    selection_rows = tuple(selections)
    if any(type(row) is not BootstrapSelection for row in selection_rows):
        raise TypeError("resolve_bootstrap selections must contain BootstrapSelection values")
    selection_map = {row.subject.qualified_id: row for row in selection_rows}
    initial_ids = {row.subject.qualified_id for row in initial_conditions.bindings}
    if len(selection_map) != len(selection_rows):
        raise ValueError("duplicate bootstrap selection")
    if set(selection_map) != initial_ids:
        raise ValueError("bootstrap selections must exactly cover initial-condition subjects")
    constraint_rows = tuple(constraints)
    if any(type(constraint) is not ConstraintProvider for constraint in constraint_rows):
        raise TypeError("bootstrap constraints must contain ConstraintProvider values")

    for binding in initial_conditions.bindings:
        normalized = layout_plan.normalized(binding.layout)
        if len(normalized.levels) != hierarchy.plan.level_count:
            raise ValueError(
                "LayoutPlan and ResolvedHierarchy disagree on materialized level count"
            )

    actions = []
    for binding in initial_conditions.bindings:
        actions.append(
            _action(
                0,
                "initial_condition",
                "initialize_level_zero",
                binding.subject,
                binding.source.canonical_identity(),
            )
        )
    entries = transfers.entries
    materialized: set[tuple[str, str]] = set()
    for entry in entries:
        if isinstance(entry.action, Recompute):
            for requirement in entry.requirements:
                identity = (DERIVED_FIELD, requirement.subject.qualified_id)
                if identity not in materialized:
                    actions.append(
                        _action(
                            0, "projection", "recompute", requirement.subject,
                            {
                                **entry.action.provider.canonical_identity(),
                                "field_name": requirement.subject.local_id,
                            },
                        )
                    )
                    materialized.add(identity)
        elif isinstance(entry.action, InvalidateThenRebuild):
            for requirement in entry.requirements:
                identity = (CACHE, requirement.subject.qualified_id)
                if identity not in materialized:
                    actions.append(
                        _action(
                            0, "projection", "invalidate_then_rebuild", requirement.subject,
                            entry.action.provider.canonical_identity(),
                        )
                    )
                    materialized.add(identity)
    for constraint in constraint_rows:
        actions.append(
            _action(
                0, "constraint", "apply_constraint", constraint.subject,
                constraint.canonical_identity(),
            )
        )

    for transition in hierarchy.plan.transitions:
        if type(transition) is not LevelTransition:
            raise TypeError("resolved hierarchy contains a non-LevelTransition row")
        level = transition.fine_level
        actions.extend(
            (
                _action(
                    level,
                    "hierarchy",
                    "tag_parent",
                    evidence={
                        "parent_level": transition.coarse_level,
                        "tagging": tagging.canonical_identity(),
                    },
                ),
                _action(
                    level,
                    "hierarchy",
                    "cluster_tags",
                    evidence=hierarchy.plan.clustering.canonical_identity(),
                ),
                _action(
                    level,
                    "hierarchy",
                    "create_level",
                    evidence=transition.canonical_identity(),
                ),
            )
        )
        phase_actions = {"transfer": [], "projection": [], "constraint": []}
        for binding in initial_conditions.bindings:
            selection = selection_map[binding.subject.qualified_id]
            if type(selection.method) is ProlongFromParent:
                try:
                    entry = transfers.for_subject(binding.subject, PROLONGATION)
                except KeyError as exc:
                    raise ValueError(
                        "ProlongFromParent requires an exact resolved prolongation provider"
                    ) from exc
                if not isinstance(entry.action, ApplyTransferProvider):
                    raise ValueError("ProlongFromParent resolved to a non-physical transfer action")
                phase_actions["transfer"].append(
                    _action(
                        level,
                        "transfer",
                        "prolong_from_parent",
                        binding.subject,
                        entry.action.to_data(),
                    )
                )
            else:
                phase_actions["projection"].append(
                    _action(
                        level,
                        "projection",
                        "analytic_reprojection",
                        binding.subject,
                        binding.source.canonical_identity(),
                    )
                )
        initial_subjects = {row.subject.qualified_id for row in initial_conditions.bindings}
        physical_actions = set()
        for entry in entries:
            if not isinstance(entry.action, ApplyTransferProvider):
                continue
            for requirement in entry.requirements:
                identity = (requirement.subject.qualified_id, entry.key.identity.token)
                if requirement.subject.qualified_id in initial_subjects \
                        or identity in physical_actions:
                    continue
                phase_actions["transfer"].append(
                    _action(
                        level,
                        "transfer",
                        "apply_transfer_provider",
                        requirement.subject,
                        {"key": entry.key.to_data(), "action": entry.action.to_data()},
                    )
                )
                physical_actions.add(identity)
        materialized = set()
        for entry in entries:
            for requirement in entry.requirements:
                if requirement.materialization == DERIVED_FIELD:
                    identity = (DERIVED_FIELD, requirement.subject.qualified_id)
                    if identity not in materialized:
                        phase_actions["projection"].append(
                            _action(
                                level, "projection", "recompute", requirement.subject,
                                {
                                    **entry.action.provider.canonical_identity(),
                                    "field_name": requirement.subject.local_id,
                                },
                            )
                        )
                        materialized.add(identity)
                elif requirement.materialization == CACHE:
                    identity = (CACHE, requirement.subject.qualified_id)
                    if identity not in materialized:
                        phase_actions["transfer"].append(
                            _action(
                                level, "transfer", "invalidate_cache", requirement.subject,
                                entry.action.provider.canonical_identity(),
                            )
                        )
                        phase_actions["projection"].append(
                            _action(
                                level, "projection", "rebuild_cache", requirement.subject,
                                entry.action.provider.canonical_identity(),
                            )
                        )
                        materialized.add(identity)
                elif requirement.materialization != PHYSICAL:
                    raise ValueError("unsupported bootstrap materialization")
        for constraint in constraint_rows:
            phase_actions["constraint"].append(
                _action(
                    level,
                    "constraint",
                    "apply_constraint",
                    constraint.subject,
                    evidence=constraint.canonical_identity(),
                )
            )
        for phase in ordering.phases:
            actions.extend(sorted(phase_actions[phase], key=lambda row: row.subject_id or ""))
        for binding in initial_conditions.bindings:
            prolongation = transfers.for_subject(binding.subject, PROLONGATION)
            if prolongation.key.space != CELL_SPACE:
                continue
            try:
                restriction = transfers.for_subject(binding.subject, RESTRICTION)
            except KeyError as exc:
                raise ValueError(
                    "cell bootstrap requires an exact restriction provider for final synchronization"
                ) from exc
            if not isinstance(restriction.action, ApplyTransferProvider):
                raise ValueError("cell bootstrap restriction resolved to a non-physical action")
            actions.append(
                _action(
                    level,
                    "synchronization",
                    "synchronize_covered_cells",
                    binding.subject,
                    restriction.action.to_data(),
                )
            )
    return BootstrapPlan(
        layout_plan.qualified_id,
        hierarchy.identity,
        transfers.identity,
        initial_conditions.identity,
        tagging,
        ordering,
        tuple(sorted(selection_rows, key=lambda row: row.subject.qualified_id)),
        constraint_rows,
        tuple(actions),
    )


__all__ = [
    "AnalyticReprojection", "BootstrapAction", "BootstrapOrdering", "BootstrapPlan",
    "BootstrapSelection", "InitialConditionBinding", "InitialConditionPlan",
    "ConstraintProvider",
    "InitialConditionPlanBuilder", "InitialConditionSource", "ProlongFromParent",
    "resolve_bootstrap",
]
