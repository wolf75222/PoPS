"""Private deterministic level-zero initialization and recursive AMR bootstrap resolution."""
# ruff: noqa: F405
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .._layout_plan_contracts import LayoutPlan
from .hierarchy import LevelTransition
from .hierarchy_resolution import ResolvedHierarchy
from .tagging_resolution import ResolvedTaggingGraph
from .transfer import (
    ResolvedAMRTransfer, CACHE, CELL_SPACE, DERIVED_FIELD,
    NativeAMRMaterializationKind, PHYSICAL, PROLONGATION, RESTRICTION,
)
from ._bootstrap_contracts import *  # noqa: F403
from ._bootstrap_contracts import _action


class _InitialConditionPlanContract(Protocol):
    """Minimal detached initialization authority consumed by AMR bootstrap."""

    layout_plan_id: str
    bindings: tuple[Any, ...]

    def canonical_identity(self) -> dict[str, Any]: ...


def _initial_condition_plan(value: Any) -> _InitialConditionPlanContract:
    """Authenticate initialization data without importing its authoring implementation."""

    layout_plan_id = getattr(value, "layout_plan_id", None)
    bindings = getattr(value, "bindings", None)
    canonical_identity = getattr(value, "canonical_identity", None)
    if not isinstance(layout_plan_id, str) or not layout_plan_id \
            or not isinstance(bindings, tuple) or not bindings \
            or not callable(canonical_identity):
        raise TypeError(
            "resolve_bootstrap initial_conditions requires an immutable initial-condition "
            "plan exposing layout_plan_id, tuple bindings, and canonical_identity()"
        )
    if any(not callable(getattr(binding, "to_data", None)) for binding in bindings):
        raise TypeError(
            "resolve_bootstrap initial-condition bindings must expose canonical to_data()"
        )
    binding_data = [binding.to_data() for binding in bindings]
    identity = canonical_identity()
    if not isinstance(identity, Mapping) \
            or set(identity) != {"schema_version", "layout_plan_id", "bindings"} \
            or identity.get("schema_version") != 1 \
            or identity.get("layout_plan_id") != layout_plan_id \
            or identity.get("bindings") != binding_data:
        raise ValueError(
            "AMR bootstrap initial-condition plan has an unauthenticated canonical identity"
        )
    return value


def _physical_initial_subjects(transfers: ResolvedAMRTransfer) -> tuple[Any, ...]:
    """Return the unique physical subjects authenticated by an AMR transfer plan."""
    if type(transfers) is not ResolvedAMRTransfer:
        raise TypeError("physical initial subjects require an exact ResolvedAMRTransfer")
    subjects = {}
    for entry in transfers.entries:
        for requirement in entry.requirements:
            subject = requirement.subject
            if requirement.materialization != PHYSICAL \
                    or subject.kind not in {"state", "particle"}:
                continue
            previous = subjects.get(subject.qualified_id)
            if previous is not None \
                    and previous.canonical_identity() != subject.canonical_identity():
                raise ValueError(
                    "AMR transfer requirements contain conflicting physical subject identities")
            subjects[subject.qualified_id] = subject
    if not subjects:
        raise ValueError("AMR bootstrap requires physical transfer subjects")
    return tuple(subjects[key] for key in sorted(subjects))


def resolve_bootstrap(
    *,
    layout_plan: LayoutPlan,
    hierarchy: ResolvedHierarchy,
    transfers: ResolvedAMRTransfer,
    initial_conditions: _InitialConditionPlanContract,
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
    initial_conditions = _initial_condition_plan(initial_conditions)
    if type(tagging) is not ResolvedTaggingGraph:
        raise TypeError("resolve_bootstrap tagging must be ResolvedTaggingGraph")
    if type(ordering) is not BootstrapOrdering:
        raise TypeError("resolve_bootstrap ordering must be BootstrapOrdering")
    if transfers.layout_plan_id != layout_plan.qualified_id or \
            initial_conditions.layout_plan_id != layout_plan.qualified_id:
        raise ValueError("bootstrap authorities belong to different LayoutPlan identities")
    if transfers.nesting_requirement != hierarchy.plan.nesting.transfer:
        raise ValueError(
            "ResolvedHierarchy transfer nesting must be derived from the AMRTransfer registry"
        )
    selection_rows = tuple(selections)
    if any(type(row) is not BootstrapSelection for row in selection_rows):
        raise TypeError("resolve_bootstrap selections must contain BootstrapSelection values")
    selection_map = {row.subject.qualified_id: row for row in selection_rows}
    initial_subjects = {
        row.subject.qualified_id: row.subject for row in initial_conditions.bindings
    }
    physical_subjects = {
        row.qualified_id: row for row in _physical_initial_subjects(transfers)
    }
    initial_ids = set(initial_subjects)
    if initial_ids != set(physical_subjects):
        missing = sorted(set(physical_subjects) - initial_ids)
        extra = sorted(initial_ids - set(physical_subjects))
        raise ValueError(
            "initial-condition subjects must exactly cover physical AMR transfer subjects; "
            "missing=%s extra=%s" % (missing, extra)
        )
    for subject_id, subject in initial_subjects.items():
        if subject.canonical_identity() != physical_subjects[subject_id].canonical_identity():
            raise ValueError(
                "initial-condition subject identity disagrees with the AMR transfer authority")
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
        native = entry.native_materialization
        provider_identity = native.provider_identity.to_data()
        if native.materialization is NativeAMRMaterializationKind.DERIVED_FIELD:
            for requirement in entry.requirements:
                identity = (DERIVED_FIELD, requirement.subject.qualified_id)
                if identity not in materialized:
                    actions.append(
                        _action(
                            0, "projection", "recompute", requirement.subject,
                            {
                                **provider_identity,
                                "field_name": requirement.subject.local_id,
                            },
                        )
                    )
                    materialized.add(identity)
        elif native.materialization is NativeAMRMaterializationKind.CACHE:
            for requirement in entry.requirements:
                identity = (CACHE, requirement.subject.qualified_id)
                if identity not in materialized:
                    actions.append(
                        _action(
                            0, "projection", "invalidate_then_rebuild", requirement.subject,
                            provider_identity,
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
                if entry.native_materialization.materialization \
                        is not NativeAMRMaterializationKind.PHYSICAL:
                    raise ValueError("ProlongFromParent resolved to a non-physical transfer action")
                phase_actions["transfer"].append(
                    _action(
                        level,
                        "transfer",
                        "prolong_from_parent",
                        binding.subject,
                        entry.native_materialization.to_data(),
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
            if entry.native_materialization.materialization \
                    is not NativeAMRMaterializationKind.PHYSICAL:
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
                        {
                            "key": entry.key.to_data(),
                            "action": entry.native_materialization.to_data(),
                        },
                    )
                )
                physical_actions.add(identity)
        materialized = set()
        for entry in entries:
            native = entry.native_materialization
            provider_identity = native.provider_identity.to_data()
            for requirement in entry.requirements:
                if requirement.materialization == DERIVED_FIELD:
                    identity = (DERIVED_FIELD, requirement.subject.qualified_id)
                    if identity not in materialized:
                        phase_actions["projection"].append(
                            _action(
                                level, "projection", "recompute", requirement.subject,
                                {
                                    **provider_identity,
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
                                provider_identity,
                            )
                        )
                        phase_actions["projection"].append(
                            _action(
                                level, "projection", "rebuild_cache", requirement.subject,
                                provider_identity,
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
            if restriction.native_materialization.materialization \
                    is not NativeAMRMaterializationKind.PHYSICAL:
                raise ValueError("cell bootstrap restriction resolved to a non-physical action")
            actions.append(
                _action(
                    level,
                    "synchronization",
                    "synchronize_covered_cells",
                    binding.subject,
                    restriction.native_materialization.to_data(),
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
    "BootstrapSelection", "ConstraintProvider", "ProlongFromParent", "resolve_bootstrap",
]
