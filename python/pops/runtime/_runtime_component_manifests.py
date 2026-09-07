"""Exact component contracts consumed by :class:`RuntimeInstance`.

Compiled extension packages may carry their own canonical ``ComponentManifest``.  Components
produced by PoPS' compiler are lowered here from the authenticated artifact instead of being
identified by a concrete Python class or by a central algorithm switch.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.codegen._plans import require_install_plan
from pops.model import ComponentManifest
from pops.output._consumer_contracts import consumer_collective_requirements

from ._consumer_ownership import consumer_block_owner
from ._runtime_plan_io import proved_platform


def _block_layouts(plan: Any) -> dict[str, tuple[str, str]]:
    rows = {}
    for assignment in plan.artifact.layout_plan.assignments:
        if assignment.subject_kind == "block":
            rows[assignment.subject.local_id] = (
                assignment.subject.qualified_id,
                assignment.layout.qualified_id,
            )
    expected = tuple(block.name for block in plan.artifact.blocks)
    if set(rows) != set(expected):
        raise ValueError("compiled block/LayoutPlan assignments are not exact")
    return rows


def _reference_block(reference: Any, layouts: Mapping[str, tuple[str, str]],
                     case_owner: Any) -> str:
    by_id = {block_id: name for name, (block_id, _layout_id) in layouts.items()}
    owner = consumer_block_owner(reference, tuple(by_id), case_owner=case_owner)
    if owner is not None:
        return by_id[owner]
    raise ValueError(
        "consumer quantity %s has no exact block owner"
        % getattr(reference, "qualified_id", reference)
    )


def _consumer_contracts(plan: Any) -> tuple[
        dict[str, tuple[str, ...]], dict[str, tuple[dict[str, Any], ...]]]:
    names = tuple(block.name for block in plan.artifact.blocks)
    block_layouts = _block_layouts(plan)
    case_owner = plan.artifact.layout_plan.owner
    resources: dict[str, set[str]] = {name: set() for name in names}
    requirements: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        name: {} for name in names
    }

    def require(block: str, resource: str, operation: str, strategy: str) -> None:
        requirements[block][(resource, operation, strategy)] = {
            "capability": "collective",
            "resource": resource,
            "operation": operation,
            "strategy": strategy,
        }

    graph = plan.artifact.plan.consumer_graph
    if graph is not None:
        for manifest in graph.nodes:
            for quantity in (*manifest.quantities, *manifest.diagnostic_quantities):
                block = _reference_block(quantity.reference, block_layouts, case_owner)
                resources[block].add(quantity.runtime_resource)
            for quantity, operation, strategy in consumer_collective_requirements(
                manifest
            ):
                block = _reference_block(quantity.reference, block_layouts, case_owner)
                require(
                    block,
                    quantity.runtime_resource,
                    operation,
                    strategy,
                )
    for name in names:
        if not resources[name]:
            resources[name].add("state:%s" % block_layouts[name][0])
    return (
        {name: tuple(sorted(values)) for name, values in resources.items()},
        {name: tuple(requirements[name][key] for key in sorted(requirements[name]))
         for name in names},
    )


def _declared_manifest(model: Any) -> ComponentManifest | None:
    value = getattr(model, "component_manifest", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    if type(value) is not ComponentManifest:
        raise TypeError("compiled component_manifest must be an exact ComponentManifest")
    return value


def component_manifests_for_install(install_plan: Any) -> Mapping[str, ComponentManifest]:
    """Return one exact manifest per installed block, with no concrete-class dispatch."""
    plan = require_install_plan(install_plan)
    _, _, _, facts = proved_platform(plan)
    layouts = _block_layouts(plan)
    resources, requirements = _consumer_contracts(plan)
    result = {}
    for ordinal, block in enumerate(plan.artifact.blocks):
        declared = _declared_manifest(block.model)
        if declared is not None:
            result[block.name] = declared
            continue
        block_id, layout_id = layouts[block.name]
        accesses = tuple({"resource": value} for value in resources[block.name])
        result[block.name] = ComponentManifest(
            uri="pops://compiled-artifact/%s/block-%d" % (
                plan.artifact.artifact_identity.hexdigest, ordinal),
            component_type="compiled_spatial_operator",
            version="1.0.0",
            signature={
                "artifact": plan.artifact.artifact_identity.token,
                "block": block_id,
                "layout": layout_id,
            },
            writes=accesses,
            requirements=requirements[block.name],
            effects=tuple(
                {"kind": "state_write", "resource": value}
                for value in resources[block.name]
            ),
            layouts=({"layout": layout_id},),
            clocks=({"clock": "solution", "access": "stage"},),
            target={"variants": [{
                "dimension": facts["dimension"],
                "scalar": facts["compute"],
                "device": facts["device"],
                "features": [],
            }]},
            determinism={"classification": "reproducible", "scope": ["rank_count"]},
            precision={
                "inputs": [facts["compute"]],
                "accumulation": facts["accumulation"],
                "outputs": [facts["storage"]],
            },
            entry_points={"step": "pops_runtime_step"},
        )
    return result


__all__ = ["component_manifests_for_install"]
