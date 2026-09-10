"""Bind typed physical maps to bounded native composite-cell transfer plans."""
from __future__ import annotations

import math
from typing import Any


def physical_amr_spec(plan: Any, requirement: Any, source_engine: Any, target_engine: Any,
                      source_block: str, target_block: str) -> dict:
    from pops.mesh.native_physical_mapping import physical_map_identity
    physical = requirement.physical_map
    dim = physical.native_dimension
    weights = [[] for _ in range(dim)]
    lower, upper = [0.] * dim, [1.] * dim
    for reduction in physical.reductions:
        weights[reduction.axis] = [float(value) for value in reduction.weights]
        lower[reduction.axis] = float(reduction.lower)
        upper[reduction.axis] = float(reduction.upper)
    def capacity(layout):
        row = plan.artifact.layout_plan.normalized(layout)
        shape = tuple(row.native_spatial_layout.shape)
        cells = math.prod(shape)
        for ratio in row.transition_ratios:
            shape = tuple(a * b for a, b in zip(shape, ratio, strict=True))
            cells += math.prod(shape)
        return cells
    source = capacity(requirement.source_layout)
    target = capacity(requirement.target_layout)
    budget = source_engine._native_step_target()._layout_transfer_capacity_budget(
        target_engine._native_step_target(), source_block, target_block, source, target)
    if any(type(value) is not int or value < 1 or value > (1 << 63) - 1
           for value in budget.values()):
        raise OverflowError("physical AMR transfer capacity exceeds signed native size limits")
    return {"physical_contract_identity": physical_map_identity(physical),
            "base_bin_weights": weights, "base_bin_lower": lower, "base_bin_upper": upper,
            "quadrature_identity": "pops://measure/piecewise-constant-base-bins@1",
            "budget": budget}


def authenticate_amr_receipt(route, receipt, expected):
    prepared = route.session.expected_receipt_contract()
    if prepared.physical_contract_identity != route.physical_contract_identity:
        raise RuntimeError("prepared AMR transfer lost its resolved physical descriptor")
    keys = ("source_element_count", "destination_element_count",
            "physical_contract_identity", "source_hierarchy_identity", "target_hierarchy_identity",
            "source_hierarchy_generation", "target_hierarchy_generation", "source_stage_identity",
            "target_stage_identity", "source_stage_generation", "target_stage_generation",
            "source_active_elements", "destination_active_elements", "canonical_jobs",
            "transported_elements", "prepared_bytes")
    for key in keys:
        expected[key] = getattr(prepared, key)
    if not route.program_invocation and (prepared.source_stage_identity != "accepted-current"
                                        or prepared.target_stage_identity != "accepted-current"):
        raise RuntimeError("accepted AMR transfer authenticated a nonaccepted source or target")
