"""Compiled implicit boundary terms remain explicit in residual/JVP reports."""
from __future__ import annotations

import pytest

from pops.mesh.boundaries.compiled_plan import (
    BoundaryLinearizationReport,
    BoundaryResidualReport,
    CompiledBoundaryPlan,
)


STATE = "case::block::state"
PRODUCER = "case::block::implicit-wall"
RESIDUAL = "case::block::implicit-wall-residual"
LINEARIZATION = "case::block::implicit-wall-jvp"


def _handle(qualified_id: str, kind: str) -> dict[str, object]:
    return {"qualified_id": qualified_id, "kind": kind}


def _component(target: str, operation: str) -> dict[str, object]:
    return {
        "target": _handle(target, "residual_operator" if operation == "residual"
                          else "linearization_operator"),
        "component_id": "pops://test/field-boundary@1",
        "component_manifest_identity": "component-manifest:test-field-boundary",
        "native_interface": {"name": "field_boundary_closure", "version": 1},
        "interface_version": 1,
        "operation": operation,
        "producer_identity": PRODUCER,
        "state_identity": STATE,
        "ghost_identity": "case::block::x-min",
        "region": {"kind": "face", "region_identity": "case::block::x-min"},
        "states": [STATE],
        "directions": [STATE] if operation == "jvp" else [],
        "fields": [],
        "parameters": [],
        "outputs": ["case::block::implicit-wall-%s-output" % operation],
        "rate": None,
        "nonlinear_iterate": STATE,
    }


def _compile_data() -> dict[str, object]:
    region = {"canonical_id": "case::block::x-min"}
    return {
        "schema_version": 1,
        "authority_type": "prepared_boundary_plan_compile",
        "ghost_plan_identity": "ghost-plan:exact",
        "source_plan": "ghost-plan:exact",
        "state": _handle(STATE, "state"),
        "required_depth": 1,
        "ncomp": 1,
        "faces": [
            {"ordinal": ordinal, "type": "foextrap", "values": [0.0]}
            for ordinal in range(4)
        ],
        "producer_order": [PRODUCER],
        "corner_policies": [],
        "interfaces": [],
        "residual_contributions": [{
            "handle": _handle("case::block::residual-contribution",
                              "boundary_residual_contribution"),
            "region": region,
            "producer": _handle(PRODUCER, "ghost_producer"),
            "residual": _handle(RESIDUAL, "residual_operator"),
        }],
        "linearization_contributions": [{
            "handle": _handle("case::block::linearization-contribution",
                              "boundary_linearization_contribution"),
            "region": region,
            "producer": _handle(PRODUCER, "ghost_producer"),
            "linearization": _handle(LINEARIZATION, "linearization_operator"),
        }],
        "component_region_templates": [
            _component(RESIDUAL, "residual"),
            _component(LINEARIZATION, "jvp"),
        ],
    }


def test_compiled_plan_reports_exact_implicit_residual_and_linearization_terms():
    plan = CompiledBoundaryPlan(_compile_data())

    residual = plan.residual_report()
    linearization = plan.linearization_report()
    assert isinstance(residual, BoundaryResidualReport)
    assert isinstance(linearization, BoundaryLinearizationReport)
    assert residual.to_dict()["terms"][0]["contribution"]["residual"]["qualified_id"] == RESIDUAL
    assert residual.to_dict()["terms"][0]["component_region"]["operation"] == "residual"
    assert linearization.to_dict()["terms"][0]["contribution"][
        "linearization"]["qualified_id"] == LINEARIZATION
    assert linearization.to_dict()["terms"][0]["component_region"]["operation"] == "jvp"
    assert residual.to_dict()["terms"][0]["contribution"]["producer"]["qualified_id"] \
        == linearization.to_dict()["terms"][0]["contribution"]["producer"]["qualified_id"]

    runtime = plan.runtime_boundary_data({})
    assert runtime["residual_report"] == residual.to_dict()
    assert runtime["linearization_report"] == linearization.to_dict()


def test_compiled_plan_preserves_signed_periodic_identification_through_bind():
    data = _compile_data()
    data["faces"][0]["type"] = "periodic"
    data["faces"][1]["type"] = "periodic"
    data["periodic_identifications"] = [{
        "source": {"qualified_id": "case::xlo"},
        "target": {"qualified_id": "case::xhi"},
        "source_face": 0,
        "target_face": 1,
        "permutation": [0, 1],
        "signs": [1, -1],
    }]

    runtime = CompiledBoundaryPlan(data).runtime_boundary_data({})

    assert runtime["periodic_identifications"] == data["periodic_identifications"]


@pytest.mark.parametrize(("table", "operation"), (
    ("residual_contributions", "residual"),
    ("linearization_contributions", "jvp"),
))
def test_compiled_boundary_report_rejects_a_missing_exact_component_row(table, operation):
    data = _compile_data()
    data["component_region_templates"] = [
        row for row in data["component_region_templates"]
        if row["operation"] != operation
    ]
    plan = CompiledBoundaryPlan(data)
    report = plan.residual_report if table == "residual_contributions" \
        else plan.linearization_report
    with pytest.raises(ValueError, match="has no exact component report row"):
        report()
