"""JSON Schema for campaign verification report summary (plan §31)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"

# Local pr summary — Dim1 + Serial, one planned/run/passed case, first-page
# topics that were not run are null with reasons (plan §31 / §31.1).
LOCAL_PR_SUMMARY = {
    "schema": "pops.verification.report.v1",
    "repository": "wolf75222/PoPS",
    "repository_sha": "0123456789abcdef0123456789abcdef01234567",
    "suite": "pr",
    "max_nodes": 2,
    "native_dimensions": [1],
    "execution_spaces": ["KokkosSerial"],
    "coverage": {
        "components": ["euler"],
        "cases_planned": 1,
        "cases_run": 1,
        "cases_passed": 1,
        "cases_failed": 0,
        "cases_not_supported": 0,
        "not_tested": [],
    },
    "failures": [],
    "orders": [
        {
            "case_id": "CP-02",
            "kind": "spatial",
            "variable": "density",
            "observed_order": 1.95,
            "threshold": 1.8,
        }
    ],
    "amr": {
        "order_retained": None,
        "invariants_ok": None,
        "interface_error": None,
        "bulk_error": None,
    },
    "poisson": {
        "potential_error": None,
        "field_error": None,
        "residual_l2": None,
    },
    "coupling": {
        "phase_error": None,
        "sign_ok": None,
        "energy_drift": None,
    },
    "parallel_invariance": {
        "ranks_ok": None,
        "threads_ok": None,
        "gpu_ok": None,
    },
    "performance": {
        "one_node": None,
        "two_node": None,
    },
    "not_applicable_reason": {
        "amr.*": "AMR not run in local pr",
        "poisson.*": "Poisson campaign not run in local pr",
        "coupling.*": "coupling not run in local pr",
        "parallel_invariance.*": "parallel invariance not run in local pr",
        "performance.one_node": "performance not measured in local pr",
        "performance.two_node": "performance not measured in local pr",
    },
    "artifacts": {
        "report_md": "REPORT.md",
        "summary_json": "summary.json",
        "coverage_csv": "coverage.csv",
        "failures_csv": "failures.csv",
    },
}

REQUIRED_TOP_LEVEL = (
    "schema",
    "repository",
    "repository_sha",
    "suite",
    "max_nodes",
    "native_dimensions",
    "execution_spaces",
    "coverage",
    "failures",
    "orders",
    "amr",
    "poisson",
    "coupling",
    "parallel_invariance",
    "performance",
    "not_applicable_reason",
    "artifacts",
)

REQUIRED_MISSING_MUST_REJECT = (
    "coverage",
    "failures",
    "orders",
    "not_applicable_reason",
)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pr_summary() -> dict:
    return copy.deepcopy(LOCAL_PR_SUMMARY)


def _release_summary() -> dict:
    instance = _pr_summary()
    instance["suite"] = "release"
    instance["native_dimensions"] = [1, 2, 3]
    instance["execution_spaces"] = [
        "KokkosSerial",
        "KokkosOpenMP",
        "KokkosCuda",
    ]
    instance["coverage"] = {
        "components": ["euler", "poisson"],
        "cases_planned": 2,
        "cases_run": 2,
        "cases_passed": 1,
        "cases_failed": 1,
        "cases_not_supported": 0,
        "not_tested": ["weekly-only polar mode"],
    }
    instance["failures"] = [
        {
            "case_id": "CP-02",
            "reason": "spatial order below threshold",
            "metrics_ref": "CP-02/metrics.json",
            "provenance_ref": "CP-02/provenance.json",
        }
    ]
    instance["amr"] = {
        "order_retained": True,
        "invariants_ok": True,
        "interface_error": 1.0e-12,
        "bulk_error": 1.0e-10,
    }
    instance["poisson"] = {
        "potential_error": 1.0e-8,
        "field_error": 1.0e-7,
        "residual_l2": 1.0e-10,
    }
    instance["coupling"] = {
        "phase_error": 0.02,
        "sign_ok": True,
        "energy_drift": 1.0e-6,
    }
    instance["parallel_invariance"] = {
        "ranks_ok": True,
        "threads_ok": True,
        "gpu_ok": False,
    }
    instance["performance"] = {
        "one_node": {
            "cells_per_second": 1.2e6,
            "notes": "local serial baseline",
        },
        "two_node": None,
    }
    instance["not_applicable_reason"] = {
        "performance.two_node": "two-node performance not collected",
    }
    return instance


def _null_paths(obj, prefix: str = "") -> list[str]:
    if obj is None:
        return [prefix] if prefix else []
    if isinstance(obj, dict):
        paths: list[str] = []
        for key, value in obj.items():
            if key == "not_applicable_reason":
                continue
            child = f"{prefix}.{key}" if prefix else key
            paths.extend(_null_paths(value, child))
        return paths
    if isinstance(obj, list):
        paths = []
        for index, value in enumerate(obj):
            paths.extend(_null_paths(value, f"{prefix}.{index}"))
        return paths
    return []


def _reason_covers(path: str, reasons: dict) -> bool:
    if path in reasons:
        return True
    parts = path.split(".")
    for key in reasons:
        key_parts = key.split(".")
        if len(key_parts) != len(parts):
            continue
        if all(
            token == "*" or token == part
            for token, part in zip(key_parts, parts, strict=True)
        ):
            return True
    return False


def unjustified_nulls(instance: dict) -> list[str]:
    reasons = instance.get("not_applicable_reason") or {}
    missing = [
        path for path in _null_paths(instance) if not _reason_covers(path, reasons)
    ]
    if instance.get("orders") == [] and not _reason_covers("orders", reasons):
        missing.append("orders")
    return missing


def test_local_pr_summary_is_valid():
    instance = _pr_summary()
    assert instance["suite"] == "pr"
    assert instance["native_dimensions"] == [1]
    assert instance["execution_spaces"] == ["KokkosSerial"]
    assert instance["coverage"]["cases_planned"] == 1
    assert instance["coverage"]["cases_run"] == 1
    assert instance["coverage"]["cases_passed"] == 1
    assert instance["failures"] == []
    assert instance["orders"][0]["kind"] == "spatial"
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_release_summary_with_failures_is_valid():
    instance = _release_summary()
    assert instance["suite"] == "release"
    assert instance["failures"]
    assert instance["amr"]["interface_error"] == 1.0e-12
    assert instance["poisson"]["residual_l2"] == 1.0e-10
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_optional_artifact_extra_paths_are_accepted():
    instance = _pr_summary()
    instance["artifacts"]["orders_csv"] = "orders.csv"
    instance["artifacts"]["amr_plot"] = "figures/amr.png"
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_performance_node_object_is_valid():
    instance = _pr_summary()
    instance["performance"]["one_node"] = {
        "cells_per_second": 8.0e5,
        "notes": "pr serial",
        "wall_seconds": 12.5,
    }
    del instance["not_applicable_reason"]["performance.one_node"]
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_empty_orders_with_reason_is_valid():
    instance = _pr_summary()
    instance["orders"] = []
    instance["not_applicable_reason"]["orders"] = "no order campaign ran"
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_empty_failures_array_is_valid():
    instance = _pr_summary()
    assert instance["failures"] == []
    _validator().validate(instance)


def test_failure_empty_refs_are_accepted():
    instance = _release_summary()
    instance["failures"][0]["metrics_ref"] = ""
    instance["failures"][0]["provenance_ref"] = ""
    _validator().validate(instance)


def test_rejects_wrong_schema_const():
    instance = _pr_summary()
    instance["schema"] = "pops.verification.report.v0"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_max_nodes_not_two():
    instance = _pr_summary()
    instance["max_nodes"] = 1
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("key", REQUIRED_MISSING_MUST_REJECT)
def test_rejects_missing_coverage_failures_orders_or_reasons(key):
    instance = _pr_summary()
    del instance[key]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("key", REQUIRED_TOP_LEVEL)
def test_rejects_missing_required_top_level_key(key):
    instance = _pr_summary()
    del instance[key]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("suite", ["", "daily", "PR", "ci"])
def test_rejects_suite_not_in_enum(suite):
    instance = _pr_summary()
    instance["suite"] = suite
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_null_first_page_field_without_reason():
    instance = _pr_summary()
    del instance["not_applicable_reason"]["performance.two_node"]
    assert "performance.two_node" in unjustified_nulls(instance)


def test_rejects_empty_orders_without_reason():
    instance = _pr_summary()
    instance["orders"] = []
    assert "orders" in unjustified_nulls(instance)
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_empty_repository_sha():
    instance = _pr_summary()
    instance["repository_sha"] = ""
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_duplicate_native_dimensions():
    instance = _pr_summary()
    instance["native_dimensions"] = [1, 1]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_unknown_execution_space():
    instance = _pr_summary()
    instance["execution_spaces"] = ["KokkosHIP"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_empty_coverage_components():
    instance = _pr_summary()
    instance["coverage"]["components"] = []
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_missing_artifact_path():
    instance = _pr_summary()
    del instance["artifacts"]["report_md"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_extra_key_on_nested_summary():
    instance = _pr_summary()
    instance["amr"]["extra"] = True
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_unknown_order_kind():
    instance = _pr_summary()
    instance["orders"][0]["kind"] = "spectral"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_performance_object_missing_cells_per_second():
    instance = _pr_summary()
    instance["performance"]["one_node"] = {"notes": "missing rate"}
    del instance["not_applicable_reason"]["performance.one_node"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)
