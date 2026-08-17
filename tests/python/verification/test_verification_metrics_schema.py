"""JSON Schema for per-run verification metrics (plan §6.1)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_metrics.v1.json"

# Plan §6.1 minimal example — field names and values verbatim.
PLAN_SECTION_6_1_EXAMPLE = {
    "schema": "pops.verification.metrics.v1",
    "case_id": "CP-02",
    "errors": {
        "density": {"l1": 0.0, "l2": 0.0, "linf": 0.0, "observed_order": None},
        "velocity_x": {"l1": 0.0, "l2": 0.0, "linf": 0.0, "observed_order": None},
        "electric_field_x": {"l1": 0.0, "l2": 0.0, "linf": 0.0, "observed_order": None},
    },
    "conservation": {
        "mass_total": {"initial": 0.0, "final": 0.0, "max_relative_drift": 0.0},
        "momentum_total": {
            "initial": [0.0],
            "final": [0.0],
            "max_relative_drift": 0.0,
        },
        "energy_total": {"initial": 0.0, "final": 0.0, "max_relative_drift": 0.0},
        "charge_total": {"initial": 0.0, "final": 0.0, "max_absolute_drift": 0.0},
        "electrostatic_energy": {
            "initial": 0.0,
            "final": 0.0,
            "max_relative_drift": 0.0,
        },
    },
    "poisson": {"residual_l2": 0.0, "gauss_defect_l2": 0.0},
    "extrema": {"rho_min": 0.0, "p_min": None},
    "symmetry": {"error": None},
    "amr": {
        "interface_error": None,
        "bulk_error": 0.0,
        "leaf_cells": 128,
        "patch_count": 1,
        "regrid_count": 0,
    },
    "not_applicable_reason": {
        "errors.*.observed_order": "single-resolution run",
        "extrema.p_min": "cold-fluid model has no pressure variable",
        "symmetry.error": "one-dimensional mode",
        "amr.interface_error": "uniform-grid configuration",
    },
    "timings_seconds": {"ghost_fill": 0.0, "poisson": 0.0, "reflux": 0.0},
}

REQUIRED_TOP_LEVEL = (
    "schema",
    "case_id",
    "errors",
    "conservation",
    "poisson",
    "extrema",
    "symmetry",
    "amr",
    "not_applicable_reason",
    "timings_seconds",
)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _plan_example() -> dict:
    return copy.deepcopy(PLAN_SECTION_6_1_EXAMPLE)


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
    return [path for path in _null_paths(instance) if not _reason_covers(path, reasons)]


def test_plan_section_6_1_example_is_valid():
    instance = _plan_example()
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def _no_poisson_variant() -> dict:
    instance = _plan_example()
    instance["timings_seconds"]["poisson"] = 0.0
    instance["poisson"]["residual_l2"] = None
    instance["poisson"]["gauss_defect_l2"] = None
    instance["conservation"]["electrostatic_energy"] = {
        "initial": None,
        "final": None,
        "max_relative_drift": None,
    }
    instance["not_applicable_reason"]["poisson.residual_l2"] = "no Poisson equation"
    instance["not_applicable_reason"]["poisson.gauss_defect_l2"] = "no Poisson equation"
    instance["not_applicable_reason"]["conservation.electrostatic_energy.*"] = (
        "no Poisson equation"
    )
    return instance


def test_no_poisson_variant_is_valid():
    instance = _no_poisson_variant()
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_rejects_no_poisson_variant_with_null_poisson_time():
    instance = _no_poisson_variant()
    instance["timings_seconds"]["poisson"] = None
    instance["not_applicable_reason"]["timings_seconds.poisson"] = "no Poisson equation"
    assert unjustified_nulls(instance) == []
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_no_amr_uniform_grid_variant_is_valid():
    instance = _plan_example()
    assert instance["amr"]["leaf_cells"] == 128
    assert instance["amr"]["patch_count"] == 1
    assert instance["amr"]["regrid_count"] == 0
    assert instance["amr"]["interface_error"] is None
    assert (
        instance["not_applicable_reason"]["amr.interface_error"]
        == "uniform-grid configuration"
    )
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_optional_time_series_and_csv_files_are_accepted():
    instance = _plan_example()
    instance["time_series"] = {"mass_total": [1.0, 1.0]}
    instance["csv_files"] = ["analysis/errors.csv", "analysis/conservation.csv"]
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_errors_accepts_case_specific_variable_names():
    instance = _plan_example()
    instance["errors"] = {
        "pressure": {"l1": 0.1, "l2": 0.05, "linf": 0.2, "observed_order": 1.9}
    }
    del instance["not_applicable_reason"]["errors.*.observed_order"]
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_no_analytic_reference_allows_null_norms_with_reasons():
    instance = _plan_example()
    instance["errors"] = {
        "density": {
            "l1": None,
            "l2": None,
            "linf": None,
            "observed_order": None,
        }
    }
    instance["not_applicable_reason"]["errors.*.l1"] = "no analytic reference"
    instance["not_applicable_reason"]["errors.*.l2"] = "no analytic reference"
    instance["not_applicable_reason"]["errors.*.linf"] = "no analytic reference"
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


def test_extrema_may_be_null_with_reasons():
    instance = _plan_example()
    instance["extrema"]["rho_min"] = None
    instance["not_applicable_reason"]["extrema.rho_min"] = "no density equation"
    _validator().validate(instance)
    assert unjustified_nulls(instance) == []


@pytest.mark.parametrize("key", REQUIRED_TOP_LEVEL)
def test_rejects_missing_required_top_level_key(key):
    instance = _plan_example()
    del instance[key]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_missing_linf_on_variable():
    instance = _plan_example()
    del instance["errors"]["density"]["linf"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_missing_leaf_cells():
    instance = _plan_example()
    del instance["amr"]["leaf_cells"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_empty_errors_object():
    instance = _plan_example()
    instance["errors"] = {}
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_null_metric_without_reason():
    instance = _plan_example()
    del instance["not_applicable_reason"]["extrema.p_min"]
    assert "extrema.p_min" in unjustified_nulls(instance)


def test_rejects_wrong_schema_const():
    instance = _plan_example()
    instance["schema"] = "pops.verification.metrics.v0"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_extra_key_on_nested_summary():
    instance = _plan_example()
    instance["poisson"]["extra"] = 0.0
    with pytest.raises(ValidationError):
        _validator().validate(instance)
