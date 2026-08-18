"""Build and validate a per-run pops.verification.metrics.v1 document.

Plan §6.1: the same fields are always present. A non-applicable quantity is
null and must have a justification in not_applicable_reason.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_metrics.v1.json"
SCHEMA_ID = "pops.verification.metrics.v1"


class MetricsError(RuntimeError):
    pass


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_document(document: Mapping[str, Any]) -> None:
    try:
        _validator().validate(document)
    except ValidationError as exc:
        raise MetricsError(f"invalid metrics document: {exc.message}") from exc


def collect_metrics(
    case_id: str,
    *,
    reason: str,
    errors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a schema-valid metrics document with justified nulls."""
    if not isinstance(case_id, str) or not case_id:
        raise MetricsError("case_id must be a non-empty string")
    if not isinstance(reason, str) or not reason:
        raise MetricsError("reason must be a non-empty string")
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "case_id": case_id,
        "errors": dict(errors)
        if errors is not None
        else {
            "status": {
                "l1": None,
                "l2": None,
                "linf": None,
                "observed_order": None,
            }
        },
        "conservation": {
            "mass_total": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
            "momentum_total": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
            "energy_total": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
            "charge_total": {
                "initial": None,
                "final": None,
                "max_absolute_drift": None,
            },
            "electrostatic_energy": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
        },
        "poisson": {"residual_l2": None, "gauss_defect_l2": None},
        "extrema": {"rho_min": None, "p_min": None},
        "symmetry": {"error": None},
        "amr": {
            "interface_error": None,
            "bulk_error": None,
            "leaf_cells": None,
            "patch_count": None,
            "regrid_count": None,
        },
        "not_applicable_reason": {
            "errors.*.l1": reason,
            "errors.*.l2": reason,
            "errors.*.linf": reason,
            "errors.*.observed_order": reason,
            "conservation.*.*": reason,
            "poisson.*": reason,
            "extrema.*": reason,
            "symmetry.error": reason,
            "amr.*": reason,
            "timings_seconds.ghost_fill": reason,
            "timings_seconds.reflux": reason,
        },
        "timings_seconds": {"ghost_fill": None, "poisson": 0, "reflux": None},
    }
    _validate_document(document)
    return document


def write_metrics(path: str | Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` as JSON after schema validation. Refuse invalid input."""
    _validate_document(document)
    Path(path).write_text(json.dumps(dict(document), indent=2) + "\n", encoding="utf-8")
