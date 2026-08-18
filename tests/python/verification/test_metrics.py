"""Writer for per-run pops.verification.metrics.v1 documents (plan §6.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_metrics.v1.json"


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_collect_metrics_writes_schema_valid_null_document():
    from verification.pops_verify.metrics import collect_metrics

    document = collect_metrics(
        "IF-08",
        reason="runner foundation records no scientific field yet",
    )
    assert document["schema"] == "pops.verification.metrics.v1"
    assert document["case_id"] == "IF-08"
    _validator().validate(document)
    assert document["timings_seconds"]["poisson"] == 0


def test_write_metrics_round_trip(tmp_path: Path):
    from verification.pops_verify.metrics import collect_metrics, write_metrics

    document = collect_metrics("IF-01", reason="no live MPI field in runner foundation")
    path = tmp_path / "metrics.json"
    write_metrics(path, document)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == document
    _validator().validate(loaded)


def test_write_metrics_refuses_invalid_document(tmp_path: Path):
    from verification.pops_verify.metrics import MetricsError, write_metrics

    path = tmp_path / "metrics.json"
    with pytest.raises(MetricsError):
        write_metrics(path, {"schema": "pops.verification.metrics.v1"})
    assert not path.exists()
