"""Fail-closed visual_data loading (plan §40.3, §40.4, §40.11)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verification.pops_verify.metrics import collect_metrics, write_metrics
from verification.pops_verify.visualization.data import (
    VisualsError,
    load_run_bundle,
    load_visual_series,
)

from test_verification_metrics_schema import PLAN_SECTION_6_1_EXAMPLE
from test_verification_provenance_schema import PLAN_SECTION_6_2_EXAMPLE


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _campaign_run(tmp_path: Path) -> Path:
    run = tmp_path / "TR-01" / "campaign-run"
    run.mkdir(parents=True)
    write_metrics(run / "metrics.json", PLAN_SECTION_6_1_EXAMPLE)
    _write_json(run / "provenance.json", PLAN_SECTION_6_2_EXAMPLE)
    _write_json(
        run / "status.json",
        {"verdict": "pass", "case_id": "TR-01", "run_id": "campaign-run"},
    )
    _write_json(
        run / "analysis" / "visual_data" / "spatial_convergence.json",
        {
            "figure_id": "spatial_convergence",
            "kind": "spatial_convergence",
            "data_kind": "campaign",
            "verdict": "pass",
            "units": {"x": "1/h", "y": "L2 error"},
            "variables": ["scalar"],
            "series": [
                {
                    "name": "L2",
                    "x": [16, 32, 64, 128],
                    "y": [1.6e-2, 4.0e-3, 1.0e-3, 2.5e-4],
                    "unit": "1",
                }
            ],
            "reference_slopes": [{"order": 2, "anchor": [16, 1.6e-2]}],
        },
    )
    return run


def test_load_campaign_bundle_requires_schema_valid_metrics_and_provenance(tmp_path):
    run = _campaign_run(tmp_path)
    bundle = load_run_bundle(run)
    assert bundle.data_kind == "campaign"
    assert bundle.verdict == "pass"
    assert bundle.metrics["schema"] == "pops.verification.metrics.v1"
    assert bundle.provenance["schema"] == "pops.verification.provenance.v1"
    assert bundle.provenance["repository_sha"]


def test_missing_metrics_fails_closed(tmp_path):
    run = _campaign_run(tmp_path)
    (run / "metrics.json").unlink()
    with pytest.raises(VisualsError, match="metrics"):
        load_run_bundle(run)


def test_missing_visual_series_fails_closed(tmp_path):
    run = _campaign_run(tmp_path)
    with pytest.raises(VisualsError, match="visual_data"):
        load_visual_series(run, "signed_error_field")


def test_empty_series_fails_closed(tmp_path):
    run = _campaign_run(tmp_path)
    _write_json(
        run / "analysis" / "visual_data" / "spatial_convergence.json",
        {
            "figure_id": "spatial_convergence",
            "kind": "spatial_convergence",
            "data_kind": "campaign",
            "verdict": "pass",
            "units": {"x": "1/h", "y": "L2 error"},
            "variables": ["scalar"],
            "series": [{"name": "L2", "x": [], "y": [], "unit": "1"}],
        },
    )
    with pytest.raises(VisualsError, match="empty"):
        load_visual_series(run, "spatial_convergence")


def test_not_run_verdict_is_preserved(tmp_path):
    run = _campaign_run(tmp_path)
    write_metrics(
        run / "metrics.json",
        collect_metrics("TR-01", reason="case not executed in this campaign"),
    )
    _write_json(
        run / "status.json",
        {"verdict": "not-run", "case_id": "TR-01", "run_id": "campaign-run"},
    )
    bundle = load_run_bundle(run)
    assert bundle.verdict == "not-run"


def test_fixture_bundle_must_be_labeled(tmp_path):
    run = tmp_path / "TR-01" / "fixture-run"
    run.mkdir(parents=True)
    write_metrics(run / "metrics.json", PLAN_SECTION_6_1_EXAMPLE)
    _write_json(run / "provenance.json", PLAN_SECTION_6_2_EXAMPLE)
    _write_json(
        run / "status.json",
        {
            "verdict": "pass",
            "case_id": "TR-01",
            "run_id": "fixture-run",
            "data_kind": "deterministic_fixture",
        },
    )
    bundle = load_run_bundle(run)
    assert bundle.data_kind == "deterministic_fixture"
    assert "FIXTURE" in bundle.data_kind_label.upper()


def test_campaign_label_cannot_be_spoofed_by_fixture_series(tmp_path):
    run = _campaign_run(tmp_path)
    payload = json.loads(
        (run / "analysis" / "visual_data" / "spatial_convergence.json").read_text()
    )
    payload["data_kind"] = "deterministic_fixture"
    _write_json(run / "analysis" / "visual_data" / "spatial_convergence.json", payload)
    with pytest.raises(VisualsError, match="data_kind"):
        load_visual_series(run, "spatial_convergence")
