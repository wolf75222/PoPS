"""Release gallery and transverse dashboards (plan §40.7, §40.8)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from test_verification_report_schema import LOCAL_PR_SUMMARY
from verification.pops_verify.report import write_verification_report
from verification.pops_verify.visualization.gallery import render_release_gallery

import matplotlib
matplotlib.use("Agg", force=True)


def test_gallery_preserves_not_applicable_orders(tmp_path: Path):
    summary = copy.deepcopy(LOCAL_PR_SUMMARY)
    summary["orders"] = []
    summary["not_applicable_reason"]["orders"] = "order campaign not run"
    write_verification_report(summary, tmp_path)
    outputs = render_release_gallery(tmp_path, formats=("svg",))
    svg = Path(outputs["orders_heatmap"]).read_text(encoding="utf-8")
    assert "not applicable" in svg.lower() or "not run" in svg.lower()
    assert "1.95" not in svg


def test_gallery_shows_measured_order_and_failure(tmp_path: Path):
    summary = copy.deepcopy(LOCAL_PR_SUMMARY)
    summary["failures"] = [
        {
            "case_id": "CP-02",
            "reason": "spatial order below threshold",
            "metrics_ref": "CP-02/metrics.json",
            "provenance_ref": "CP-02/provenance.json",
        }
    ]
    write_verification_report(summary, tmp_path)
    outputs = render_release_gallery(tmp_path, formats=("svg",))
    orders = Path(outputs["orders_heatmap"]).read_text(encoding="utf-8")
    assert "1.95" in orders
    failures = Path(outputs["failure_map"]).read_text(encoding="utf-8")
    assert "CP-02" in failures
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "Visual gallery" in report
    assert "orders_heatmap" in report
