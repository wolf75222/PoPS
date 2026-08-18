"""Phase 8 completeness and quality gates (plan §40.5, §40.10.5, §40.11)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verification.pops_verify.visualization.fixtures import write_fixture_run
from verification.pops_verify.visualization.gates import (
    VisualsGateError,
    check_release_completeness,
    check_visual_manifest,
)
from verification.pops_verify.visualization.render import render_run

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def test_rendered_pr_manifest_passes_quality_gate(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    render_run(run, suite="pr", formats=("svg", "png"))
    check_visual_manifest(run, suite="pr")


def test_publication_figure_missing_format_fails_gate(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    manifest_path = run / "analysis" / "visual_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publication = next(item for item in manifest["figures"] if item["role"] == "publication")
    publication["outputs"].pop("pdf", None)
    publication["output_hashes"].pop("pdf", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(VisualsGateError, match="required formats|required hashes"):
        check_visual_manifest(run, suite="pr")


def test_hero_without_quantitative_companion_fails(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    render_run(run, suite="pr", formats=("svg",))
    manifest_path = run / "analysis" / "visual_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"].append(
        {
            **manifest["figures"][0],
            "figure_id": "hero_figure",
            "kind": "hero_figure",
            "role": "hero",
            "quantitative_companion": None,
            "pr": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(VisualsGateError, match="quantitative"):
        check_visual_manifest(run, suite="pr")


def test_release_gate_requires_every_r_cell_or_justification(tmp_path: Path):
    campaign = tmp_path / "campaign"
    tr01 = write_fixture_run(campaign, "TR-01", dimension=1)
    render_run(tr01, suite="release", formats=("svg", "png", "pdf"))
    with pytest.raises(VisualsGateError, match="TR-01"):
        check_release_completeness(campaign, suite="release")


def test_not_run_r_dimension_is_counted_not_invented(tmp_path: Path):
    campaign = tmp_path / "campaign"
    run = write_fixture_run(campaign, "RB-09", dimension=1, verdict="not-run")
    render_run(run, suite="release", formats=("svg",))
    report = check_release_completeness(
        campaign,
        suite="release",
        executed_only=True,
    )
    assert report["cells_checked"] >= 1
    assert report["missing"] == []
