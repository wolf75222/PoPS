"""Render a run into the Phase 8 analysis tree (plan §40.2, §40.10)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.visualization.fixtures import write_fixture_run
from verification.pops_verify.visualization.render import render_run

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"

import matplotlib
matplotlib.use("Agg", force=True)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def test_fixture_run_is_labeled_and_schema_valid(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["data_kind"] == "deterministic_fixture"
    assert status["verdict"] == "pass"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["case_id"] == "TR-01"
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["dimension"] == 1


def test_render_tr01_fixture_writes_manifest_and_publication_figures(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    result = render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    manifest_path = run / "analysis" / "visual_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validator().validate(manifest)
    assert manifest["data_kind"] == "deterministic_fixture"
    assert manifest["verdict"] == "pass"
    assert manifest["does_not_prove"].lower().startswith("this fixture")
    figure_ids = {item["figure_id"] for item in manifest["figures"]}
    assert "spatial_convergence" in figure_ids
    assert "reference_profile" in figure_ids
    conv = next(item for item in manifest["figures"] if item["figure_id"] == "spatial_convergence")
    svg = (run / conv["outputs"]["svg"]).read_text(encoding="utf-8")
    assert "DETERMINISTIC FIXTURE" in svg
    assert provenance_sha_in_caption(svg, manifest["repository_sha"])
    assert result["visual_manifest"].endswith("visual_manifest.json")


def provenance_sha_in_caption(svg: str, sha: str) -> bool:
    return sha in svg


def test_render_identical_fixture_twice_is_stable(tmp_path: Path):
    first = write_fixture_run(tmp_path / "a", "PO-01", dimension=1)
    second = write_fixture_run(tmp_path / "b", "PO-01", dimension=1)
    render_run(first, suite="pr", formats=("svg",))
    render_run(second, suite="pr", formats=("svg",))
    left = json.loads((first / "analysis" / "visual_manifest.json").read_text())
    right = json.loads((second / "analysis" / "visual_manifest.json").read_text())
    left_hashes = {
        item["figure_id"]: item["output_hashes"].get("svg") for item in left["figures"]
    }
    right_hashes = {
        item["figure_id"]: item["output_hashes"].get("svg") for item in right["figures"]
    }
    assert left_hashes == right_hashes
    assert all(digest for digest in left_hashes.values())


def test_render_refuses_missing_source_data(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    (run / "analysis" / "visual_data" / "spatial_convergence.json").unlink()
    render_run(run, suite="pr", formats=("svg",))
    manifest = json.loads((run / "analysis" / "visual_manifest.json").read_text())
    assert manifest["verdict"] != "pass"
    ids = {item["figure_id"] for item in manifest["figures"]}
    assert "spatial_convergence" not in ids
    assert not list((run / "analysis").rglob("spatial_convergence.svg"))


def test_not_run_writes_manifest_without_fake_curves(tmp_path: Path):
    run = write_fixture_run(tmp_path, "IF-01", dimension=1, verdict="not-run")
    result = render_run(run, suite="pr", formats=("svg",))
    manifest = json.loads(Path(result["visual_manifest"]).read_text(encoding="utf-8"))
    assert manifest["verdict"] == "not-run"
    assert manifest["figures"] == []
    figures_dir = run / "analysis" / "figures"
    if figures_dir.exists():
        assert list(figures_dir.rglob("*.svg")) == []
