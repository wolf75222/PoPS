"""Second Phase 8 re-review findings (plan v1.5 §40.2, §40.6.11, §40.7)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from test_verification_report_schema import LOCAL_PR_SUMMARY
from verification.pops_verify.report import write_verification_report
from verification.pops_verify.visualization.catalog import (
    SCIENTIFIC_CASE_IDS,
    catalog_entry,
    visual_contract_for,
)
from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.fixtures import write_fixture_run
from verification.pops_verify.visualization.gallery import render_release_gallery
from verification.pops_verify.visualization.gates import VisualsGateError, check_visual_manifest
from verification.pops_verify.visualization.plots import file_sha256
from verification.pops_verify.visualization.render import render_run

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"
ISOSURFACE_FAMILIES = frozenset({"TR", "EU", "PO", "AM", "CP", "RB", "GE", "IF"})

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _manifest(run: Path) -> dict:
    return json.loads((run / "analysis" / "visual_manifest.json").read_text(encoding="utf-8"))


def _figure(manifest: dict, figure_id: str) -> dict:
    return next(item for item in manifest["figures"] if item["figure_id"] == figure_id)


def test_figure_metadata_comes_from_provenance_and_payload(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    manifest = _manifest(run)
    profile = _figure(manifest, "reference_profile")
    assert profile["resolutions"] == provenance["resolution"]
    assert profile["resolutions"] != [16, 32, 64, 128]
    assert profile["amr_levels"] == list(range(int(provenance["amr_total_levels"])))
    payload = json.loads(
        (run / "analysis" / "visual_data" / "reference_profile.json").read_text()
    )
    assert profile["times"] == payload["times"]
    assert profile["step_numbers"] == payload["step_numbers"]
    assert profile["times"] != [1.0] or payload["times"] == [1.0]
    conv = _figure(manifest, "spatial_convergence")
    assert conv["resolutions"] == [16, 32, 64, 128]
    assert conv["times"] == json.loads(
        (run / "analysis" / "visual_data" / "spatial_convergence.json").read_text()
    )["times"]


def test_field_color_range_is_taken_from_rendered_limits(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=2)
    render_run(run, suite="release", formats=("svg",))
    signed = _figure(_manifest(run), "signed_error_field")
    assert signed["color_range"] is not None
    assert signed["color_range"][0] < 0 < signed["color_range"][1]
    assert signed["color_range"][0] == -signed["color_range"][1]


def test_digests_hash_identity_files_not_unrelated_provenance(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    render_run(run, suite="pr", formats=("svg",))
    manifest = _manifest(run)
    assert manifest["resolved_case_digest"] == file_sha256(run / "resolved_case.json")
    assert manifest["program_digest"] == file_sha256(run / "program.json")
    assert manifest["native_artifact_digest"] == file_sha256(run / "native_artifact.json")
    assert manifest["resolved_case_digest"] != provenance["component_catalog_digest"]
    assert manifest["program_digest"] != provenance["native_header_signature"]
    assert manifest["native_artifact_digest"] != provenance["native_variant_manifest_digest"]
    profile = _figure(manifest, "reference_profile")
    assert profile["resolved_case_digest"] == manifest["resolved_case_digest"]


def test_missing_resolved_case_refuses_remap(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    identity = run / "resolved_case.json"
    if identity.exists():
        identity.unlink()
    with pytest.raises(VisualsError, match="resolved_case"):
        render_run(run, suite="pr", formats=("svg",))


def test_section_40_6_11_3d_contracts_declare_isosurface():
    for case_id in SCIENTIFIC_CASE_IDS:
        family = case_id.split("-", 1)[0]
        if family not in ISOSURFACE_FAMILIES:
            continue
        entry = catalog_entry(case_id)
        if entry.dimension_codes[2] == "N/A":
            continue
        assert "isosurface" in entry.artifacts["3d"], case_id
        contract = visual_contract_for(case_id)
        declared = contract["dimensions"]["3d"].get("required") or contract[
            "dimensions"
        ]["3d"].get("required_when_executed")
        assert "isosurface" in declared, case_id


def test_missing_isosurface_engine_emits_explicit_not_run_evidence(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=3)
    render_run(run, suite="release", formats=("svg", "png", "pdf"))
    manifest = _manifest(run)
    item = _figure(manifest, "isosurface")
    assert item["verdict"] in {"not-run", "not-supported"}
    assert item["outputs"]
    evidence = run / item["outputs"][next(iter(item["outputs"]))]
    assert evidence.is_file() and evidence.stat().st_size > 0
    text = evidence.read_text(encoding="utf-8") if evidence.suffix == ".svg" else ""
    if text:
        assert "not-run" in text.lower() or "not-supported" in text.lower()
    assert "isosurface" in manifest["dimensions"]["3d"]["artifacts"]


def test_exact_and_numerical_fields_share_one_color_range(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=2)
    render_run(run, suite="release", formats=("svg",))
    manifest = _manifest(run)
    exact = _figure(manifest, "exact_field")
    numerical = _figure(manifest, "numerical_field")
    assert exact["color_range"] == numerical["color_range"]
    assert exact["color_range"] is not None
    assert exact["color_range"][0] < exact["color_range"][1]


def test_release_gate_rejects_mismatched_field_colorbars(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=2)
    render_run(run, suite="release", formats=("svg", "png", "pdf"))
    manifest_path = run / "analysis" / "visual_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact = next(item for item in manifest["figures"] if item["figure_id"] == "exact_field")
    exact["color_range"] = [0.0, 1.0]
    numerical = next(
        item for item in manifest["figures"] if item["figure_id"] == "numerical_field"
    )
    numerical["color_range"] = [0.0, 2.0]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(VisualsGateError, match="colorbar|color_range"):
        check_visual_manifest(run, suite="release")


def test_gallery_report_section_is_idempotent_and_emits_manifest(tmp_path: Path):
    summary = copy.deepcopy(LOCAL_PR_SUMMARY)
    summary["repository_sha"] = "fixture:pops-visuals-v1"
    write_verification_report(summary, tmp_path)
    render_release_gallery(tmp_path, formats=("svg",))
    render_release_gallery(tmp_path, formats=("svg",))
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert report.count("## Visual gallery") == 1
    manifest_path = tmp_path / "analysis" / "visual_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
        manifest
    )
    assert manifest["schema"] == "pops.verification.visual_manifest.v1"
    assert manifest["verdict"] != "pass"
    for item in manifest["figures"]:
        assert item["verdict"] in {"pass", "fail", "not-run", "not-supported"}
        if item["verdict"] in {"not-run", "not-supported"}:
            assert item["kind"] != "spatial_convergence" or "1.95" not in json.dumps(item)


def test_storyboard_events_follow_visual_contract(tmp_path: Path):
    for case_id in ("TR-01", "AM-01"):
        run = write_fixture_run(tmp_path / case_id, case_id, dimension=2)
        render_run(run, suite="release", formats=("svg",))
        expected = list(visual_contract_for(case_id)["animation"]["key_events"])
        payload = json.loads(
            (run / "analysis" / "visual_data" / "storyboard.json").read_text()
        )
        events = [frame["event"] for frame in payload["frames"]]
        assert events == expected, case_id
        svg = (run / "analysis" / "storyboards" / "storyboard.svg").read_text()
        for event in expected:
            assert event in svg, (case_id, event)
    assert visual_contract_for("TR-01")["animation"]["key_events"] != visual_contract_for(
        "AM-01"
    )["animation"]["key_events"]
