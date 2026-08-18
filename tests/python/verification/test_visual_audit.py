"""Audit fixes for Phase 8 visual semantics (plan v1.5 §40)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from test_verification_report_schema import LOCAL_PR_SUMMARY
from verification.pops_verify.report import write_verification_report
from verification.pops_verify.visualization.catalog import visual_contract_for
from verification.pops_verify.visualization.fixtures import write_fixture_run
from verification.pops_verify.visualization.gallery import render_release_gallery
from verification.pops_verify.visualization.plots import derive_figure_verdict
from verification.pops_verify.visualization.render import render_run

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_SHA_PREFIX = "fixture:"

import matplotlib
matplotlib.use("Agg", force=True)


def _manifest(run: Path) -> dict:
    return json.loads((run / "analysis" / "visual_manifest.json").read_text(encoding="utf-8"))


def _figure(manifest: dict, figure_id: str) -> dict:
    return next(item for item in manifest["figures"] if item["figure_id"] == figure_id)


def test_repo_does_not_commit_generated_phase8_binaries():
    examples = REPO_ROOT / "verification" / "examples" / "phase8"
    generated = []
    if examples.exists():
        generated = [
            path
            for path in examples.rglob("*")
            if path.suffix.lower() in {".png", ".pdf", ".svg", ".mp4", ".gif"}
        ]
    assert generated == []


def test_order1_data_cannot_pass_order2_reference():
    payload = {
        "kind": "spatial_convergence",
        "units": {"x": "1/h", "y": "L2 error"},
        "series": [
            {
                "name": "L2",
                "x": [16, 32, 64, 128],
                "y": [1.6e-2, 8.0e-3, 4.0e-3, 2.0e-3],
                "unit": "1",
            }
        ],
        "reference_slopes": [{"order": 2, "anchor": [16, 1.6e-2]}],
    }
    assert derive_figure_verdict(payload) == "fail"


def test_order2_data_can_pass_order2_reference():
    payload = {
        "kind": "spatial_convergence",
        "units": {"x": "1/h", "y": "L2 error"},
        "series": [
            {
                "name": "L2",
                "x": [16, 32, 64, 128],
                "y": [1.6e-2, 4.0e-3, 1.0e-3, 2.5e-4],
                "unit": "1",
            }
        ],
        "reference_slopes": [{"order": 2, "anchor": [16, 1.6e-2]}],
    }
    assert derive_figure_verdict(payload) == "pass"


def test_rendered_convergence_verdict_matches_data(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    visual = json.loads(
        (run / "analysis" / "visual_data" / "spatial_convergence.json").read_text()
    )
    visual["series"] = [
        {
            "name": "L2",
            "x": [16, 32, 64, 128],
            "y": [1.6e-2, 8.0e-3, 4.0e-3, 2.0e-3],
            "unit": "1",
        }
    ]
    visual["reference_slopes"] = [{"order": 2, "anchor": [16, 1.6e-2]}]
    (run / "analysis" / "visual_data" / "spatial_convergence.json").write_text(
        json.dumps(visual, indent=2) + "\n", encoding="utf-8"
    )
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    manifest = _manifest(run)
    conv = _figure(manifest, "spatial_convergence")
    assert conv["verdict"] == "fail"
    assert manifest["verdict"] == "fail"


def test_unrelated_kinds_are_not_convergence_aliases(tmp_path: Path):
    run = write_fixture_run(tmp_path, "IF-01", dimension=1)
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    parity = json.loads(
        (run / "analysis" / "visual_data" / "backend_parity.json").read_text()
    )
    assert parity["kind"] == "backend_parity"
    assert "backends" in parity
    assert "values" in parity
    assert "series" not in parity or "L1" not in {
        item.get("name") for item in parity.get("series") or []
    }
    manifest = _manifest(run)
    item = _figure(manifest, "backend_parity")
    svg = (run / item["outputs"]["svg"]).read_text(encoding="utf-8")
    assert "KokkosSerial" in svg or "backend" in svg.lower()
    assert "order 2" not in svg


def test_performance_breakdown_uses_named_stages(tmp_path: Path):
    run = write_fixture_run(tmp_path, "PF-06", dimension=1)
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    payload = json.loads(
        (run / "analysis" / "visual_data" / "performance_breakdown.json").read_text()
    )
    assert payload["kind"] == "performance_breakdown"
    assert payload["stages"] == ["ghost_fill", "poisson", "reflux"]
    item = _figure(_manifest(run), "performance_breakdown")
    svg = (run / item["outputs"]["svg"]).read_text(encoding="utf-8")
    assert "ghost_fill" in svg
    assert "poisson" in svg


def test_every_fixture_figure_is_labeled_with_fixture_sha(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    manifest = _manifest(run)
    assert manifest["repository_sha"].startswith(FIXTURE_SHA_PREFIX)
    assert len(manifest["repository_sha"]) != 40
    for item in manifest["figures"]:
        svg = (run / item["outputs"]["svg"]).read_text(encoding="utf-8")
        assert "DETERMINISTIC FIXTURE" in svg
        assert item["repository_sha"].startswith(FIXTURE_SHA_PREFIX)


def test_gallery_fixture_caption_is_not_campaign(tmp_path: Path):
    summary = copy.deepcopy(LOCAL_PR_SUMMARY)
    summary["repository_sha"] = "fixture:pops-visuals-v1"
    write_verification_report(summary, tmp_path)
    outputs = render_release_gallery(tmp_path, formats=("svg", "png", "pdf"))
    for name, path in outputs.items():
        if name.endswith("_status"):
            continue
        text = Path(path).read_text(encoding="utf-8")
        assert "DETERMINISTIC FIXTURE" in text
        assert "campaign dashboard" not in text.lower()
        assert "fixture:" in text


def test_gallery_omits_placeholder_pass_for_missing_topics(tmp_path: Path):
    summary = copy.deepcopy(LOCAL_PR_SUMMARY)
    summary["repository_sha"] = "fixture:pops-visuals-v1"
    summary["orders"] = []
    summary["not_applicable_reason"]["orders"] = "order campaign not run"
    write_verification_report(summary, tmp_path)
    outputs = render_release_gallery(tmp_path, formats=("svg",))
    status = json.loads((tmp_path / "analysis" / "gallery_status.json").read_text())
    assert status["orders_heatmap"] == "not-run"
    svg = Path(outputs["orders_heatmap"]).read_text(encoding="utf-8")
    assert "1.95" not in svg
    assert "DETERMINISTIC FIXTURE" in svg
    assert status["verdict"] != "pass"


def test_3d_slices_are_distinct_or_omitted(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=3)
    visual_dir = run / "analysis" / "visual_data"
    slices = []
    for name in ("slice_xy", "slice_xz", "slice_yz"):
        path = visual_dir / f"{name}.json"
        if path.is_file():
            slices.append(json.loads(path.read_text())["field"])
    if slices:
        assert len(slices) == 3
        assert slices[0] != slices[1]
        assert slices[0] != slices[2]
        assert slices[1] != slices[2]
        render_run(run, suite="release", formats=("svg", "png", "pdf"))
        manifest = _manifest(run)
        for figure_id in ("slice_xy", "slice_xz", "slice_yz"):
            item = _figure(manifest, figure_id)
            assert item["verdict"] in {"pass", "fail"}
    else:
        render_run(run, suite="release", formats=("svg", "png", "pdf"))
        manifest = _manifest(run)
        assert manifest["verdict"] != "pass"
        assert not list((run / "analysis").rglob("slice_xy.svg"))


def test_missing_po01_am01_3d_does_not_pass(tmp_path: Path):
    for case_id in ("PO-01", "AM-01"):
        run = write_fixture_run(tmp_path / case_id, case_id, dimension=3)
        visual_dir = run / "analysis" / "visual_data"
        for path in visual_dir.glob("slice_*.json"):
            path.unlink()
        for name in ("amr_boxes.json", "isosurface.json", "linecut.json"):
            candidate = visual_dir / name
            if candidate.exists():
                candidate.unlink()
        render_run(run, suite="release", formats=("svg", "png", "pdf"))
        manifest = _manifest(run)
        assert manifest["verdict"] != "pass", case_id
        assert manifest["dimensions"]["3d"]["justification"]
        assert "slice_xy" not in manifest["dimensions"]["3d"]["artifacts"]


def test_am01_storyboard_events_match_distinct_frames(tmp_path: Path):
    run = write_fixture_run(tmp_path, "AM-01", dimension=2)
    render_run(run, suite="release", formats=("svg", "png", "pdf"))
    contract = visual_contract_for("AM-01")
    expected = list(contract["animation"]["key_events"])
    payload = json.loads(
        (run / "analysis" / "visual_data" / "storyboard.json").read_text()
    )
    events = [frame["event"] for frame in payload["frames"]]
    assert events == expected
    fields = [frame["series"][0]["y"] for frame in payload["frames"]]
    assert len({tuple(values) for values in fields}) == len(fields)
    manifest = _manifest(run)
    story = _figure(manifest, "storyboard")
    assert story["times"] == [frame["time"] for frame in payload["frames"]]
    assert story["verdict"] == "pass"
    svg = (run / story["outputs"]["svg"]).read_text(encoding="utf-8")
    for event in expected:
        assert event in svg


def test_animation_without_ffmpeg_is_not_pass_and_has_no_movie_claim(tmp_path: Path):
    run = write_fixture_run(tmp_path, "AM-01", dimension=2)
    render_run(run, suite="release", formats=("svg", "png", "pdf", "mp4", "gif"))
    manifest = _manifest(run)
    animation = _figure(manifest, "animation")
    assert "mp4" not in animation["outputs"]
    assert "gif" not in animation["outputs"]
    assert animation["verdict"] in {"not-run", "not-supported"}
    frames = list((run / "analysis" / "animations" / "frames").rglob("frame_*.png"))
    assert len(frames) >= 2
    assert animation["times"] == sorted(animation["times"])
    assert len(animation["times"]) == len(frames)


def test_publication_figures_require_svg_png_pdf_hashes(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    render_run(run, suite="pr", formats=("svg",))
    manifest = _manifest(run)
    for item in manifest["figures"]:
        if item["role"] != "publication":
            continue
        assert set(item["outputs"]) == {"svg", "png", "pdf"}
        assert set(item["output_hashes"]) == {"svg", "png", "pdf"}
        for fmt, rel in item["outputs"].items():
            path = run / rel
            assert path.is_file()
            assert path.stat().st_size > 0


def test_signed_error_profile_uses_signed_values(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    payload = json.loads(
        (run / "analysis" / "visual_data" / "signed_error_profile.json").read_text()
    )
    assert payload["kind"] == "signed_error_profile"
    names = [item["name"] for item in payload["series"]]
    assert names == ["error"]
    values = payload["series"][0]["y"]
    assert min(values) < 0 < max(values)
    render_run(run, suite="pr", formats=("svg", "png", "pdf"))
    item = _figure(_manifest(run), "signed_error_profile")
    svg = (run / item["outputs"]["svg"]).read_text(encoding="utf-8")
    assert "signed" in svg.lower() or "error" in svg.lower()


def test_signed_error_field_is_not_a_scaled_snapshot(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=2)
    exact = json.loads((run / "analysis" / "visual_data" / "exact_field.json").read_text())
    numerical = json.loads(
        (run / "analysis" / "visual_data" / "numerical_field.json").read_text()
    )
    signed = json.loads(
        (run / "analysis" / "visual_data" / "signed_error_field.json").read_text()
    )
    expected = [
        [n - e for n, e in zip(nrow, erow, strict=True)]
        for nrow, erow in zip(numerical["field"], exact["field"], strict=True)
    ]
    assert signed["field"] == expected
    assert min(value for row in expected for value in row) < 0
    assert max(value for row in expected for value in row) > 0
    render_run(run, suite="release", formats=("svg", "png", "pdf"))
    item = _figure(_manifest(run), "signed_error_field")
    assert item["transform"] == "diverging"
