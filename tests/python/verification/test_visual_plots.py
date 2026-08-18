"""Plot transforms and headless rendering (plan §40.4)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from verification.pops_verify.visualization.plots import (
    VisualsError,
    file_sha256,
    prepare_convergence,
    prepare_field,
    prepare_profile,
    render_prepared,
)

import matplotlib
matplotlib.use("Agg", force=True)


CONVERGENCE = {
    "figure_id": "spatial_convergence",
    "kind": "spatial_convergence",
    "data_kind": "deterministic_fixture",
    "verdict": "pass",
    "units": {"x": "1/h", "y": "L2 error"},
    "variables": ["scalar"],
    "series": [
        {
            "name": "L1",
            "x": [16, 32, 64, 128],
            "y": [3.2e-2, 8.0e-3, 2.0e-3, 5.0e-4],
            "unit": "1",
        },
        {
            "name": "L2",
            "x": [16, 32, 64, 128],
            "y": [1.6e-2, 4.0e-3, 1.0e-3, 2.5e-4],
            "unit": "1",
        },
        {
            "name": "Linf",
            "x": [16, 32, 64, 128],
            "y": [4.8e-2, 1.2e-2, 3.0e-3, 7.5e-4],
            "unit": "1",
        },
    ],
    "reference_slopes": [{"order": 2, "anchor": [16, 1.6e-2]}],
}

PROFILE = {
    "figure_id": "reference_profile",
    "kind": "reference_profile",
    "data_kind": "deterministic_fixture",
    "verdict": "pass",
    "units": {"x": "x / L", "y": "scalar"},
    "variables": ["scalar"],
    "series": [
        {
            "name": "exact",
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "y": [0.0, 1.0, 0.0, -1.0, 0.0],
            "unit": "1",
        },
        {
            "name": "numerical",
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "y": [0.0, 0.99, 0.01, -0.98, 0.0],
            "unit": "1",
        },
        {
            "name": "error",
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "y": [0.0, -0.01, 0.01, 0.02, 0.0],
            "unit": "1",
        },
    ],
}

FIELD = {
    "figure_id": "signed_error_field",
    "kind": "signed_error_field",
    "data_kind": "deterministic_fixture",
    "verdict": "pass",
    "units": {"x": "x / L", "y": "y / L", "field": "error"},
    "variables": ["error"],
    "x": [0.0, 0.5, 1.0],
    "y": [0.0, 0.5, 1.0],
    "field": [[-0.2, 0.0, 0.1], [0.0, 0.05, -0.1], [0.15, -0.05, 0.0]],
}


def test_prepare_convergence_keeps_raw_points_and_reference_slope():
    prepared = prepare_convergence(CONVERGENCE)
    assert prepared["scale"] == "loglog"
    assert prepared["series"][1]["y"] == [1.6e-2, 4.0e-3, 1.0e-3, 2.5e-4]
    assert prepared["reference_slopes"][0]["order"] == 2
    assert prepared["xlabel"] == "1/h"
    assert prepared["ylabel"] == "L2 error"


def test_prepare_convergence_refuses_empty_series():
    payload = dict(CONVERGENCE)
    payload["series"] = [{"name": "L2", "x": [], "y": [], "unit": "1"}]
    with pytest.raises(VisualsError, match="empty"):
        prepare_convergence(payload)


def test_prepare_field_uses_diverging_scale_centered_on_zero():
    prepared = prepare_field(FIELD)
    assert prepared["cmap"] == "RdBu_r"
    assert prepared["vmin"] == -prepared["vmax"]
    assert prepared["center"] == 0.0


def test_render_convergence_writes_labeled_svg_and_png(tmp_path: Path):
    prepared = prepare_convergence(CONVERGENCE)
    outputs = render_prepared(
        prepared,
        tmp_path,
        formats=("svg", "png", "pdf"),
        caption="DETERMINISTIC FIXTURE — not a PoPS campaign result",
        provenance_sha="0123456789abcdef0123456789abcdef01234567",
    )
    svg = Path(outputs["svg"]).read_text(encoding="utf-8")
    assert "1/h" in svg
    assert "L2 error" in svg
    assert "0123456789abcdef0123456789abcdef01234567" in svg
    assert "DETERMINISTIC FIXTURE" in svg
    png = Path(outputs["png"])
    assert png.stat().st_size > 1000
    image = matplotlib.image.imread(png)
    assert image.shape[0] >= 200
    assert image.shape[1] >= 200
    assert Path(outputs["pdf"]).read_bytes()[:4] == b"%PDF"


def test_identical_data_produces_identical_hashes(tmp_path: Path):
    prepared = prepare_convergence(CONVERGENCE)
    first = render_prepared(
        prepared,
        tmp_path / "a",
        formats=("svg",),
        caption="fixture",
        provenance_sha="abc",
    )
    second = render_prepared(
        prepared,
        tmp_path / "b",
        formats=("svg",),
        caption="fixture",
        provenance_sha="abc",
    )
    assert file_sha256(first["svg"]) == file_sha256(second["svg"])
    assert hashlib.sha256(Path(first["svg"]).read_bytes()).hexdigest()


def test_prepare_profile_keeps_exact_and_numerical_separate():
    prepared = prepare_profile(PROFILE)
    names = [item["name"] for item in prepared["series"]]
    assert names == ["exact", "numerical", "error"]
    assert prepared["series"][0]["style"] == "exact"
