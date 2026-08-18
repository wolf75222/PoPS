"""Storyboard and 2-d field rendering (plan §40.4.2, §40.4.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from verification.pops_verify.visualization.fixtures import write_fixture_run
from verification.pops_verify.visualization.render import render_run

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def test_tr01_2d_fixture_writes_field_and_linecut_figures(tmp_path: Path):
    run = write_fixture_run(tmp_path, "TR-01", dimension=2)
    render_run(run, suite="release", formats=("svg", "png"))
    signed = run / "analysis" / "figures" / "diagnostic" / "signed_error_field.svg"
    assert signed.is_file()
    text = signed.read_text(encoding="utf-8")
    assert "error" in text.lower() or "signed" in text.lower() or "x / L" in text
    assert "DETERMINISTIC FIXTURE" in text


def test_am01_storyboard_uses_accepted_events(tmp_path: Path):
    run = write_fixture_run(tmp_path, "AM-01", dimension=2)
    render_run(run, suite="release", formats=("svg",))
    storyboard = run / "analysis" / "storyboards" / "storyboard.svg"
    assert storyboard.is_file()
    text = storyboard.read_text(encoding="utf-8")
    assert "before_entry" in text
    assert "periodic_crossing" in text
    assert "fixture:" in text
    assert "DETERMINISTIC FIXTURE" in text
