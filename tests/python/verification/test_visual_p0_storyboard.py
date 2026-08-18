"""P0 storyboard events must not depend on an MP4 animation contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verification.pops_verify.visualization.catalog import (
    P0_STATIC_CASES,
    catalog_entry,
    visual_contract_for,
)
from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.fixtures import write_fixture_run

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "render_verification_visuals.py"
DEFAULT_EVENTS = ["initial", "mid", "final"]

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _storyboard_events(case_id: str) -> list[str]:
    contract = visual_contract_for(case_id)
    storyboard = contract.get("storyboard") or {}
    return list(storyboard.get("key_events") or [])


def test_cp01_and_am09_storyboard_events_are_independent_of_mp4():
    for case_id in ("CP-01", "AM-09"):
        entry = catalog_entry(case_id)
        assert entry.storyboard_required
        assert "required" not in entry.animation
        contract = visual_contract_for(case_id)
        assert contract["animation"] is None, case_id
        events = _storyboard_events(case_id)
        assert events == DEFAULT_EVENTS, case_id
        assert len(events) == 3


def test_fixture_storyboard_for_cp01_and_am09(tmp_path: Path):
    for case_id in ("CP-01", "AM-09"):
        run = write_fixture_run(tmp_path / case_id, case_id, dimension=1)
        payload = json.loads(
            (run / "analysis" / "visual_data" / "storyboard.json").read_text()
        )
        events = [frame["event"] for frame in payload["frames"]]
        assert events == DEFAULT_EVENTS, case_id
        assert len(events) == 3


def test_missing_storyboard_events_raise_visuals_error(monkeypatch, tmp_path: Path):
    original = visual_contract_for

    def stripped(case_id: str) -> dict:
        contract = original(case_id)
        contract["animation"] = None
        contract["storyboard"] = None
        return contract

    monkeypatch.setattr(
        "verification.pops_verify.visualization.fixtures.visual_contract_for",
        stripped,
    )
    with pytest.raises(VisualsError, match="key_events"):
        write_fixture_run(tmp_path, "CP-01", dimension=1)


def test_cli_examples_covers_every_p0_case(tmp_path: Path):
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--examples",
            str(tmp_path),
            "--formats",
            "svg",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ValueError" not in completed.stderr
    assert "Traceback" not in completed.stderr
    for case_id in P0_STATIC_CASES:
        manifests = list((tmp_path / case_id).rglob("visual_manifest.json"))
        assert manifests, case_id


def test_cli_examples_converts_uncaught_errors_to_exit_2(tmp_path: Path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_verification_visuals_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cli)

    def boom(*_args, **_kwargs):
        raise ValueError("storyboard requires visual_contract_for key_events")

    monkeypatch.setattr(cli, "write_fixture_run", boom)
    assert cli.main(["--examples", str(tmp_path)]) == 2
