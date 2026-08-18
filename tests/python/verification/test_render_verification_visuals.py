"""CLI for Phase 8 rendering (plan §40.3)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "render_verification_visuals.py"

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def test_cli_renders_fixture_run(tmp_path: Path):
    from verification.pops_verify.visualization.fixtures import write_fixture_run

    run = write_fixture_run(tmp_path, "TR-01", dimension=1)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            str(run),
            "--formats",
            "svg,png",
            "--suite",
            "pr",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((run / "analysis" / "visual_manifest.json").read_text())
    assert manifest["schema"] == "pops.verification.visual_manifest.v1"


def test_check_visuals_catalog_cli():
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_verification_visuals.py")],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "89 visual contracts" in completed.stdout


def test_cli_examples_writes_tr01_and_po01(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--examples",
            str(tmp_path),
            "--formats",
            "svg,png,pdf",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "TR-01" / "fixture-1d" / "analysis" / "visual_manifest.json").is_file()
    assert (tmp_path / "PO-01" / "fixture-1d" / "analysis" / "visual_manifest.json").is_file()
