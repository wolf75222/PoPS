"""Executable acceptance for the final ADC-694 HyQMOM15 lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"


def test_hyqmom15_example_runs_outputs_and_restarts_bit_identically(tmp_path) -> None:
    output = tmp_path / "complete"
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--cells", "8", "--output-dir", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "HDF5:" in completed.stdout
    assert "ParaView:" in completed.stdout
    assert "checkpoint:" in completed.stdout
    assert "bit-identical restart: True" in completed.stdout
    report_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("report: "))
    report = json.loads(report_line.removeprefix("report: "))
    assert report["finite"] is True
    assert report["n_moments"] == 15
    assert report["runtime_steps"] == 2

    from pops.output import read_hdf5, read_paraview

    hdf5_path = output / "accepted" / "hyqmom15.h5"
    paraview_path = output / "accepted" / "hyqmom15.vtu"
    assert read_hdf5(hdf5_path).arrays
    assert read_paraview(paraview_path).arrays
    assert (output / "manual_restart.npz").is_file()
    scheduled = tuple((output / "accepted").rglob("*.npz"))
    assert len(scheduled) == 1
    with np.load(output / "manual_restart.npz", allow_pickle=False) as stored:
        assert [str(name) for name in stored["blocks"]] == ["plasma"]
        assert int(stored["macro_step"]) == 1
        assert "runtime_consumer_graph" in stored
        assert "field_provider_slots" in stored
