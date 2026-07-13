"""Executable acceptance for the normative explicit-IMEX + AMR target."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"


def test_example_runs_and_every_scientific_format_reopens(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["POPS_INCLUDE"] = str(ROOT / "include")
    environment["POPS_KOKKOS_ROOT"] = sys.prefix
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--output-dir", str(tmp_path / "published")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "HDF5:" in completed.stdout
    assert "ParaView:" in completed.stdout
    assert "checkpoint:" in completed.stdout
    assert "bit-identical restart: True" in completed.stdout
    report_line, = [line for line in completed.stdout.splitlines() if line.startswith("report: ")]
    report = json.loads(report_line.removeprefix("report: "))
    assert report["finite"] is True
    assert report["checkpoint_restart_bit_identical"] is True
    assert report["levels"] == 2
    assert report["runtime_steps"] == 1

    from pops.output import read_hdf5, read_npz, read_paraview

    output = tmp_path / "published"
    readers = {".h5": read_hdf5, ".npz": read_npz, ".vtu": read_paraview}
    for suffix, reader in readers.items():
        # Scientific writers use the stable ``consumer__clock__step`` stem. Checkpoints are also
        # NPZ containers but intentionally have a different schema and must not be opened as output.
        paths = tuple(output.rglob("*__*%s" % suffix))
        assert paths, "the accepted step did not publish %s" % suffix
        reopened = reader(paths[-1])
        assert reopened.arrays
    assert tuple(output.rglob("manual_restart*.npz"))


def test_normative_example_uses_only_the_final_root_lifecycle() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "pops.validate(" in source
    assert "pops.resolve(" in source
    assert "pops.compile(" in source
    assert "pops.bind(" in source
    assert "pops.run(simulation," in source
    assert ".run(**" not in source
    assert "BindInputs" not in source
    assert source.count("case.program(") == 1
    assert source.count("case.consumers(") == 1

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    root_resolve = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pops"
        and node.func.attr == "resolve"
    ]
    assert len(root_resolve) == 1
    assert all(keyword.arg != "strict" for keyword in root_resolve[0].keywords)
    root_run = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pops"
        and node.func.attr == "run"
    ]
    assert len(root_run) == 1
