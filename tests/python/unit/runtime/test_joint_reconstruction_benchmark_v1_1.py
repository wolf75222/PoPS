"""Regression for the installed artifact's layout-program property shape."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[4]
DOCS = ROOT / "docs" / "development" / "api_040"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiled_artifact_exposes_layout_paths_as_property() -> None:
    source = ast.parse((ROOT / "python" / "pops" / "codegen"
                        / "_compiled_artifact.py").read_text(encoding="utf-8"))
    classes = (node for node in source.body if isinstance(node, ast.ClassDef))
    method = next(method for cls in classes for method in cls.body
                  if isinstance(method, ast.FunctionDef)
                  and method.name == "layout_program_paths")
    assert any(isinstance(decorator, ast.Name) and decorator.id == "property"
               for decorator in method.decorator_list)


def test_v1_1_reads_property_and_preserves_v1_failure_evidence(tmp_path: Path) -> None:
    old = _load(DOCS / "joint_reconstruction_benchmark.py")
    fixed = _load(DOCS / "joint_reconstruction_benchmark_v1_1.py")
    component = tmp_path / "block.so"
    program = tmp_path / "program.so"
    simulation = tmp_path / "simulation.so"
    for path in (component, program, simulation):
        path.write_bytes(path.name.encode())

    class InstalledArtifactShape:
        blocks = (SimpleNamespace(model=SimpleNamespace(so_path=component)),)
        so_path = simulation

        @property
        def layout_program_paths(self):
            return {"uniform": str(program)}

    artifact = InstalledArtifactShape()
    with pytest.raises(TypeError, match="not callable"):
        old._dso_sizes(artifact)
    rows = fixed._dso_sizes(artifact)
    assert {Path(path) for path in rows} == {component, program, simulation}
    assert all(row["sha256"] == fixed._sha(Path(path)) for path, row in rows.items())
    assert fixed.SCHEMA.endswith("v1.1")
    assert (fixed.N, fixed.WIDTH, fixed.STEPS, fixed.DT, fixed.ORDER,
            fixed.WARMUPS, fixed.SAMPLES, fixed.RTOL, fixed.ATOL,
            fixed.MASS_ATOL) == (old.N, old.WIDTH, old.STEPS, old.DT,
                                 old.ORDER, old.WARMUPS, old.SAMPLES,
                                 old.RTOL, old.ATOL, old.MASS_ATOL)


def test_resource_companion_pins_corrected_protocol() -> None:
    probe = _load(DOCS / "joint_reconstruction_resource_probe_v1_1.py")
    benchmark = DOCS / "joint_reconstruction_benchmark_v1_1.py"
    assert probe.V1_1_SHA256 == hashlib.sha256(benchmark.read_bytes()).hexdigest()
    assert probe._frozen_v1_1().SCHEMA.endswith("v1.1")
    assert probe.SCHEMA.endswith("v1.1")
