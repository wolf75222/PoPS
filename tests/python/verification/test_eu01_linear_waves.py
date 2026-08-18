"""EU-01 Euler linear eigenmodes (in-memory oracle; no solver required)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.provenance import RUN_FIELDS
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "linear_waves"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"eu01_linear_waves_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width, width


def test_entropy_mode_at_t1_equals_t0_when_u_zero():
    exact = _load_case_module("exact")
    background = exact.background()
    assert background["u"] == 0.0
    x, _ = _cell_centers()
    t0 = exact.exact_mode(x, 0.0, mode="entropy")
    t1 = exact.exact_mode(x, 1.0, mode="entropy")
    np.testing.assert_allclose(t1, t0, rtol=0.0, atol=0.0)


def test_right_acoustic_travels_at_plus_c():
    exact = _load_case_module("exact")
    x, dx = _cell_centers()
    speed = exact.acoustic_speed(exact.background())
    evolved = exact.exact_mode(x, dx / speed, mode="right")
    initial = exact.exact_mode(x, 0.0, mode="right")
    np.testing.assert_allclose(evolved, np.roll(initial, 1, axis=1), rtol=0.0, atol=1.0e-14)


def test_eigenvectors_are_linearly_independent():
    exact = _load_case_module("exact")
    vectors = exact.right_eigenvectors(exact.background())
    matrix = np.column_stack((vectors["left"], vectors["entropy"], vectors["right"]))
    assert matrix.shape == (3, 3)
    assert abs(float(np.linalg.det(matrix))) > 1.0e-12


def test_write_eu01_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_eu01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = run.run_native(16, t_end=0.05, mode="entropy")
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    array = np.asarray(field, dtype=np.float64)
    assert array.shape == (3, 16)
    assert np.isfinite(array).all()
    assert np.all(array[0] > 0.0)


def test_eu01_request_returns_run_fields():
    """A campaign request must return provenance fields, not a raw array."""
    import inspect

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="EU-01", pops_native_dim=1, min_resolution=16)
    )
    fields = run.campaign_run_fields(16, 0.05, request)
    missing = [key for key in RUN_FIELDS if key not in fields]
    assert missing == []
    assert fields["mpi_enabled"] is False
    assert fields["mpi_ranks"] == 1
    assert fields["resolution"] == [16]
    assert fields["cfl"] == run.CFL
    assert fields["final_time"] == 0.05
    assert fields["time_program"] == "SSPRK2"
