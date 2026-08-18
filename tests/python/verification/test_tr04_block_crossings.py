"""TR-04 face/edge/corner crossing (in-memory two-block exact translation)."""
from __future__ import annotations

import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "block_crossings"
TR02_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
PLACEMENTS = ("face", "edge", "corner")
FACE = 0.5
N_BLOCKS = 2
BLOCK_EDGES = (0.0, 0.5, 1.0)
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_call_owners(source: str) -> list[str]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    owners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pops"
        ):
            continue
        current: ast.AST | None = node
        owner = "<module>"
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = current.name
                break
            current = parents.get(current)
        owners.append(owner)
    return owners


def test_exact_loads_tr02_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "gaussian_pulse" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    tr02 = load_sibling_module(TR02_EXACT)
    width = 1.0 / N_CELLS
    centers = (np.arange(N_CELLS, dtype=np.float64) + 0.5) * width
    np.testing.assert_array_equal(
        exact.exact_gaussian(centers, 0.25),
        tr02.exact_gaussian(centers, 0.25),
    )


def test_three_1d_placements_use_face_at_half_and_two_block_join():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert tuple(exact.PLACEMENTS) == PLACEMENTS
    assert exact.FACE == FACE
    assert exact.N_BLOCKS == N_BLOCKS
    assert tuple(exact.BLOCK_EDGES) == BLOCK_EDGES
    assert tuple(run.two_block_join()) == BLOCK_EDGES
    face_time = (FACE - exact.X0) / exact.A
    assert exact.placement_time("face") == face_time
    assert exact.placement_time("edge") == face_time + exact.PERIOD
    assert exact.placement_time("corner") == face_time + 2.0 * exact.PERIOD
    centers, volumes = exact.two_block_cell_centers(N_CELLS)
    assert centers.size == N_CELLS
    assert volumes.size == N_CELLS
    assert exact.N_BLOCKS == len(BLOCK_EDGES) - 1
    left = centers[centers < FACE]
    right = centers[centers > FACE]
    assert left.size == N_CELLS // 2
    assert right.size == N_CELLS // 2
    assert left.max() < FACE < right.min()


def test_placements_agree_field_to_field_linf_is_zero_for_exact_translation():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    sampled = {}
    volumes = None
    for name in PLACEMENTS:
        _centers, field, cell_volumes = run.sample_placement(name, N_CELLS)
        sampled[name] = field
        volumes = cell_volumes
        peak = float(_centers[int(np.argmax(field))])
        assert abs(peak - FACE) <= (1.0 / N_CELLS)
    for left, right in combinations(sampled.values(), 2):
        assert analyze.field_to_field_linf(left, right) == 0.0
        assert reference_errors(left, right, volumes).linf == 0.0
    assert analyze.placements_agree(N_CELLS) == 0.0


def test_write_tr04_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tr04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_siblings_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "amr_crossing_layout" in text
    assert "uniform_periodic_layout" not in text
    assert "pops.amr" in text or "AMRHierarchy" in text
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    from tests.python.support.requirements import missing_compiler_requirement

    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (16,)
    assert np.isfinite(field).all()
