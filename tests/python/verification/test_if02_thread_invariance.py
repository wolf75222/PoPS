"""IF-02 OpenMP thread-count invariance (in-memory exact fields; no live OpenMP)."""
from __future__ import annotations

import ast
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "thread_invariance"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
THREAD_COUNTS = (1, 2, 4, 8)
N_CELLS = 32
T = 0.25
ORDERS_REASON = "exact-field identity / no live OpenMP"


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


def test_exact_loads_tr01_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    tr01 = load_sibling_module(TR01_EXACT)
    centers = exact.cell_centers(N_CELLS)
    np.testing.assert_array_equal(
        exact.exact_sine(centers, T),
        tr01.exact_sine_1d(centers, T),
    )


def test_thread_counts_are_one_two_four_eight():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert tuple(exact.THREAD_COUNTS) == THREAD_COUNTS
    assert tuple(run.thread_counts()) == THREAD_COUNTS
    for n_threads in THREAD_COUNTS:
        slices = exact.thread_slices(N_CELLS, n_threads)
        assert len(slices) == n_threads
        assert slices[0][0] == 0
        assert slices[-1][1] == N_CELLS
        assert all(stop - start == N_CELLS // n_threads for start, stop in slices)
        covered = [index for start, stop in slices for index in range(start, stop)]
        assert covered == list(range(N_CELLS))


def test_all_thread_labels_agree_exactly():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    volumes = exact.cell_volumes(N_CELLS)
    fields = run.exact_fields_for_thread_counts(N_CELLS, t=T)
    assert tuple(fields) == THREAD_COUNTS
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        assert errors.l1 == 0.0
        assert errors.l2 == 0.0
        assert errors.linf == 0.0
        np.testing.assert_array_equal(left, right)
    assert analyze.threads_agree(N_CELLS, t=T) == 0.0
    assert run.max_thread_difference(N_CELLS, t=T) == 0.0


def test_write_if02_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_readme_documents_romeo_only_live_kokkos_sweep():
    text = (CASE_DIR / "README.md").read_text(encoding="utf-8")
    assert "ROMEO" in text
    assert "Kokkos" in text
    lowered = text.lower()
    assert "thread" in lowered
    assert "openmp" in lowered


def test_run_native_threads_sets_omp_and_compares_tr01_fields(monkeypatch):
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "OMP_NUM_THREADS" in text
    assert "from exact import" not in text
    assert "pops.run(" not in text
    assert run._TR01_RUN == TR01_EXACT.with_name("run.py")

    seen: list[tuple[str | None, int, float]] = []

    class _FakeTr01:
        NativeUnavailable = RuntimeError

        def run_native(self, n_cells, t_end=1.0):
            seen.append((os.environ.get("OMP_NUM_THREADS"), int(n_cells), float(t_end)))
            field = np.zeros(int(n_cells), dtype=np.float64)
            field[0] = float(os.environ["OMP_NUM_THREADS"])
            return field

    monkeypatch.setattr(run, "_tr01_run", lambda: _FakeTr01())
    previous = os.environ.get("OMP_NUM_THREADS")
    try:
        pairwise = run.run_native_threads((1, 2, 4, 8), n_cells=N_CELLS, t_end=T)
    finally:
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous
    assert seen == [
        ("1", N_CELLS, T),
        ("2", N_CELLS, T),
        ("4", N_CELLS, T),
        ("8", N_CELLS, T),
    ]
    assert pairwise[(1, 2)] == 1.0
    assert pairwise[(1, 8)] == 7.0
    assert pairwise[(4, 8)] == 4.0
    assert os.environ.get("OMP_NUM_THREADS") == previous


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
