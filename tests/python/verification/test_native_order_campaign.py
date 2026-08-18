"""Native-order campaign helper (in-memory L2 ∝ h²; no live compile)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "native_order"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RESOLUTIONS = (16, 32, 64, 128)
ORDER_THRESHOLD = 1.8
SPACES = ("Serial", "OpenMP")
DIMENSIONS = ("Dim1", "Dim2")


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _job_label(job) -> str:
    if isinstance(job, str):
        return job
    label = getattr(job, "label", None)
    if label is not None:
        return str(label)
    if isinstance(job, dict) and "label" in job:
        return str(job["label"])
    return str(job)


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


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def test_plan_jobs_emits_serial_openmp_dim1_dim2_labels():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    jobs = run.plan_jobs()
    labels = [_job_label(job) for job in jobs]
    assert len(labels) == 4
    assert len(set(labels)) == 4
    for space in SPACES:
        for dim in DIMENSIONS:
            assert any(space in label and dim in label for label in labels)


def test_manufactured_l2_h2_series_observed_order_at_least_1_8():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "pops" not in _imported_module_names(text)
    assert "from exact import" not in text
    assert tuple(exact.RESOLUTIONS) == RESOLUTIONS
    n = np.asarray(exact.RESOLUTIONS, dtype=np.float64)
    spacings = 1.0 / n
    errors = exact.manufactured_l2(exact.RESOLUTIONS)
    np.testing.assert_allclose(errors, spacings**2)
    orders = observed_order(errors, spacings)
    assert orders.size == 3
    assert np.all(orders >= ORDER_THRESHOLD)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))


def test_write_native_order_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_native_order_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    observed = [row["observed_order"] for row in loaded["orders"]]
    assert observed
    assert all(value >= ORDER_THRESHOLD for value in observed)
    assert all(row["threshold"] == ORDER_THRESHOLD for row in loaded["orders"])
    assert set(loaded["execution_spaces"]) == {"KokkosSerial", "KokkosOpenMP"}
    assert set(loaded["native_dimensions"]) == {1, 2}


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
        assert "pops.compile" not in text
