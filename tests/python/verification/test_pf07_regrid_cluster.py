"""PF-07 regrid / clustering stand-in (1-d pulse tags; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "regrid_cluster"
TR02_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
MIN_PATCH_WIDTH = 4
ORDERS_REASON = "kernel microbench stand-in, not a timed PF run"


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _load_tr02(name: str):
    return load_sibling_module(TR02_DIR / f"{name}.py")


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


def test_clustered_patch_count_le_raw_tag_count():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    tr02 = _load_tr02("exact")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert exact.MIN_PATCH_WIDTH == MIN_PATCH_WIDTH
    result = run.cluster_tagged_pulse()
    raw = int(result["raw_tag_count"])
    patches = int(result["patch_count"])
    assert raw > 0
    assert patches <= raw
    assert patches >= 1
    field = np.asarray(result["field"], dtype=np.float64)
    centers = np.asarray(result["centers"], dtype=np.float64)
    np.testing.assert_allclose(field, tr02.exact_gaussian(centers, 0.0))


def test_all_tagged_cells_covered():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    result = run.cluster_tagged_pulse()
    tags = np.asarray(result["tags"], dtype=bool)
    assert np.any(tags)
    covered = exact.coverage_mask(result["patches"], tags.size)
    np.testing.assert_array_equal(covered, np.asarray(result["covered"], dtype=bool))
    assert np.all(covered[tags])
    for _start, width in result["patches"]:
        assert int(width) >= exact.MIN_PATCH_WIDTH


def test_write_pf07_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_siblings_reuse_tr02_via_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "gaussian_pulse" in text


def test_run_native_points_at_official_benchmarks():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "benchmarks/manifest.toml" in text
    with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
        run.run_native()



def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
