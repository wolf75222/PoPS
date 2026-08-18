"""EU-06 exact uniform-flow preservation (in-memory oracle; no solver)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "uniform_flow"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
BUMP_AMPLITUDE = 0.25


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


def test_exact_field_is_invariant_in_t():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    background = exact.background()
    assert background["rho"] == 1.0
    assert background["u"] == 1.0
    assert background["v"] == 0.0
    assert background["p"] == 1.0
    assert exact.GAMMA == 1.4
    x, y = exact.cell_centers(N_CELLS)
    initial = exact.exact_primitives(x, y, 0.0)
    later = exact.exact_primitives(x, y, 1.0)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_array_equal(later[key], initial[key])
        assert exact.is_spatially_constant(initial[key])
        assert exact.is_spatially_constant(later[key])
        np.testing.assert_array_equal(initial[key], background[key])


def test_linf_leftover_of_one_cell_bump_is_the_bump_amplitude():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    leftover = run.one_cell_bump_leftover_linf(
        n_cells=N_CELLS, amplitude=BUMP_AMPLITUDE
    )
    assert leftover == BUMP_AMPLITUDE
    for kind in ("block_face", "cf"):
        bumped = run.manufactured_interface_bump(
            n_cells=N_CELLS, amplitude=BUMP_AMPLITUDE, kind=kind
        )
        oracle = exact.exact_primitives(*exact.cell_centers(N_CELLS), 0.0)["rho"]
        volumes = exact.cell_volumes(N_CELLS)
        assert run.leftover_linf(bumped, oracle, volumes) == BUMP_AMPLITUDE


def test_write_eu06_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_eu06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "machine-zero free-stream" in loaded["not_applicable_reason"]["orders"]


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
        assert "from exact import" not in text
