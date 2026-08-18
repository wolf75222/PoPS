"""IF-04 checkpoint/restart identity (in-memory JSON; optional native restart)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "checkpoint_restart"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
TR01_RUN = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "run.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RESTORE_MISSING = (
    "restore_checkpoint_payload requires owner and executor; "
    "no public path restore returns scientific {centers, q, t}"
)


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
    centers = exact.cell_centers(exact.N_CELLS)
    np.testing.assert_array_equal(
        exact.manufactured_q(centers, exact.T),
        tr01.exact_sine(centers, exact.T),
    )


def test_round_trip_linf_is_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    state = exact.manufactured_state()
    restored = run.round_trip(state)
    volumes = exact.cell_volumes()
    for key in ("centers", "q"):
        errors = reference_errors(restored[key], state[key], volumes)
        assert errors.l1 == 0.0
        assert errors.l2 == 0.0
        assert errors.linf == 0.0


def test_round_trip_preserves_t():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    state = exact.manufactured_state()
    restored = run.round_trip(state)
    assert restored["t"] == state["t"]
    assert restored["t"] == exact.T


def test_write_if04_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "in-memory checkpoint identity" in loaded["not_applicable_reason"]["orders"]


def test_checkpoint_consumer_installs_on_tr01_case():
    run = _load_case_module("run")
    tr01 = load_sibling_module(TR01_RUN)
    case = run.install_checkpoint_consumer(tr01.build_case(8))
    assert case.options()["has_consumers"] is True
    assert run.refuse_native_restore() == RESTORE_MISSING


@pytest.mark.compiler
def test_run_native_restart_linf_or_skips():
    run = _load_case_module("run")
    try:
        result = run.run_native(n_cells=16, t=0.1)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert result["continuous"].shape == result["restarted"].shape
    assert np.isfinite(result["continuous"]).all()
    assert np.isfinite(result["restarted"]).all()
    assert float(result["linf"]) < 1.0e-3


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
        assert "h5py" not in text
        assert "hdf5" not in text.lower()
