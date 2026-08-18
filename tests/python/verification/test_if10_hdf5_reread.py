"""IF-10 HDF5-shaped round-trip (in-memory npz stand-in; native reread refused)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "hdf5_reread"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
TR01_RUN = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "run.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
ORDERS_REASON = "npz stand-in / live HDF5 on ROMEO"
COMPONENTS = ["q"]
OWNER = "rank0"
REREAD_MISSING = (
    "TR-01 Case.blocks() exposes no public state Handle for ScientificOutput.fields; "
    "read_hdf5 is not a {centers, q, components, owner} reread"
)


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _imports_module(source: str, root: str) -> bool:
    tree = ast.parse(source)
    prefix = root + "."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == root or alias.name.startswith(prefix) for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == root or node.module.startswith(prefix):
                return True
    return False


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
        exact.manufactured_q(centers),
        tr01.exact_sine_1d(centers, 0.0),
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


def test_component_order_preserved():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    state = exact.manufactured_state()
    assert list(state["components"]) == COMPONENTS
    assert state["owner"] == OWNER
    restored = run.round_trip(state)
    assert list(restored["components"]) == COMPONENTS
    assert restored["owner"] == OWNER
    multi = dict(state)
    multi["components"] = ["rho", "u", "E"]
    restored_multi = run.round_trip(multi)
    assert list(restored_multi["components"]) == ["rho", "u", "E"]


def test_write_if10_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if10_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_native_npz_reread_or_skips():
    run = _load_case_module("run")
    tr01 = load_sibling_module(TR01_RUN)
    case = tr01.build_case(8)
    assert run.public_state_handles(case) == ()
    assert run.refuse_native_reread() == REREAD_MISSING
    try:
        result = run.run_native(8, t_end=0.05)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert Path(result["path"]).is_file()


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert not _imports_module(exact_text, "pops")
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
        assert not _imports_module(text, "h5py")
