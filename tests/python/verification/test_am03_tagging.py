"""AM-03 gradient / second-diff tagging (public 1-d AMR plus in-memory helpers)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "tagging"
TR02_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")


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


def _periodic_halo(mask, width: int) -> np.ndarray:
    """Cells at periodic distance 1..width from a tagged cell, excluding the mask."""
    selected = np.asarray(mask, dtype=bool)
    added = np.zeros(selected.shape, dtype=bool)
    for shift in range(1, int(width) + 1):
        added |= np.roll(selected, shift)
        added |= np.roll(selected, -shift)
    return added & ~selected


def test_tagged_set_contains_pulse_core_for_documented_theta():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    tr02 = _load_tr02("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert exact.THETA > 0.0
    assert exact.THETA2 > 0.0
    tagged = run.raw_tag_mask()
    core = run.pulse_core_mask()
    assert np.any(core)
    assert np.all(tagged[core])
    centers, field = run.sample_field()
    np.testing.assert_allclose(field, tr02.exact_gaussian(centers, 0.0))


def test_buffer_of_two_adds_exactly_two_cells_each_side_periodic():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert tuple(exact.BUFFER_WIDTHS) == (1, 2, 4)
    raw = run.raw_tag_mask()
    buffered = run.buffered_tag_mask(2)
    added = buffered & ~raw
    expected = _periodic_halo(raw, 2)
    np.testing.assert_array_equal(added, expected)
    np.testing.assert_array_equal(buffered, raw | expected)
    farther = _periodic_halo(raw, 3) & ~expected
    assert not np.any(buffered & farther)
    assert int(np.count_nonzero(added)) == 4


def test_hysteresis_does_not_oscillate_on_static_field():
    run = _load_case_module("run")
    raw = run.raw_tag_mask()
    empty = np.zeros_like(raw)
    first = run.hysteresis_update(empty)
    second = run.hysteresis_update(first)
    third = run.hysteresis_update(second)
    np.testing.assert_array_equal(first, raw)
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(third, second)
    full = np.ones_like(raw)
    from_full = run.hysteresis_update(full)
    from_full_again = run.hysteresis_update(from_full)
    np.testing.assert_array_equal(from_full_again, from_full)
    from_raw = run.hysteresis_update(raw)
    np.testing.assert_array_equal(from_raw, raw)
    np.testing.assert_array_equal(run.hysteresis_update(from_raw), from_raw)


def test_write_am03_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["amr"]["invariants_ok"] is True


def test_siblings_reuse_tr02_via_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "gaussian_pulse" in text


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
    case = run.build_case(8)
    plan = run.resolve_plan(n_cells=8)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(run.run_native(8, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert field.size > 0
    assert field.shape == (8,)
    assert np.isfinite(field).all()
