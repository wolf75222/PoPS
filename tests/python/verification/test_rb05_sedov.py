"""RB-05 Sedov off-center (2-d self-similar radius; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS
from verification.pops_verify.symmetry import radial_anisotropy

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "sedov"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
TIME_EXPONENT_2D = 2.0 / 5.0


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


def test_shock_radius_scales_as_t_to_the_two_fifths_in_2d():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    np.testing.assert_allclose(exact.self_similar_time_exponent(), TIME_EXPONENT_2D)
    t1 = 1.0
    t2 = 32.0
    radius1 = exact.shock_radius(t1)
    radius2 = exact.shock_radius(t2)
    np.testing.assert_allclose(radius2 / radius1, t2**TIME_EXPONENT_2D / t1**TIME_EXPONENT_2D)
    np.testing.assert_allclose(radius2 / radius1, 4.0)
    times = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    scaled = np.asarray([exact.shock_radius(float(t)) for t in times]) / times**TIME_EXPONENT_2D
    np.testing.assert_allclose(scaled, scaled[0])
    center = exact.blast_center()
    domain_center = exact.domain_center()
    assert center != domain_center
    assert exact.X0 != domain_center[0] or exact.Y0 != domain_center[1]


def test_constant_polar_radius_has_zero_radial_anisotropy():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in run_text
    assert "from exact import" not in run_text
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = exact.polar_shock_radius(theta, 1.0)
    np.testing.assert_allclose(radii, radii[0])
    np.testing.assert_allclose(radial_anisotropy(radii), 0.0)
    sampled = run.polar_radii(theta, 1.0)
    np.testing.assert_allclose(sampled, radii)
    np.testing.assert_allclose(run.front_anisotropy(theta, 1.0), 0.0)
    np.testing.assert_allclose(radial_anisotropy(sampled), 0.0)


def test_write_rb05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_rb05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


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
    plan = run.resolve_plan(8)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 2


def test_run_native_refuses_non_dim2(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM") as exc_info:
        run.run_native(8, t_end=0.01)
    message = str(exc_info.value)
    assert "fallback" in message.lower()
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.run_native(8, t_end=0.01)


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    missing = missing_compiler_requirement()
    try:
        conserved = run.run_native(16, t_end=0.05)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert set(conserved) == set(run.COMPONENT_ORDER)
    for field in conserved.values():
        array = np.asarray(field, dtype=np.float64)
        assert array.shape == (16, 16)
        assert np.isfinite(array).all()
    primitives = run.conserved_to_primitives(conserved)
    assert np.all(primitives["rho"] > 0.0)
    assert np.all(primitives["p"] > 0.0)
