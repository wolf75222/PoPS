"""GE-06 Cartesian diocotron + AMR companion (in-memory oracle; uniform Dim2 native)."""
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

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "geometry" / "diocotron_amr"
CP11_EXACT = (
    REPO_ROOT / "verification" / "cases" / "euler_poisson" / "diocotron" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
UNUSED_MODE = 3
FFT_ATOL = 1.0e-12


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


def _chebyshev_halo(mask, width: int) -> np.ndarray:
    """Cells at Chebyshev distance 1..width from a tagged cell, excluding the mask."""
    selected = np.asarray(mask, dtype=bool)
    added = np.zeros(selected.shape, dtype=bool)
    span = int(width)
    for shift_i in range(-span, span + 1):
        for shift_j in range(-span, span + 1):
            if shift_i == 0 and shift_j == 0:
                continue
            added |= np.roll(np.roll(selected, shift_i, axis=0), shift_j, axis=1)
    return added & ~selected


def test_tagged_set_covers_the_ring():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    import_lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert not any(line == "import pops" or line.startswith("import pops.") for line in import_lines)
    assert not any(line.startswith("from pops") for line in import_lines)
    assert exact.THETA > 0.0
    assert exact.THETA < (exact.N0 - exact.N_BG)
    tagged = run.raw_tag_mask()
    ring = run.ring_mask()
    assert np.any(ring)
    assert np.all(tagged[ring])
    field = run.sample_field()
    np.testing.assert_allclose(field[ring], exact.N0)
    np.testing.assert_allclose(field[~ring], exact.N_BG)


def test_two_level_envelope_is_tagged_plus_buffer_2():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert exact.BUFFER_CELLS == 2
    raw = run.raw_tag_mask()
    envelope = run.envelope_mask()
    added = envelope & ~raw
    expected = _chebyshev_halo(raw, 2)
    np.testing.assert_array_equal(added, expected)
    np.testing.assert_array_equal(envelope, raw | expected)
    assert np.any(added)
    farther = _chebyshev_halo(raw, 3) & ~expected
    assert not np.any(envelope & farther)
    assert np.all(envelope[run.ring_mask()])


def test_unused_mode_m3_fft_of_unperturbed_ring_is_near_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert exact.UNUSED_MODE == UNUSED_MODE
    amplitude = run.unused_mode_amplitude()
    assert np.isfinite(amplitude)
    assert amplitude <= FFT_ATOL
    _, samples = run.angular_density()
    np.testing.assert_allclose(samples, exact.N0, rtol=0.0, atol=FFT_ATOL)
    spectrum = exact.angular_fft(samples)
    assert float(np.abs(spectrum[UNUSED_MODE])) <= FFT_ATOL
    np.testing.assert_allclose(np.abs(spectrum[0]), exact.N0, rtol=0.0, atol=FFT_ATOL)


def test_write_ge06_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_ge06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["amr"]["invariants_ok"] is True


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text
        assert "diocotron" in text


def test_run_reuses_or_duplicates_cp11_ring():
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "euler_poisson" in run_text
    assert "diocotron" in run_text
    assert "load_sibling_module" in run_text
    if CP11_EXACT.is_file():
        assert "load_sibling_module" in exact_text or "euler_poisson" in run_text
    else:
        assert "r1" in exact_text.lower() or "R1" in exact_text
        assert "n0" in exact_text.lower() or "N0" in exact_text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
            assert "run_native" in owners
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_run_authors_cartesian_euler_poisson_not_polar():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "fields=" in text
    assert 'model.aux("potential")' in text
    assert 'model.aux("phi_grad_x")' in text
    assert "Cartesian2D" in text
    assert "AMRHierarchy" in text
    assert "GeometricMG" in text
    assert "PolarMesh" not in text
    assert "Polar2D" not in text
    reason = run.refuse_public_polar_runtime()
    assert isinstance(reason, str)
    assert reason.strip()
    assert "Cartesian only" in reason
    assert "polar" in reason.lower()


def test_cp11_density_is_not_evaluated_on_ge06_mesh():
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "do not evaluate" in run_text
    assert "_exact.unperturbed_density" in run_text
    assert "_cp11.density" not in run_text
    assert "_cp11.unperturbed_density" not in run_text


def test_resolve_plan_is_cartesian_dim2():
    run = _load_case_module("run")
    case = run.build_case(8)
    plan = run.resolve_plan(8)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 2


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(8, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (3, 8, 8)
    assert field.flags["C_CONTIGUOUS"]
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
