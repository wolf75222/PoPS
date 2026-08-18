"""EU-02 isentropic vortex (2-d oracle; 1-d not applicable)."""
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
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "isentropic_vortex"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_mesh(exact, n_cells: int = N_CELLS):
    length = float(exact.PERIOD)
    width = length / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y


def test_density_and_pressure_are_positive():
    exact = _load("exact")
    x, y = _cell_mesh(exact)
    state = exact.exact_vortex(x, y, 0.0, u_inf=1.0, v_inf=0.0)
    assert np.all(np.asarray(state["rho"]) > 0.0)
    assert np.all(np.asarray(state["p"]) > 0.0)


def test_translation_by_one_zero_over_dt():
    exact = _load("exact")
    x, y = _cell_mesh(exact)
    dt = 0.5
    evolved = exact.exact_vortex(x, y, dt, u_inf=1.0, v_inf=0.0)
    shifted = exact.exact_vortex(x - dt, y, 0.0, u_inf=1.0, v_inf=0.0)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_allclose(
            evolved[key], shifted[key], rtol=0.0, atol=1.0e-14
        )


def test_entropy_function_is_constant_on_vortex():
    exact = _load("exact")
    x, y = _cell_mesh(exact)
    state = exact.exact_vortex(x, y, 0.25, u_inf=1.0, v_inf=0.0)
    entropy = exact.entropy_function(state["rho"], state["p"])
    background = exact.background()
    expected = exact.entropy_function(background["rho"], background["p"])
    np.testing.assert_allclose(entropy, expected, rtol=0.0, atol=1.0e-12)


def test_write_eu02_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load("analyze")
    written = analyze.write_eu02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    reasons = " ".join(item["reason"] for item in loaded["failures"]).lower()
    assert "native" in reasons


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text
        assert "from exact import" not in text


def test_pack_unpack_conserved_round_trip():
    run = _load("run")
    packed = run.pack_conserved(run.initial_conserved(N_CELLS))
    assert packed.shape == (4, N_CELLS, N_CELLS)
    assert packed.flags["C_CONTIGUOUS"]
    named = run.unpack_conserved(packed, N_CELLS)
    assert tuple(named) == run.COMPONENT_ORDER
    for name in run.COMPONENT_ORDER:
        np.testing.assert_array_equal(named[name], packed[run.COMPONENT_ORDER.index(name)])
    primitives = run.conserved_to_primitives(packed)
    assert np.all(primitives["rho"] > 0.0)
    assert np.all(primitives["p"] > 0.0)


def test_analyze_series_pass_shape_is_removed():
    analyze = _load("analyze")
    assert not hasattr(analyze, "analyze_series")


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load("run")
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


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_eu02_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1 or loaded["coverage"]["cases_not_supported"] == 1


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="EU-02", pops_native_dim=2, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
