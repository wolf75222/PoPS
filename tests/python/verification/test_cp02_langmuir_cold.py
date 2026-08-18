"""CP-02 cold Langmuir wave (in-memory closed 1-d oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.phase import frequency_error, numerical_frequency
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64
N_PERIODS = 8
SAMPLES_PER_PERIOD = 32
PROBE_X = 0.25


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _spectral_dx(field, spacing: float) -> np.ndarray:
    samples = np.asarray(field, dtype=np.float64)
    wave = 2.0 * np.pi * np.fft.fftfreq(samples.size, d=float(spacing))
    return np.fft.ifft(1j * wave * np.fft.fft(samples)).real


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("run_native"):
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_utility_oracle_omega_pe_units():
    exact = _load_case_module("exact")
    np.testing.assert_allclose(exact.E_CHARGE, 1.0)
    np.testing.assert_allclose(exact.M_E, 1.0)
    np.testing.assert_allclose(exact.EPS0, 1.0)
    np.testing.assert_allclose(exact.N0, 1.0)
    omega = np.sqrt(exact.N0 * exact.E_CHARGE**2 / (exact.M_E * exact.EPS0))
    np.testing.assert_allclose(omega, 1.0)
    np.testing.assert_allclose(exact.plasma_frequency(), 1.0)


def test_utility_oracle_gauss_law():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(N_CELLS)
    spacing = 1.0 / float(N_CELLS)
    for time in (0.0, 0.3, 1.25, 2.0 * np.pi):
        electric = exact.e_field(centers, time)
        density = exact.n_e(centers, time)
        dE_dx = _spectral_dx(electric, spacing)
        gauss_rhs = exact.E_CHARGE * (exact.N_I - density) / exact.EPS0
        np.testing.assert_allclose(dE_dx, gauss_rhs, rtol=0.0, atol=1.0e-12)


def test_utility_oracle_e_probe_frequency():
    exact = _load_case_module("exact")
    omega_pe = float(exact.plasma_frequency())
    period = 2.0 * np.pi / omega_pe
    times = np.arange(N_PERIODS * SAMPLES_PER_PERIOD, dtype=np.float64) * (
        period / SAMPLES_PER_PERIOD
    )
    samples = exact.e_field(PROBE_X, times)
    omega_fft = numerical_frequency(times, samples, method="fft")
    omega_fit = numerical_frequency(times, samples, method="phase_fit")
    np.testing.assert_allclose(frequency_error(omega_fft, omega_pe), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(frequency_error(omega_fit, omega_pe), 0.0, atol=1.0e-6)


def test_write_cp02_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_case_modules_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    tree = ast.parse(exact_text)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".", 1)[0])
    assert "pops" not in imported


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run(" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_compiler_run_native_returns_field_when_kokkos_present():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="CP-02",
        pops_native_dim=1,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=8,
        resources=CampaignResources(resolutions=(8,)),
        evidence_status="required",
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    assert "resolution" in result
    assert "result" in result


def test_report_fails_closed_without_native_output(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp02_report(tmp_path)
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert (
        loaded["coverage"]["cases_failed"] + loaded["coverage"]["cases_not_supported"]
        >= 1
    )
    reasons = " ".join(item["reason"] for item in loaded["failures"])
    notes = " ".join(loaded["coverage"].get("not_tested") or [])
    blob = (reasons + " " + notes).lower()
    assert (
        "native" in blob
        or "kokkos" in blob
        or "supported" in blob
        or "required" in blob
        or "not " in blob
        or "no " in blob
    )


def test_analyze_native_requires_native_field():
    analyze = _load_case_module("analyze")
    try:
        analyze.analyze_native({})
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError("analyze_native must refuse an empty mapping")


def test_analyze_native_computes_field_errors():
    analyze = _load_case_module("analyze")
    result = analyze.analyze_native(
        {
            "field": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "oracle": np.array([1.0, 2.0, 2.5], dtype=np.float64),
            "volumes": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        }
    )
    assert result["linf"] == 0.5
    assert result["l1"] > 0.0
    assert result["l2"] > 0.0


def test_write_report_stays_fail_closed_with_native_mapping(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp02_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0


def test_analyze_native_computes_frequency_from_probe():
    analyze = _load_case_module("analyze")
    times = np.linspace(0.0, 4.0 * np.pi, 128, endpoint=False)
    result = analyze.analyze_native(
        {
            "probe": np.cos(times),
            "times": times,
            "omega_ref": 1.0,
            "oracle_probe": np.cos(times),
        }
    )
    assert result["frequency_error"] < 0.05
    np.testing.assert_allclose(result["phase_error"], 0.0, atol=1.0e-12)


def test_initial_conserved_uses_cell_averages_not_point_samples():
    from verification.pops_verify.cell_averages import analytic_cell_averages

    run = _load_case_module("run")
    exact = _load_case_module("exact")
    n_cells = 16
    conserved = np.asarray(run.initial_conserved(n_cells), dtype=np.float64)
    centers, _ = exact.uniform_cell_centers(n_cells)
    point_n = exact.n_e(centers, 0.0)
    width = 1.0 / float(n_cells)
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    average_n = analytic_cell_averages(lambda x: exact.n_e(x, 0.0), lo, hi)
    np.testing.assert_allclose(conserved[0], average_n)
    assert not np.allclose(conserved[0], point_n, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(conserved[1], 0.0, atol=1.0e-15)


def test_oracle_producer_cp02_returns_cell_average_conserved():
    from verification.pops_verify.cell_averages import analytic_cell_averages
    from verification.pops_verify.oracle_producers import produce_oracle

    exact = _load_case_module("exact")
    n_cells = 32
    time = 0.3
    dummy = np.ones((2, n_cells), dtype=np.float64)
    oracle = produce_oracle(
        "CP-02",
        {
            "job": {
                "case_id": "CP-02",
                "pops_native_dim": 1,
                "min_resolution": n_cells,
            }
        },
        dummy,
        {"final_time": time},
    )
    width = 1.0 / float(n_cells)
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    density = analytic_cell_averages(lambda x: exact.n_e(x, time), lo, hi)
    momentum = analytic_cell_averages(
        lambda x: exact.n_e(x, time) * exact.u_e(x, time), lo, hi
    )
    np.testing.assert_allclose(oracle[0], density)
    np.testing.assert_allclose(oracle[1], momentum)


def test_run_native_uses_fixed_phase_dt_not_fluid_cfl():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "FixedDt" in text
    assert "AdaptiveCFL" not in text
    run = _load_case_module("run")
    step = float(run.phase_dt(16))
    exact = _load_case_module("exact")
    expected = 0.4 * (1.0 / 16.0) * exact.K / exact.plasma_frequency()
    np.testing.assert_allclose(step, expected)
    assert step < 0.5


def test_run_native_passes_artifact_for_program_bytes():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "artifact=artifact" in text
    assert "program_bytes" in text or "artifact=artifact" in text


def test_run_order_campaign_requires_request_and_native(tmp_path: Path):
    run = _load_case_module("run")
    with pytest.raises(run.NativeUnavailable):
        run.run_order_campaign(tmp_path, request=None)
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    request = CampaignRequest(
        case_id="CP-02",
        pops_native_dim=1,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=16,
        resources=CampaignResources(resolutions=(16, 32, 64, 128)),
        evidence_status="required",
    )
    try:
        run.run_order_campaign(tmp_path, request=request, resolutions=(16,))
    except run.NativeUnavailable as exc:
        message = str(exc).lower()
        assert (
            "native" in message
            or "kokkos" in message
            or "leaf" in message
            or "compiler" in message
        )
        return
    raise AssertionError("local campaign must not silently succeed without a leaf")


def test_write_cp02_report_refuses_injected_order_pass(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp02_report(
        tmp_path,
        native={
            "field": np.array([1.0, 1.0], dtype=np.float64),
            "oracle": np.array([1.0, 1.0], dtype=np.float64),
            "volumes": np.array([1.0, 1.0], dtype=np.float64),
            "orders": [2.0, 2.0, 2.0],
        },
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["orders"] == [] or loaded["coverage"]["cases_failed"] == 1


def test_write_campaign_visual_data_has_units_and_real_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    x = np.linspace(0.5 / 8.0, 1.0 - 0.5 / 8.0, 8)
    density = 1.0 + 1.0e-4 * np.cos(2.0 * np.pi * x)
    payload = analyze.write_campaign_visual_data(
        tmp_path,
        {
            "case_id": "CP-02",
            "source": "native",
            "resolutions": [16, 32, 64, 128],
            "l1": [1.0e-3, 2.6e-4, 6.6e-5, 1.7e-5],
            "l2": [1.1e-3, 2.8e-4, 7.1e-5, 1.8e-5],
            "linf": [1.4e-3, 3.6e-4, 9.1e-5, 2.3e-5],
            "orders": [1.94, 1.97, 1.98],
            "x": x,
            "density": {"numerical": density, "exact": density * 0.999, "error": density * 0.001},
            "velocity": {"numerical": np.zeros_like(x), "exact": np.zeros_like(x), "error": np.zeros_like(x)},
            "potential": {"numerical": -density, "exact": -density, "error": np.zeros_like(x)},
            "electric": {"numerical": np.sin(2.0 * np.pi * x), "exact": np.sin(2.0 * np.pi * x), "error": np.zeros_like(x)},
            "times": np.array([0.0, np.pi, 2.0 * np.pi], dtype=np.float64),
            "probe_e": [1.0e-4, 0.0, -1.0e-4],
            "omega_fft": 1.002,
            "omega_phase": 1.001,
            "omega_zero": 0.999,
            "ke": [1.0e-10, 0.0, 1.0e-10],
            "ese": [0.0, 1.0e-10, 0.0],
            "space_time_n": np.stack([density, density * 0.5, density]),
        },
    )
    visual_dir = tmp_path / "analysis" / "visual_data"
    for name in (
        "reference_profile",
        "signed_error_profile",
        "spatial_convergence",
        "phase_amplitude",
        "frequency_spectrum",
        "invariants_vs_time",
        "hero_figure",
    ):
        path = visual_dir / f"{name}.json"
        assert path.is_file(), name
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document.get("units")
        assert document.get("data_kind") == "campaign"
        if name != "hero_figure":
            assert document.get("series")
    assert payload["reference_profile"]["series"][0]["y"]


def test_closed_form_satisfies_linearized_eigenmode():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(64)
    spacing = 1.0 / 64.0
    dt = 1.0e-4
    time = 0.4
    omega = exact.plasma_frequency()
    n0 = exact.n_e(centers, time)
    n1 = exact.n_e(centers, time + dt)
    u0 = exact.u_e(centers, time)
    e0 = exact.e_field(centers, time)
    dndx = _spectral_dx(n0, spacing)
    dudx = _spectral_dx(u0, spacing)
    continuity = (n1 - n0) / dt + exact.N0 * dudx
    momentum = (exact.u_e(centers, time + dt) - u0) / dt - (exact.Q_E if hasattr(exact, "Q_E") else -exact.E_CHARGE) / exact.M_E * e0
    assert float(np.max(np.abs(continuity))) < 5.0e-6
    assert float(np.max(np.abs(momentum))) < 5.0e-6
    gauss = _spectral_dx(e0, spacing) - exact.E_CHARGE * (exact.N_I - n0) / exact.EPS0
    np.testing.assert_allclose(gauss, 0.0, atol=1.0e-12)
    assert float(np.max(np.abs(dndx))) > 0.0
    np.testing.assert_allclose(omega, 1.0)


def test_acceptance_reconstruction_is_public_weno5z():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "WENO5Z" in text
    assert "reconstruction.WENO5Z" in text or "WENO5Z()" in text
    run = _load_case_module("run")
    assert getattr(run, "DEFAULT_RECONSTRUCTION", None) == "weno5z"
    brick = run._reconstruction_brick("weno5z")
    assert getattr(brick, "name", None) == "weno5z"
    variant = run._reconstruction_brick("vanleer")
    assert getattr(variant, "name", None) == "muscl"


def test_frequency_official_omega_is_phase_fit_not_first_success():
    analyze = _load_case_module("analyze")
    times = np.linspace(0.0, 2.0 * np.pi, 65)
    probe = np.cos(times)
    block = analyze._frequency_block(times, probe, probe, 1.0)
    assert block["omega_num"] == block["omega_phase"]
    assert block["omega_fft"] is not None
    assert block["omega_zero"] is not None
    assert block["fft_official"] is False
    assert block["omega_num"] != block["omega_fft"] or block["e_omega"] == pytest.approx(
        abs(block["omega_num"] - 1.0) / 1.0, rel=0.0, abs=1.0e-12
    )
    assert block["e_omega"] == pytest.approx(
        abs(block["omega_num"] - 1.0) / 1.0, rel=0.0, abs=1.0e-12
    )
    assert block["method_disagreement"] is not None
    assert "harmonic_h2" in block
    assert "65-sample FFT" in block["omega_fft_note"] or "not omega_num" in block["omega_fft_note"]


def test_energy_baseline_is_total_energy_not_ese_oscillation():
    run = _load_case_module("run")
    block = run.energy_baseline([1.0, 0.0, 1.0], [1.0, 2.0, 1.0])
    assert block["conservation"] == "total_energy"
    assert block["max_relative_drift"] == pytest.approx(0.0)
    assert block["ese_oscillation"] == pytest.approx(1.0)
    assert block["ke_oscillation"] == pytest.approx(1.0)


def test_t0_electric_from_density_is_not_fake_zero():
    run = _load_case_module("run")
    exact = _load_case_module("exact")
    n_cells = 32
    density = run.cell_average_fields(n_cells, 0.0)["n"]
    phi, electric = run.fields_from_density(density)
    assert float(np.max(np.abs(electric))) > 0.0
    assert float(np.max(np.abs(phi))) > 0.0
    oracle = run.cell_average_fields(n_cells, 0.0)
    np.testing.assert_allclose(electric, oracle["e"], atol=1.0e-8)
    energy = run.energy_baseline([0.0, 1.0e-10], [0.0, 1.0e-10])
    assert energy["initial"] == pytest.approx(2.0e-10)
    assert energy["max_relative_drift"] is not None


def test_linear_eigenmode_vector_matches_closed_form():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(64)
    for time in (0.0, 0.4, 1.25, 2.0 * np.pi):
        mode = exact.linear_eigenmode_fields(centers, time)
        np.testing.assert_allclose(mode["n"], exact.n_e(centers, time))
        np.testing.assert_allclose(mode["u"], exact.u_e(centers, time), atol=1.0e-15)
        np.testing.assert_allclose(mode["e"], exact.e_field(centers, time), atol=1.0e-15)
        np.testing.assert_allclose(mode["phi"], exact.phi(centers, time), atol=1.0e-15)


def test_readme_and_case_do_not_overclaim_generic_pops_eigenmode():
    readme = (CASE_DIR / "README.md").read_text(encoding="utf-8")
    case = (CASE_DIR / "case.toml").read_text(encoding="utf-8")
    blob = (readme + "\n" + case).lower()
    assert "does not auto-generate" in blob or "closed_form_eigenmode" in blob
    assert "linear_eigenmode_and_closed_form" not in case


def test_run_temporal_and_spatial_campaigns_are_labeled():
    run = _load_case_module("run")
    assert callable(run.run_temporal_campaign)
    assert callable(run.run_spatial_campaign)
    assert run.TEMPORAL_N >= 256
    dts = run.default_temporal_dts()
    assert len(dts) >= 4
    assert dts[1] == pytest.approx(dts[0] / 2.0)
    assert dts[0] > run.phase_dt(run.TEMPORAL_N)
    spatial = run.phase_dt_h2(32)
    assert spatial < run.phase_dt(32)
    assert run.FREQUENCY_PERIODS >= 2
    stamps = run.default_sample_times(run.FREQUENCY_PERIODS * run.period())
    assert stamps.size == run.SAMPLES_PER_PERIOD
    assert stamps[-1] == pytest.approx(run.FREQUENCY_PERIODS * run.period())
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert 'family="temporal"' in text or "family='temporal'" in text
    assert 'family="spatial"' in text or "family='spatial'" in text
    assert 'family="global"' in text or "family='global'" in text or 'family: str = "global"' in text


def test_sixty_four_stamp_grid_clips_coarse_temporal_dt():
    run = _load_case_module("run")
    horizon = run.period()
    stamps = run.default_sample_times(horizon)
    dts = run.default_temporal_dts()
    assert run.stamp_grid_clips_dt(stamps, dts[0])
    assert run.stamp_grid_clips_dt(stamps, dts[1])
    final = run.final_order_sample_times(horizon)
    assert final.size == 1
    assert float(final[0]) == pytest.approx(horizon)
    for step in dts:
        assert not run.stamp_grid_clips_dt(final, step)


def test_temporal_family_uses_uninterrupted_horizon_and_distinct_steps():
    run = _load_case_module("run")
    horizon = run.period()
    stamps = run.sample_times_for_family("temporal", horizon)
    assert stamps.size == 1
    assert float(stamps[0]) == pytest.approx(horizon)
    dts = run.default_temporal_dts()
    steps = [run.expected_accepted_steps(horizon, step) for step in dts]
    assert steps == [16, 32, 64, 128]
    assert steps[0] != steps[1]
    assert len(set(steps)) == 4
    first = run.result_array_digest(np.full((2, 8), float(steps[0])))
    second = run.result_array_digest(np.full((2, 8), float(steps[1])))
    assert first != second


def test_run_native_temporal_is_one_uninterrupted_pops_run():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_native")
    source = ast.get_source_segment(text, fn)
    assert source is not None
    assert "sample_times_for_family" in source
    assert "final_order_sample_times" in text
    assert "accepted_steps" in source


def test_campaign_payload_uses_parent_coupling_kwarg():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "coupling=" in text
    from verification.pops_verify.campaign import CampaignJob, CampaignRequest
    from verification.pops_verify.native_evidence import maybe_campaign_payload

    payload = maybe_campaign_payload(
        CampaignRequest.from_job(CampaignJob(case_id="CP-02", pops_native_dim=1)),
        np.ones(8, dtype=np.float64),
        artifact=None,
        simulation=object(),
        coupling={"phase_error": None, "sign_ok": True, "energy_drift": None},
        n_cells=8,
        t_end=1.0,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
    assert payload["coupling"]["sign_ok"] is True
    assert "program_bytes" not in payload


def test_write_campaign_metrics_persists_frequency_diagnostics(tmp_path: Path):
    analyze = _load_case_module("analyze")
    x = np.linspace(0.5 / 8.0, 1.0 - 0.5 / 8.0, 8)
    density = 1.0 + 1.0e-4 * np.cos(2.0 * np.pi * x)
    campaign = {
        "case_id": "CP-02",
        "source": "native",
        "resolutions": [16, 32, 64, 128],
        "l1": [1.0e-3, 2.6e-4, 6.6e-5, 1.7e-5],
        "l2": [1.1e-3, 2.8e-4, 7.1e-5, 1.8e-5],
        "linf": [1.4e-3, 3.6e-4, 9.1e-5, 2.3e-5],
        "orders": [1.94, 1.97, 1.98],
        "x": x,
        "density": {"numerical": density, "exact": density * 0.999, "error": density * 0.001},
        "velocity": {"numerical": np.zeros_like(x), "exact": np.zeros_like(x), "error": np.zeros_like(x)},
        "potential": {"numerical": -density, "exact": -density, "error": np.zeros_like(x)},
        "electric": {
            "numerical": np.sin(2.0 * np.pi * x),
            "exact": np.sin(2.0 * np.pi * x),
            "error": np.zeros_like(x),
        },
        "times": np.array([0.0, np.pi, 2.0 * np.pi], dtype=np.float64),
        "probe_e": [1.0e-4, 0.0, -1.0e-4],
        "omega_fft": 0.98,
        "omega_phase": 0.999,
        "omega_zero": 1.01,
        "omega_num": 0.999,
        "e_omega": 0.001,
        "method_disagreement": 0.03,
        "harmonic_h2": 1.0e-8,
        "ke": [1.0e-10, 0.0, 1.0e-10],
        "ese": [1.0e-10, 2.0e-10, 1.0e-10],
        "mass": [1.0, 1.0, 1.0],
        "momentum": [0.0, 0.0, 0.0],
        "charge": [0.0, 0.0, 0.0],
        "sign_ok": True,
    }
    claim = {"orders": [1.94, 1.97, 1.98], "order_pass": True, "gated_orders": [1.97, 1.98]}
    analyze._write_campaign_metrics(tmp_path, campaign, claim)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    series = metrics["time_series"]
    assert series["e_omega"] == pytest.approx(0.001)
    assert series["method_disagreement"] == pytest.approx(0.03)
    assert series["harmonic_h2"] == pytest.approx(1.0e-8)
    assert series["omega_num"] == pytest.approx(0.999)
    assert series["fft_official"] is False
    metrics_full = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_full["conservation"]["electrostatic_energy"]["max_relative_drift"] is None
    assert series["ese_oscillation"] == pytest.approx(1.0e-10)
    visual = analyze.write_campaign_visual_data(tmp_path, campaign)
    phase = visual["phase_amplitude"]
    assert phase["e_omega"] == pytest.approx(0.001)
    assert phase["method_disagreement"] == pytest.approx(0.03)
    assert phase["harmonic_h2"] == pytest.approx(1.0e-8)
