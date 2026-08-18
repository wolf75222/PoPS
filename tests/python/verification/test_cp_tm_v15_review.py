"""v1.5 review wave: required-fail, helper contract, runner artifacts, no invented facts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.campaign import CampaignJob, CampaignRequest, CampaignResources
from verification.pops_verify.capabilities import AuthenticatedArtifact
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_verification.py"
METRICS_SCHEMA = REPO_ROOT / "schemas" / "verification_metrics.v1.json"
PROVENANCE_SCHEMA = REPO_ROOT / "schemas" / "verification_provenance.v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_verification_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _artifact() -> AuthenticatedArtifact:
    return AuthenticatedArtifact(
        dimension=1,
        path=REPO_ROOT / "python",
        sha256="0" * 64,
        version="test",
        abi_key="test",
        has_mpi=False,
        has_kokkos=False,
        hdf5_collective=False,
        doctor_ok=False,
        native_variant_manifest_digest="0" * 64,
        native_header_signature="0" * 64,
        component_catalog_digest="0" * 64,
    )


def _request(case_id: str, *, evidence_status: str = "required", **kwargs) -> CampaignRequest:
    return CampaignRequest(
        case_id=case_id,
        pops_native_dim=int(kwargs.get("pops_native_dim", 1)),
        suite="pr",
        execution_space=str(kwargs.get("execution_space", "KokkosSerial")),
        mpi_mode=str(kwargs.get("mpi_mode", "off")),
        min_resolution=kwargs.get("min_resolution", 8),
        resources=CampaignResources(resolutions=(8,)),
        evidence_status=evidence_status,
        output_dir=kwargs.get("output_dir"),
    )


def test_helper_is_native_evidence_not_campaign_native():
    evidence = REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    competing = REPO_ROOT / "verification" / "pops_verify" / "campaign_native.py"
    assert evidence.is_file()
    assert not competing.exists()
    from verification.pops_verify import native_evidence

    assert hasattr(native_evidence, "maybe_campaign_payload")
    assert hasattr(native_evidence, "campaign_run_fields")


def test_maybe_campaign_payload_uses_result_and_cmake_build_type(monkeypatch):
    from verification.pops_verify.native_evidence import maybe_campaign_payload

    monkeypatch.setenv("CMAKE_BUILD_TYPE", "RelWithDebInfo")
    request = _request("CP-02")
    field = np.array([1.0, 2.0], dtype=np.float64)
    payload = maybe_campaign_payload(
        request,
        field,
        n_cells=8,
        t_end=0.25,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
    assert payload["result"] is field or np.array_equal(payload["result"], field)
    assert "field" not in payload
    assert payload["build_type"] == "RelWithDebInfo"
    assert payload["final_time"] == 0.25
    from verification.pops_verify.native_evidence import run_fields_from_payload
    from verification.pops_verify.provenance import RUN_FIELDS

    run_fields = run_fields_from_payload(payload)
    assert "result" not in run_fields
    assert set(run_fields) <= set(RUN_FIELDS)


def test_required_missing_capability_is_fail_never_not_supported(tmp_path: Path):
    from verification.pops_verify.case_authoring import load_sibling_module

    cases = (
        (
            REPO_ROOT / "verification" / "cases" / "euler_poisson" / "debye_screen",
            "write_cp09_report",
        ),
        (
            REPO_ROOT / "verification" / "cases" / "euler_poisson" / "diocotron",
            "write_cp11_report",
        ),
    )
    request = _request("CP-09", evidence_status="required")
    for case_dir, writer in cases:
        analyze = load_sibling_module(case_dir / "analyze.py")
        written = getattr(analyze, writer)(tmp_path / writer, request=request)
        assert written == ARTIFACTS
        loaded = json.loads((tmp_path / writer / "summary.json").read_text(encoding="utf-8"))
        assert loaded["coverage"]["cases_passed"] == 0
        assert loaded["coverage"]["cases_failed"] == 1
        assert loaded["coverage"]["cases_not_supported"] == 0
        assert loaded["failures"]


def test_frequency_error_is_not_mapped_to_phase_error():
    analyze = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "analyze.py"
    )
    times = np.linspace(0.0, 4.0 * np.pi, 128, endpoint=False)
    diagnostics = analyze.analyze_native(
        {"probe": np.cos(times), "times": times, "omega_ref": 1.0}
    )
    assert diagnostics["frequency_error"] < 0.05
    assert "phase_error" not in diagnostics
    from verification.pops_verify.native_evidence import native_report_sections

    sections = native_report_sections(diagnostics)
    assert sections["coupling"]["phase_error"] is None


def test_tm02_injected_orders_are_not_scientific_evidence():
    analyze = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "time" / "noncommuting_strang" / "analyze.py"
    )
    injected = analyze.analyze_native(
        {
            "lie": {"orders": (1.0, 1.0)},
            "strang": {"orders": (2.0, 2.0)},
        }
    )
    assert not injected.get("lie_orders")
    assert not injected.get("strang_orders")
    dts = (0.1, 0.05, 0.025)
    lie = (0.08, 0.04, 0.02)
    strang = (0.04, 0.01, 0.0025)
    derived = analyze.analyze_native(
        {
            "lie": {"linf": lie, "dts": dts},
            "strang": {"linf": strang, "dts": dts},
        }
    )
    np.testing.assert_allclose(derived["lie_orders"], observed_order(lie, dts))
    np.testing.assert_allclose(derived["strang_orders"], observed_order(strang, dts))


def test_tm02_official_euler_subflows_fail_honestly(tmp_path: Path):
    analyze = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "time" / "noncommuting_strang" / "analyze.py"
    )
    run_text = (
        REPO_ROOT / "verification" / "cases" / "time" / "noncommuting_strang" / "run.py"
    ).read_text(encoding="utf-8")
    assert "libtime.Lie" in run_text
    assert "libtime.Strang" in run_text
    assert "current + (fraction * program.dt) * change" in run_text
    written = analyze.write_tm02_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["coverage"]["cases_passed"] == 0
    blob = json.dumps(loaded).lower()
    assert "euler" in blob or "subflow" in blob or "strang" in blob


def test_tm05_mapping_records_actual_final_time_when_t_end_is_none():
    from verification.pops_verify.native_evidence import maybe_campaign_payload

    request = _request("TM-05")
    payload = maybe_campaign_payload(
        request,
        np.array([0.0], dtype=np.float64),
        n_cells=1,
        t_end=0.1,
        time_program="IMEX",
        cfl=0.0,
        dimension=1,
    )
    assert payload["final_time"] == 0.1
    run = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "time" / "ap_limit" / "run.py"
    )
    source = (
        REPO_ROOT / "verification" / "cases" / "time" / "ap_limit" / "run.py"
    ).read_text(encoding="utf-8")
    assert "horizon" in source or "actual" in source
    assert "t_end, request" not in source.replace(" ", "")
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert result["final_time"] == pytest.approx(0.1)


def test_misclassified_toys_fail_required_with_precise_blocker(tmp_path: Path):
    specs = (
        (
            "CP-05",
            REPO_ROOT / "verification" / "cases" / "euler_poisson" / "multifluid_modes",
            "multifluid",
            "poisson",
        ),
        (
            "CP-06",
            REPO_ROOT / "verification" / "cases" / "euler_poisson" / "ion_acoustic",
            "ion-acoustic",
            "poisson",
        ),
        (
            "TM-08",
            REPO_ROOT / "verification" / "cases" / "time" / "reversible_strang",
            "reversible",
            "strang",
        ),
    )
    for case_id, case_dir, token_a, token_b in specs:
        run = load_sibling_module(case_dir / "run.py")
        request = _request(case_id)
        with pytest.raises(run.NativeUnavailable) as info:
            run.run_native(request=request)
        message = str(info.value).lower()
        assert token_a in message
        assert token_b in message
        analyze = load_sibling_module(case_dir / "analyze.py")
        writer = next(
            name
            for name in dir(analyze)
            if name.startswith("write_") and name.endswith("_report")
        )
        written = getattr(analyze, writer)(tmp_path / case_id)
        assert written == ARTIFACTS
        loaded = json.loads((tmp_path / case_id / "summary.json").read_text(encoding="utf-8"))
        blob = json.dumps(loaded).lower()
        assert loaded["coverage"]["cases_passed"] == 0
        assert loaded["coverage"]["cases_not_supported"] == 0
        assert token_a in blob or token_b in blob


def test_owned_run_native_uses_bind_public_mpi_mode():
    owned = (
        REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "run.py",
        REPO_ROOT / "verification" / "cases" / "time" / "noncommuting_strang" / "run.py",
        REPO_ROOT / "verification" / "cases" / "time" / "ap_limit" / "run.py",
        REPO_ROOT / "verification" / "cases" / "euler_poisson" / "pressure_balance" / "run.py",
    )
    for path in owned:
        text = path.read_text(encoding="utf-8")
        assert "bind_public(" in text, path
        assert "mpi_mode" in text and "request" in text, path
        assert "pops.bind(" not in text or "def bind_public" in text


def test_fail_and_pass_write_schema_valid_metrics_via_parent_runner(tmp_path: Path):
    from verification.pops_verify.native_evidence import campaign_run_fields

    runner = _load_runner()
    artifact = _artifact()
    job = CampaignJob(
        case_id="CP-02",
        pops_native_dim=1,
        min_resolution=8,
        evidence_status="required",
    )
    fail_dir = tmp_path / "fail"
    fail_request = CampaignRequest.from_job(job, output_dir=fail_dir)
    fail_record = {"status": "fail", "reason": "no native Kokkos output"}
    runner._write_job_artifacts(job, None, fail_request, fail_record, artifact)
    metrics = json.loads((fail_dir / "metrics.json").read_text(encoding="utf-8"))
    Draft202012Validator(json.loads(METRICS_SCHEMA.read_text(encoding="utf-8"))).validate(
        metrics
    )
    assert not (fail_dir / "provenance.json").exists()

    pass_dir = tmp_path / "pass"
    pass_request = CampaignRequest.from_job(job, output_dir=pass_dir)
    fields = campaign_run_fields(
        request=pass_request,
        n_cells=8,
        t_end=0.05,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
    honest = runner._honest_run_fields(
        job,
        artifact,
        {**fields, "result": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert "result" not in honest
    pass_record = {"status": "pass", "reason": "authenticated fixture"}
    runner._write_job_artifacts(
        job, None, pass_request, pass_record, artifact, run_fields=honest
    )
    provenance = json.loads((pass_dir / "provenance.json").read_text(encoding="utf-8"))
    Draft202012Validator(
        json.loads(PROVENANCE_SCHEMA.read_text(encoding="utf-8")),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(provenance)
    assert "result" not in provenance
    assert provenance["final_time"] == 0.05


def test_scientific_array_is_unknown_provenance_run_field():
    from verification.pops_verify.native_evidence import maybe_campaign_payload
    from verification.pops_verify.provenance import ProvenanceError, collect_provenance

    request = _request("CP-02")
    payload = maybe_campaign_payload(
        request,
        np.array([1.0, 2.0], dtype=np.float64),
        n_cells=8,
        t_end=0.05,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
    extras = {key: value for key, value in payload.items() if key != "case_id"}
    with pytest.raises(ProvenanceError, match="unknown run field"):
        collect_provenance(
            "CP-02",
            pops_native_dim=1,
            dimension=1,
            nodes=1,
            pops_version="test",
            doctor_ok=False,
            component_catalog_digest="0" * 64,
            native_header_signature="0" * 64,
            native_variant_manifest_digest="0" * 64,
            **extras,
        )
