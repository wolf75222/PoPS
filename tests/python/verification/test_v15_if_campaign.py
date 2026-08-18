"""v1.5 IF-01…IF-10 campaign contracts: request, fail-closed bind, honest status."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from verification.pops_verify.campaign import CampaignJob, CampaignRequest, CampaignResources
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]

IF_CASES = (
    ("IF-01", "infrastructure/mpi_invariance"),
    ("IF-02", "infrastructure/thread_invariance"),
    ("IF-03", "infrastructure/space_parity"),
    ("IF-04", "infrastructure/checkpoint_restart"),
    ("IF-05", "infrastructure/output_cadence"),
    ("IF-06", "infrastructure/deterministic_reductions"),
    ("IF-07", "infrastructure/path_parity"),
    ("IF-08", "infrastructure/native_dim_guard"),
    ("IF-09", "infrastructure/float_precision"),
    ("IF-10", "infrastructure/hdf5_reread"),
)


def _case_dir(rel: str) -> Path:
    return REPO_ROOT / "verification" / "cases" / rel


def _load_run(rel: str):
    return load_sibling_module(_case_dir(rel) / "run.py")


def _request(case_id: str, **job_kwargs) -> CampaignRequest:
    return CampaignRequest.from_job(CampaignJob(case_id=case_id, pops_native_dim=1, **job_kwargs))


@pytest.mark.parametrize("case_id,rel", IF_CASES)
def test_run_native_accepts_campaign_request(case_id, rel):
    run = _load_run(rel)
    assert "request" in inspect.signature(run.run_native).parameters
    request = _request(case_id, min_resolution=8)
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable as exc:
        assert "skip" not in str(exc).lower()
        return
    except ValueError as exc:
        assert "mpi mode" in str(exc) or "execution space" in str(exc)
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "comparison_artifacts" in result


@pytest.mark.parametrize("case_id,rel", IF_CASES)
def test_invalid_mpi_mode_is_refused(case_id, rel):
    run = _load_run(rel)
    request = _request(case_id, mpi_mode="maybe")
    with pytest.raises((ValueError, run.NativeUnavailable), match="mpi mode"):
        run.run_native(request=request)


@pytest.mark.parametrize("case_id,rel", IF_CASES)
def test_missing_binary_is_not_supported_not_pytest_pass(case_id, rel, monkeypatch):
    run = _load_run(rel)
    request = _request(case_id, min_resolution=8)
    if hasattr(run, "_native_unavailable_reason"):
        monkeypatch.setattr(run, "_native_unavailable_reason", lambda: "compiler missing")
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable as exc:
        assert str(exc)
        assert "pytest.skip" not in str(exc)
        return
    assert result.get("status") in ("fail", "not-supported")


def test_if01_analytic_splits_are_not_mpi_proof():
    run = _load_run("infrastructure/mpi_invariance")
    assert run.analytic_placements_are_not_mpi_proof() is True
    request = _request("IF-01", mpi_mode="off", min_resolution=16)
    fields = run.campaign_run_fields(16, 0.25, request)
    assert "comparison_artifacts" in fields
    assert fields["comparison_artifacts"]["kind"] == "mpi_decomposition"


def test_if02_environment_only_threads_are_not_openmp_proof():
    run = _load_run("infrastructure/thread_invariance")
    serial = _request(
        "IF-02",
        execution_space="KokkosSerial",
        resources=CampaignResources(omp_threads=8),
    )
    with pytest.raises(run.NativeUnavailable, match="OpenMP"):
        run.run_native(request=serial)
    text = (_case_dir("infrastructure/thread_invariance") / "run.py").read_text(
        encoding="utf-8"
    )
    assert "environment-only" in text or "not OpenMP proof" in text


def test_if04_restart_semantics_require_half_time():
    run = _load_run("infrastructure/checkpoint_restart")
    assert run.json_round_trip_is_not_restart_proof() is True
    required = run.restart_semantic_fields()
    assert "checkpoint_time" in required
    assert "final_time" in required
    assert "continuous" in required
    assert "restarted" in required


def test_if10_npz_is_not_hdf5():
    run = _load_run("infrastructure/hdf5_reread")
    assert run.npz_round_trip_is_not_hdf5() is True
    text = (_case_dir("infrastructure/hdf5_reread") / "run.py").read_text(encoding="utf-8")
    assert "NPZ is not HDF5" in text or "npz_round_trip_is_not_hdf5" in text
    assert "read_hdf5" in text or "HDF5" in text
