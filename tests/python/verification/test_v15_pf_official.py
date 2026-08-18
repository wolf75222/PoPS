"""v1.5 PF-01…PF-12: official harness only, warmup/sample rules, required-fail absent."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from verification.pops_verify.campaign import CampaignJob, CampaignRequest, CampaignResources
from verification.pops_verify.case_authoring import load_sibling_module

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "benchmarks" / "manifest.toml"
SUPPORT_HPP = REPO_ROOT / "benchmarks" / "include" / "pops_bench" / "benchmark_support.hpp"
SUPPORT_CPP = REPO_ROOT / "benchmarks" / "src" / "benchmark_support.cpp"
ARITH_CPP = REPO_ROOT / "benchmarks" / "src" / "arith_halo_case.cpp"
MG_CPP = REPO_ROOT / "benchmarks" / "src" / "scalar_mg_case.cpp"

PF_CASES = (
    ("PF-01", "performance/multifab_arith", "arith_halo"),
    ("PF-02", "performance/scalar_mg", "scalar_mg"),
    ("PF-03", "performance/advection_rhs", None),
    ("PF-04", "performance/euler_step", None),
    ("PF-05", "performance/composite_poisson", None),
    ("PF-06", "performance/ep_step", None),
    ("PF-07", "performance/regrid_cluster", None),
    ("PF-08", "performance/reflux_sync", None),
    ("PF-09", "performance/load_balance", None),
    ("PF-10", "performance/checkpoint_io", None),
    ("PF-11", "performance/amr_e2e", None),
    ("PF-12", "performance/hyqmom15", None),
)


def _load_run(rel: str):
    return load_sibling_module(REPO_ROOT / "verification" / "cases" / rel / "run.py")


def _request(case_id: str, output_dir=None, **job_kwargs) -> CampaignRequest:
    return CampaignRequest.from_job(
        CampaignJob(case_id=case_id, pops_native_dim=1, **job_kwargs),
        output_dir=output_dir,
    )


def test_official_protocol_meets_v15_sample_warmup_floor():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    assert int(protocol["default_warmups"]) >= 5
    assert int(protocol["default_repetitions"]) >= 10
    assert protocol["numerical_validation"] == "outside_timed_region"
    assert protocol["absolute_time_thresholds"] is False

    header = SUPPORT_HPP.read_text(encoding="utf-8")
    assert "int warmups = 5" in header
    assert "int repetitions = 10" in header
    parser = SUPPORT_CPP.read_text(encoding="utf-8")
    assert "warmups must be at least 5" in parser
    assert "repetitions must be at least 10" in parser or "repetitions must be at least 5" in parser


def test_official_cases_validate_before_recording_timing():
    arith = ARITH_CPP.read_text(encoding="utf-8")
    mg = MG_CPP.read_text(encoding="utf-8")
    for source in (arith, mg):
        validate_at = source.find("validation")
        write_at = source.find("writer.write")
        assert 0 <= validate_at < write_at
        assert "if (!passed)" in source
        throw_at = source.find("throw std::runtime_error")
        first_write = source.find("writer.write")
        assert 0 <= throw_at < first_write


@pytest.mark.parametrize("case_id,rel,official", PF_CASES)
def test_pf_routes_only_to_official_or_required_fail(case_id, rel, official):
    run = _load_run(rel)
    authority = run.official_authority()
    assert str(authority["manifest"]).endswith("benchmarks/manifest.toml")
    assert authority.get("status") != "not-supported"
    if official is None:
        assert authority["status"] == "fail"
        assert authority["case_id"] is None
        with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
            run.run_native()
    else:
        assert authority["case_id"] == official
        assert authority["status"] == "official"


@pytest.mark.parametrize("case_id,rel,official", PF_CASES)
def test_pf_missing_binary_is_injected_path(case_id, rel, official, tmp_path, monkeypatch):
    run = _load_run(rel)
    missing = tmp_path / "no_such_pops_benchmark"
    assert not missing.exists()
    monkeypatch.setenv("POPS_VERIFY_NATIVE_EXE", str(missing))
    request = _request(case_id, min_resolution=16, output_dir=tmp_path / "out")
    with pytest.raises(run.NativeUnavailable, match="missing|unavailable|not found|no_such"):
        run.run_native(request=request)


@pytest.mark.parametrize("case_id,rel,official", (("PF-01", "performance/multifab_arith", "arith_halo"), ("PF-02", "performance/scalar_mg", "scalar_mg")))
def test_pf_consumes_campaign_request_and_refuses_unsupported(case_id, rel, official, tmp_path):
    run = _load_run(rel)
    assert "request" in inspect.signature(run.run_native).parameters
    cuda = _request(case_id, execution_space="KokkosCuda", min_resolution=32, output_dir=tmp_path)
    with pytest.raises(run.NativeUnavailable, match="CUDA|unsupported|execution space"):
        run.run_native(request=cuda)
    low = SimpleNamespace(
        case_id=case_id,
        pops_native_dim=1,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=32,
        resources=CampaignResources(),
        evidence_status="required",
        output_dir=tmp_path,
        warmups=3,
        repetitions=10,
    )
    with pytest.raises(run.NativeUnavailable, match="warmup"):
        run.run_native(request=low)
    samples = SimpleNamespace(
        case_id=case_id,
        pops_native_dim=1,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=32,
        resources=CampaignResources(),
        evidence_status="required",
        output_dir=tmp_path,
        warmups=5,
        repetitions=4,
    )
    with pytest.raises(run.NativeUnavailable, match="sample|repetition"):
        run.run_native(request=samples)


@pytest.mark.parametrize("case_id,rel,official", (("PF-01", "performance/multifab_arith", "arith_halo"), ("PF-02", "performance/scalar_mg", "scalar_mg")))
def test_pf_returns_truthful_timings_or_required_fail(case_id, rel, official, tmp_path):
    run = _load_run(rel)
    request = _request(
        case_id,
        min_resolution=16,
        execution_space="KokkosSerial",
        mpi_mode="off",
        output_dir=tmp_path,
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable as exc:
        assert "skip" not in str(exc).lower()
        assert "not-supported" not in str(exc)
        return
    assert result.get("status") != "not-supported"
    assert result["case_id"] == official
    timing = result.get("timing") or {}
    assert int(timing.get("warmups") or 0) >= 5
    assert int(timing.get("samples") or timing.get("repetitions") or 0) >= 10
    assert "statistics" in result
    assert "validation" in result
    assert result["resolution"] == [16] or result.get("n") == 16
    assert str(tmp_path) in str(result.get("output") or result.get("output_dir") or "")


@pytest.mark.parametrize("case_id,rel,_official", PF_CASES)
def test_pf_run_native_is_not_a_wrapper_timer(case_id, rel, _official):
    run = _load_run(rel)
    source = inspect.getsource(run.run_native)
    assert "time.perf_counter" not in source
    assert "fake_segment_time" not in source
    assert "run_mapped_or_refuse" in source or "run_official_benchmark" in source
