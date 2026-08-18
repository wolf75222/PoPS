"""v1.5 IF-01…IF-10 campaign contracts: request, fail-closed bind, honest status."""
from __future__ import annotations

import inspect
import sys
import types
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


def _request(case_id: str, output_dir=None, **job_kwargs) -> CampaignRequest:
    return CampaignRequest.from_job(
        CampaignJob(case_id=case_id, pops_native_dim=1, **job_kwargs),
        output_dir=output_dir,
    )


def _patch_native_module(monkeypatch, module):
    selector = types.ModuleType("pops._native_selector")
    selector.selected_native_module = lambda *, required=False: module
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)


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
    assert result.get("status") != "not-supported"


@pytest.mark.parametrize("case_id,rel", IF_CASES)
def test_invalid_mpi_mode_is_refused(case_id, rel):
    run = _load_run(rel)
    request = _request(case_id, mpi_mode="maybe")
    with pytest.raises((ValueError, run.NativeUnavailable), match="mpi mode"):
        run.run_native(request=request)


@pytest.mark.parametrize("case_id,rel", IF_CASES)
def test_missing_binary_is_deterministic_injected_path(case_id, rel, tmp_path, monkeypatch):
    """A real on-disk native binary must not change this test. Inject a missing path."""
    run = _load_run(rel)
    missing = tmp_path / "no_such_native_leaf"
    assert not missing.exists()
    monkeypatch.setenv("POPS_VERIFY_NATIVE_EXE", str(missing))
    request = _request(case_id, min_resolution=8, output_dir=tmp_path / "out")
    with pytest.raises(run.NativeUnavailable, match="missing|unavailable|not found|no_such"):
        run.run_native(request=request)


def test_if01_analytic_splits_are_not_mpi_proof():
    run = _load_run("infrastructure/mpi_invariance")
    assert run.analytic_placements_are_not_mpi_proof() is True
    request = _request("IF-01", mpi_mode="off", min_resolution=16)
    fields = run.campaign_run_fields(16, 0.25, request)
    assert "comparison_artifacts" in fields
    assert fields["comparison_artifacts"]["kind"] == "mpi_decomposition"


def test_if01_refuses_launcher_env_without_native_world(monkeypatch):
    run = _load_run("infrastructure/mpi_invariance")
    _patch_native_module(monkeypatch, None)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("PMI_SIZE", "2")
    monkeypatch.setenv("POPS_CAMPAIGN_RANKS", "2")
    mpi = _request("IF-01", mpi_mode="on", min_resolution=16)
    with pytest.raises(run.NativeUnavailable, match="native|communicator|unavailable"):
        run.discovered_mpi_ranks()
    with pytest.raises(run.NativeUnavailable, match="native|communicator|unavailable"):
        run.campaign_run_fields(16, 0.25, mpi)


def test_if01_refuses_native_singleton_despite_slurm(monkeypatch):
    run = _load_run("infrastructure/mpi_invariance")

    class _Module:
        __has_mpi__ = True

        def n_ranks(self):
            return 1

        def my_rank(self):
            return 0

    _patch_native_module(monkeypatch, _Module())
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    mpi = _request("IF-01", mpi_mode="on", min_resolution=16)
    assert run.discovered_mpi_ranks() == 1
    with pytest.raises(run.NativeUnavailable, match="serial fallback|1 rank"):
        run.campaign_run_fields(16, 0.25, mpi)


def test_if01_records_native_world_size_not_launcher(monkeypatch):
    run = _load_run("infrastructure/mpi_invariance")

    class _Module:
        __has_mpi__ = True

        def n_ranks(self):
            return 2

        def my_rank(self):
            return 0

    _patch_native_module(monkeypatch, _Module())
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    mpi = _request("IF-01", mpi_mode="on", min_resolution=16)
    fields = run.campaign_run_fields(16, 0.25, mpi)
    assert fields["mpi_enabled"] is True
    assert fields["mpi_ranks"] == 2


def test_if02_environment_only_threads_are_not_openmp_proof():
    run = _load_run("infrastructure/thread_invariance")
    serial = _request(
        "IF-02",
        execution_space="KokkosSerial",
        resources=CampaignResources(omp_threads=8),
    )
    with pytest.raises(run.NativeUnavailable, match="OpenMP"):
        run.run_native(request=serial)


def test_if02_serial_backend_cannot_pass_as_openmp(monkeypatch):
    run = _load_run("infrastructure/thread_invariance")

    class _Resource:
        execution_backend = "Serial"

    class _Module:
        def native_execution_resource(self):
            return _Resource()

    _patch_native_module(monkeypatch, _Module())
    request = _request(
        "IF-02",
        execution_space="KokkosOpenMP",
        resources=CampaignResources(omp_threads=8),
    )
    with pytest.raises(run.NativeUnavailable, match="OpenMP|Serial"):
        run.run_native(request=request)


def test_if02_openmp_compares_native_outputs_across_threads(monkeypatch):
    run = _load_run("infrastructure/thread_invariance")

    class _Resource:
        execution_backend = "OpenMP"

    class _Module:
        def native_execution_resource(self):
            return _Resource()

    _patch_native_module(monkeypatch, _Module())
    seen: list[int] = []

    class _FakeTr01:
        NativeUnavailable = RuntimeError

        def run_native(self, n_cells, t_end=1.0):
            import os

            import numpy as np

            threads = int(os.environ["OMP_NUM_THREADS"])
            seen.append(threads)
            field = np.zeros(int(n_cells), dtype=np.float64)
            field[0] = float(threads)
            return field

    monkeypatch.setattr(run, "_tr01_run", lambda: _FakeTr01())
    request = _request(
        "IF-02",
        execution_space="KokkosOpenMP",
        min_resolution=16,
        resources=CampaignResources(omp_threads=4),
    )
    result = run.run_native(request=request)
    assert set(seen) >= {1, 4} or len(seen) >= 2
    artifacts = result["comparison_artifacts"]
    assert artifacts["kind"] == "openmp_threads"
    assert "pairwise_linf" in artifacts
    assert result.get("status") != "not-supported"


def test_if03_missing_openmp_backend_is_required_fail(monkeypatch):
    run = _load_run("infrastructure/space_parity")

    class _Resource:
        execution_backend = "Serial"

    class _Module:
        def native_execution_resource(self):
            return _Resource()

    _patch_native_module(monkeypatch, _Module())
    request = _request("IF-03", execution_space="KokkosOpenMP", min_resolution=8)
    with pytest.raises(run.NativeUnavailable, match="OpenMP|Serial"):
        run.run_native(request=request)


def test_if07_dsl_only_cannot_return_campaign_success():
    run = _load_run("infrastructure/path_parity")
    request = _request("IF-07", min_resolution=8)
    with pytest.raises(run.NativeUnavailable, match="hybrid|native"):
        run.run_native(request=request)
    source = inspect.getsource(run.run_native)
    assert "refuse_hybrid_native_cpp" in source


def test_if09_missing_float32_is_required_fail_not_not_supported():
    run = _load_run("infrastructure/float_precision")
    request = _request("IF-09", min_resolution=8)
    with pytest.raises(run.NativeUnavailable, match="float32"):
        run.run_native(request=request)
    text = (_case_dir("infrastructure/float_precision") / "run.py").read_text(encoding="utf-8")
    assert "not-supported" not in text


def test_if10_collective_flag_is_native_capability_not_request(monkeypatch, tmp_path):
    run = _load_run("infrastructure/hdf5_reread")

    class _Module:
        __has_parallel_hdf5__ = False
        __has_mpi__ = True

    _patch_native_module(monkeypatch, _Module())
    hdf5 = tmp_path / "state.h5"
    hdf5.write_bytes(b"\x89HDF\r\n\x1a\n" + b"not-a-full-file")
    assert run.authenticated_hdf5_collective(hdf5) is False
    request = _request("IF-10", mpi_mode="on", min_resolution=8)
    fields = run.campaign_hdf5_fields(request, hdf5)
    assert fields["hdf5_collective_enabled"] is False
    assert fields["mpi_enabled"] is True


def test_if10_npz_is_not_hdf5():
    run = _load_run("infrastructure/hdf5_reread")
    assert run.npz_round_trip_is_not_hdf5() is True
    text = (_case_dir("infrastructure/hdf5_reread") / "run.py").read_text(encoding="utf-8")
    assert "NPZ is not HDF5" in text or "npz_round_trip_is_not_hdf5" in text
    assert "read_hdf5" in text or "HDF5" in text
