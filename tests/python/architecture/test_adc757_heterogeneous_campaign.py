from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[3]
VERIFY = ROOT / "benchmarks" / "adc757" / "verify.py"
PROBE = ROOT / "benchmarks" / "adc757" / "runtime_probe.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module():
    return _load_module(VERIFY, "pops_adc757_hardware_verify")


def _metrics(
    *,
    time: float,
    throughput: float,
    work: float,
    imbalance: float,
    migration_bytes: float = 0.0,
    migration_seconds: float = 0.0,
) -> dict:
    return {
        "time_to_solution_seconds": time,
        "throughput_cell_updates_per_second": throughput,
        "memory_traffic_bytes": 1_000_000.0,
        "kernel_launches": 40,
        "task_count": 20,
        "communication_bytes": 10_000.0,
        "communication_seconds": 0.01,
        "fallback_count": 0,
        "useful_work_cell_updates": work,
        "imbalance_ratio": imbalance,
        "migration_bytes": migration_bytes,
        "migration_seconds": migration_seconds,
    }


def _correctness() -> dict:
    return {
        "passed": True,
        "mass_error": 0.0,
        "restart_max_error": 0.0,
        "rollback_max_error": 0.0,
        "ledger_balance_error": 0.0,
    }


def _runtime_mode(mode: str, backend: str, ranks: int, token: str) -> dict:
    artifact_identity = f"artifact:{mode}"
    abi_key = f"abi:{mode}"
    return {
        "id": mode,
        "scenario_id": "adc757_amr_advection_runtime_v1",
        "installation": {
            "wheel_name": f"pops-{mode}.whl",
            "wheel_sha256": token * 64,
            "installed_tree_sha256": "e" * 64,
            "native_sha256": "f" * 64,
            "package_file": f"/opt/pops-{mode}/pops/__init__.py",
            "native_extension": f"/opt/pops-{mode}/pops/_pops.so",
            "python_executable": f"/opt/pops-{mode}/bin/python",
            "outside_source_checkout": True,
        },
        "doctor": {"passed": True, "checks_sha256": "d" * 64},
        "artifact": {
            "identity": artifact_identity,
            "abi_key": abi_key,
            "module_abi_key": f"module:{mode}",
            "abi_compatible": True,
        },
        "execution": {
            "backend": backend,
            "mpi_ranks": ranks,
            "accepted_steps": 4,
            "final_time": 0.04,
            "solution_sha256": "a" * 64,
        },
    }


def _installed_runtime() -> dict:
    modes = [
        _runtime_mode("serial", "Serial", 1, "1"),
        _runtime_mode("threaded", "OpenMP", 1, "2"),
        _runtime_mode("gpu", "Cuda", 1, "3"),
        _runtime_mode("gpu_mpi", "Cuda", 2, "4"),
    ]
    gpu_mpi = modes[-1]["artifact"]
    common = {
        "consumed": True,
        "artifact_identity": gpu_mpi["identity"],
        "abi_key": gpu_mpi["abi_key"],
    }
    return {
        "schema": "pops.adc757.installed-runtime-matrix.v1",
        "status": "passed",
        "revision": "candidate",
        "scenario_id": "adc757_amr_advection_runtime_v1",
        "modes": modes,
        "authorities": {
            "cell_local_time": {
                **common,
                "identity": "pops.local-time.runtime@1",
                "accepted_steps": 4,
                "fallback_count": 0,
                "receipt_sha256": "b" * 64,
            },
            "amr_rebalance_migration": {
                **common,
                "identity": "pops.amr.rebalance.runtime@1",
                "moved_patches": 2,
                "migration_bytes": 4096,
                "post_migration_steps": 2,
                "receipt_sha256": "c" * 64,
            },
        },
    }


def _runtime_digest(runtime: dict) -> str:
    payload = json.dumps(runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _report() -> dict:
    runtime = _installed_runtime()
    gpu_mpi = runtime["modes"][-1]
    return {
        "schema": "pops.adc757.heterogeneous-numerics.v1",
        "status": "passed",
        "provenance": {
            "revision": "candidate",
            "build_identity": "headers+compiler+flags",
            "installed_wheel_sha256": gpu_mpi["installation"]["wheel_sha256"],
            "module_abi_sha256": hashlib.sha256(
                gpu_mpi["artifact"]["module_abi_key"].encode("utf-8")
            ).hexdigest(),
            "runtime_evidence_sha256": _runtime_digest(runtime),
            "mpi_ranks": 2,
            "topology_identity": "two-level-amr-two-rank",
            "timestamp_utc": "2026-08-03T00:00:00Z",
        },
        "protocol": {
            "ordering": "ABBA",
            "clock": "steady_clock",
            "device_fence": "before_and_after",
            "mpi_barrier": "before_and_after",
            "rank_aggregation": "max",
            "warmups": 2,
        },
        "device": {
            "execution_space": "Cuda",
            "assignments": [
                {"rank": 0, "uuid": "GPU-0"},
                {"rank": 1, "uuid": "GPU-1"},
            ],
        },
        "streams": {
            "identities": ["cuda:stream:0", "cuda:stream:1"],
            "correctness_parity": True,
            "overlap_observed": True,
            "workspace_disjoint": True,
        },
        "installed_runtime": runtime,
        "scenarios": [
            {
                "id": "prepared_local_time",
                "baseline": _metrics(time=1.0, throughput=100.0, work=1000, imbalance=1.6),
                "candidate": _metrics(time=0.8, throughput=125.0, work=700, imbalance=1.3),
                "correctness": _correctness(),
                "minimum_speedup": 1.02,
                "abba_time_to_solution_seconds": [[1.0, 0.8, 0.8, 1.0] for _ in range(5)],
            },
            {
                "id": "cost_aware_load_balance",
                "baseline": _metrics(time=1.0, throughput=100.0, work=1000, imbalance=1.8),
                "candidate": _metrics(
                    time=0.75,
                    throughput=133.0,
                    work=1000,
                    imbalance=1.1,
                    migration_bytes=100_000,
                    migration_seconds=0.02,
                ),
                "correctness": _correctness(),
                "minimum_speedup": 1.02,
                "abba_time_to_solution_seconds": [[1.0, 0.75, 0.75, 1.0] for _ in range(5)],
            },
        ],
    }


def _make_local_time_slow(report: dict) -> None:
    scenario = report["scenarios"][0]
    scenario["candidate"]["time_to_solution_seconds"] = 1.1
    scenario["abba_time_to_solution_seconds"] = [[1.0, 1.1, 1.1, 1.0] for _ in range(5)]


def test_adc757_hardware_report_accepts_complete_device_evidence() -> None:
    module = _module()
    assert module.validate(_report(), expected_revision="candidate")["status"] == "passed"


def test_adc757_installed_probe_never_promotes_header_presence_to_runtime_evidence() -> None:
    probe = _load_module(PROBE, "pops_adc757_runtime_probe")
    refusal = probe.refusal_payload(
        revision="candidate",
        installation={"wheel_sha256": "a" * 64},
        module_abi_key="module-abi",
        doctor={"passed": True, "checks_sha256": "b" * 64},
        support={
            "cell_local_time_commit_receipt_primitives": True,
            "amr_rebalance_migration_primitives": True,
        },
    )
    assert refusal["status"] == "refused"
    assert [item["code"] for item in refusal["blockers"]] == [
        "installed_runtime_matrix_receipts_unavailable"
    ]
    assert "vector kernels are not a PoPS runtime proof" in refusal["blockers"][0]["detail"]


def test_adc757_installed_probe_names_missing_c_and_g_routes_exactly() -> None:
    probe = _load_module(PROBE, "pops_adc757_runtime_probe_blockers")
    refusal = probe.refusal_payload(
        revision="candidate",
        installation={"wheel_sha256": "a" * 64},
        module_abi_key="module-abi",
        doctor={"passed": True, "checks_sha256": "b" * 64},
        support={
            "cell_local_time_commit_receipt_primitives": False,
            "amr_rebalance_migration_primitives": False,
        },
    )
    assert [item["code"] for item in refusal["blockers"]] == [
        "adc757g_local_time_runtime_unavailable",
        "adc757c_amr_migration_runtime_unavailable",
        "installed_runtime_matrix_receipts_unavailable",
    ]


def test_adc757_installed_probe_refuses_false_authority_receipts_before_abba() -> None:
    probe = _load_module(PROBE, "pops_adc757_runtime_probe_receipts")
    matrix = _installed_runtime()
    matrix["authorities"]["amr_rebalance_migration"]["consumed"] = False
    gpu_mpi = matrix["modes"][-1]
    installation = {
        **gpu_mpi["installation"],
        "revision": "candidate",
        "version": "1.0.0",
    }
    with pytest.raises(probe.RuntimeProbeError, match="not consumed"):
        probe._accept_external_matrix(
            matrix,
            revision="candidate",
            installation=installation,
            module_abi_key=gpu_mpi["artifact"]["module_abi_key"],
            doctor=gpu_mpi["doctor"],
            support={
                "cell_local_time_commit_receipt_primitives": True,
                "amr_rebalance_migration_primitives": True,
            },
        )


def test_adc757_romeo_driver_uses_only_the_exact_installed_candidate() -> None:
    cmake = (ROOT / "benchmarks" / "adc757" / "CMakeLists.txt").read_text(encoding="utf-8")
    job = (ROOT / "benchmarks" / "romeo" / "adc757_heterogeneous_numerics.sbatch").read_text(
        encoding="utf-8"
    )
    assert "POPS_ADC757_INCLUDE_ROOT" in cmake
    assert "pops_adc757_installed" in cmake
    assert "find_package(Kokkos CONFIG REQUIRED)" in cmake
    assert "find_package(MPI REQUIRED COMPONENTS CXX)" in cmake
    assert "find_package(pops" not in cmake
    assert "add_subdirectory" not in cmake
    assert 'scripts/build_python.sh" --mpi' in job
    assert "scripts/prove_installed_wheel.py" in job
    assert "benchmarks/adc757/runtime_probe.py" in job
    assert "toolchain.pops_include()" in job
    assert '-DPOPS_ADC757_INCLUDE_ROOT="${POPS_INCLUDE_ROOT}"' in job
    assert "--runtime-evidence-sha256" in job
    assert job.index("runtime_probe.py") < job.index("for scenario in")


def test_adc757_campaign_manifest_requires_the_complete_hardware_contract() -> None:
    manifest = tomllib.loads((ROOT / "benchmarks" / "manifest.toml").read_text(encoding="utf-8"))
    campaign = manifest["campaigns"]["adc757_heterogeneous_numerics"]
    assert campaign == {
        "routine_ci": False,
        "source": "benchmarks/adc757",
        "report_schema": "pops.adc757.heterogeneous-numerics.v1",
        "verifier": "benchmarks/adc757/verify.py",
        "requires_real_device": True,
        "requires_distinct_device_per_rank": True,
        "minimum_mpi_ranks": 2,
        "minimum_streams": 2,
        "requires_stream_overlap": True,
        "requires_restart_rollback_and_ledger_parity": True,
        "requires_exact_installed_wheels": True,
        "requires_pops_doctor": True,
        "runtime_scenario": "adc757_amr_advection_runtime_v1",
        "runtime_modes": ["serial", "threaded", "gpu", "gpu_mpi"],
        "required_runtime_authorities": ["cell_local_time", "amr_rebalance_migration"],
        "scenarios": ["prepared_local_time", "cost_aware_load_balance"],
        "metrics": list(_metrics(time=1.0, throughput=1.0, work=1.0, imbalance=1.0)),
        "job_script": "benchmarks/romeo/adc757_heterogeneous_numerics.sbatch",
        "submit_script": "benchmarks/romeo/submit_adc757_heterogeneous_numerics.sh",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["device"].update(execution_space="OpenMP"), "accelerator"),
        (
            lambda report: report["streams"].update(identities=["cuda:stream:0", "cuda:stream:0"]),
            "alias",
        ),
        (_make_local_time_slow, "speedup"),
        (
            lambda report: report["scenarios"][1]["candidate"].update(imbalance_ratio=2.0),
            "imbalance",
        ),
        (
            lambda report: report["scenarios"][0]["correctness"].update(restart_max_error=1.0),
            "restart_max_error",
        ),
        (
            lambda report: report["installed_runtime"]["modes"][0]["doctor"].update(passed=False),
            "doctor",
        ),
        (
            lambda report: report["installed_runtime"]["modes"][1]["execution"].update(
                solution_sha256="9" * 64
            ),
            "same solution",
        ),
        (
            lambda report: report["installed_runtime"]["authorities"]["cell_local_time"].update(
                consumed=False
            ),
            "not consumed",
        ),
    ],
)
def test_adc757_hardware_report_refuses_false_closure(mutation, message: str) -> None:
    module = _module()
    report = _report()
    mutation(report)
    with pytest.raises(module.EvidenceError, match=message):
        module.validate(report, expected_revision="candidate")
