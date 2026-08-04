from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_mode(mode: str, backend: str, ranks: int, token: str) -> dict:
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
            "identity": f"artifact:{mode}",
            "abi_key": f"abi:{mode}",
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


def _runtime_evidence() -> dict:
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


def _runtime_digest() -> str:
    payload = json.dumps(
        _runtime_evidence(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _measurement(scenario: str, route: str, time: float) -> dict:
    candidate = route == "candidate"
    local_time = scenario == "prepared_local_time"
    work = 700.0 if local_time and candidate else 1000.0
    migration_bytes = 100_000.0 if not local_time and candidate else 0.0
    migration_seconds = 0.02 if not local_time and candidate else 0.0
    return {
        "schema": "pops.adc757.heterogeneous-numerics.measurement.v1",
        "status": "passed",
        "revision": "candidate",
        "build_identity": "nvcc_wrapper-Cuda",
        "installed_wheel_sha256": "4" * 64,
        "module_abi_sha256": hashlib.sha256(b"module:gpu_mpi").hexdigest(),
        "runtime_evidence_sha256": _runtime_digest(),
        "execution_space": "Cuda",
        "mpi_ranks": 2,
        "scenario": scenario,
        "route": route,
        "device_assignments": [
            {"rank": 0, "uuid": "GPU-0"},
            {"rank": 1, "uuid": "GPU-1"},
        ],
        "streams": {
            "identities": ["cuda:instance=2:lane=0", "cuda:instance=3:lane=1"],
            "correctness_parity": True,
            "overlap_observed": True,
            "workspace_disjoint": True,
        },
        "metrics": {
            "time_to_solution_seconds": time,
            "throughput_cell_updates_per_second": 1300.0 if candidate else 1000.0,
            "memory_traffic_bytes": 1_000_000.0,
            "kernel_launches": 18.0 if candidate else 32.0,
            "task_count": 20.0,
            "communication_bytes": migration_bytes,
            "communication_seconds": migration_seconds,
            "fallback_count": 0.0,
            "useful_work_cell_updates": work,
            "imbalance_ratio": 1.1 if candidate else 1.8,
            "migration_bytes": migration_bytes,
            "migration_seconds": migration_seconds,
        },
        "correctness": {
            "passed": True,
            "mass_error": 0.0,
            "restart_max_error": 0.0,
            "rollback_max_error": 0.0,
            "ledger_balance_error": 0.0,
        },
    }


def _measurements() -> list[dict]:
    values: list[dict] = []
    for scenario in ("prepared_local_time", "cost_aware_load_balance"):
        for _ in range(5):
            values.extend(
                [
                    _measurement(scenario, "baseline", 1.0),
                    _measurement(scenario, "candidate", 0.7),
                    _measurement(scenario, "candidate", 0.7),
                    _measurement(scenario, "baseline", 1.0),
                ]
            )
    return values


def test_adc757_assembler_builds_a_report_accepted_by_the_independent_verifier() -> None:
    assembler = _module(ROOT / "benchmarks" / "adc757" / "assemble.py", "adc757_assemble")
    verifier = _module(ROOT / "benchmarks" / "adc757" / "verify.py", "adc757_verify")
    report = assembler.assemble(
        _measurements(),
        revision="candidate",
        minimum_speedup=1.01,
        runtime_evidence=_runtime_evidence(),
    )
    assert verifier.validate(report, expected_revision="candidate")["status"] == "passed"


def test_adc757_assembler_refuses_measurements_that_are_not_abba_ordered() -> None:
    assembler = _module(ROOT / "benchmarks" / "adc757" / "assemble.py", "adc757_assemble_bad")
    measurements = _measurements()
    measurements[1], measurements[2] = measurements[2], measurements[1]
    measurements[1]["route"] = "baseline"
    with pytest.raises(assembler.AssemblyError, match="A,B,B,A"):
        assembler.assemble(
            measurements,
            revision="candidate",
            minimum_speedup=1.01,
            runtime_evidence=_runtime_evidence(),
        )


def test_adc757_assembler_refuses_a_nonpassing_installed_runtime() -> None:
    assembler = _module(ROOT / "benchmarks" / "adc757" / "assemble.py", "adc757_refusal")
    evidence = _runtime_evidence()
    evidence["status"] = "refused"
    with pytest.raises(assembler.AssemblyError, match="did not pass"):
        assembler.assemble(
            _measurements(),
            revision="candidate",
            minimum_speedup=1.01,
            runtime_evidence=evidence,
        )
