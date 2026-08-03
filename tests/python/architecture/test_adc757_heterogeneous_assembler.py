from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    report = assembler.assemble(_measurements(), revision="candidate", minimum_speedup=1.01)
    assert verifier.validate(report, expected_revision="candidate")["status"] == "passed"


def test_adc757_assembler_refuses_measurements_that_are_not_abba_ordered() -> None:
    assembler = _module(ROOT / "benchmarks" / "adc757" / "assemble.py", "adc757_assemble_bad")
    measurements = _measurements()
    measurements[1], measurements[2] = measurements[2], measurements[1]
    measurements[1]["route"] = "baseline"
    with pytest.raises(assembler.AssemblyError, match="A,B,B,A"):
        assembler.assemble(measurements, revision="candidate", minimum_speedup=1.01)
