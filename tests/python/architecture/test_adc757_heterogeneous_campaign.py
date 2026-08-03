from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[3]
VERIFY = ROOT / "benchmarks" / "adc757" / "verify.py"


def _module():
    spec = importlib.util.spec_from_file_location("pops_adc757_hardware_verify", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(*, time: float, throughput: float, work: float, imbalance: float,
             migration_bytes: float = 0.0, migration_seconds: float = 0.0) -> dict:
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


def _report() -> dict:
    return {
        "schema": "pops.adc757.heterogeneous-numerics.v1",
        "status": "passed",
        "provenance": {
            "revision": "candidate",
            "build_identity": "headers+compiler+flags",
            "mpi_ranks": 2,
            "topology_identity": "two-level-amr-two-rank",
            "timestamp_utc": "2026-08-03T00:00:00Z",
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
        "scenarios": [
            {
                "id": "prepared_local_time",
                "baseline": _metrics(time=1.0, throughput=100.0, work=1000, imbalance=1.6),
                "candidate": _metrics(time=0.8, throughput=125.0, work=700, imbalance=1.3),
                "correctness": _correctness(),
                "minimum_speedup": 1.02,
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
            },
        ],
    }


def test_adc757_hardware_report_accepts_complete_device_evidence() -> None:
    module = _module()
    assert module.validate(_report(), expected_revision="candidate")["status"] == "passed"


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
        "scenarios": ["prepared_local_time", "cost_aware_load_balance"],
        "metrics": list(_metrics(
            time=1.0, throughput=1.0, work=1.0, imbalance=1.0
        )),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["device"].update(execution_space="OpenMP"), "accelerator"),
        (
            lambda report: report["streams"].update(
                identities=["cuda:stream:0", "cuda:stream:0"]
            ),
            "alias",
        ),
        (
            lambda report: report["scenarios"][0]["candidate"].update(
                time_to_solution_seconds=1.1
            ),
            "speedup",
        ),
        (
            lambda report: report["scenarios"][1]["candidate"].update(
                imbalance_ratio=2.0
            ),
            "imbalance",
        ),
        (
            lambda report: report["scenarios"][0]["correctness"].update(
                restart_max_error=1.0
            ),
            "restart_max_error",
        ),
    ],
)
def test_adc757_hardware_report_refuses_false_closure(mutation, message: str) -> None:
    module = _module()
    report = _report()
    mutation(report)
    with pytest.raises(module.EvidenceError, match=message):
        module.validate(report, expected_revision="candidate")
