#!/usr/bin/env python3
"""Authenticate ADC-757 ABBA measurements and assemble the closure report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


MEASUREMENT_SCHEMA = "pops.adc757.heterogeneous-numerics.measurement.v1"
REPORT_SCHEMA = "pops.adc757.heterogeneous-numerics.v1"
RUNTIME_SCHEMA = "pops.adc757.installed-runtime-matrix.v1"
RUNTIME_SCENARIO = "adc757_amr_advection_runtime_v1"
RUNTIME_MODES = ("serial", "threaded", "gpu", "gpu_mpi")
SCENARIOS = ("prepared_local_time", "cost_aware_load_balance")
ROUTE_ORDER = ("baseline", "candidate", "candidate", "baseline")
METRICS = (
    "time_to_solution_seconds",
    "throughput_cell_updates_per_second",
    "memory_traffic_bytes",
    "kernel_launches",
    "task_count",
    "communication_bytes",
    "communication_seconds",
    "fallback_count",
    "useful_work_cell_updates",
    "imbalance_ratio",
    "migration_bytes",
    "migration_seconds",
)
CORRECTNESS = (
    "mass_error",
    "restart_max_error",
    "rollback_max_error",
    "ledger_balance_error",
)


class AssemblyError(ValueError):
    """Measurements cannot support an ADC-757 closure report."""


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssemblyError(f"{where} must be an object")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AssemblyError(f"{where} must be finite and non-negative")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_runtime_evidence(value: Any, *, revision: str) -> dict[str, Any]:
    evidence = _object(value, "installed runtime evidence")
    expected = {"schema", "status", "revision", "scenario_id", "modes", "authorities"}
    if set(evidence) != expected:
        raise AssemblyError(f"installed runtime evidence fields differ: {sorted(evidence)}")
    if evidence["schema"] != RUNTIME_SCHEMA:
        raise AssemblyError("installed runtime evidence has an unexpected schema")
    if evidence["status"] != "passed":
        raise AssemblyError("installed runtime evidence did not pass")
    if evidence["revision"] != revision:
        raise AssemblyError("installed runtime evidence belongs to another revision")
    if evidence["scenario_id"] != RUNTIME_SCENARIO:
        raise AssemblyError("installed runtime evidence used another scientific scenario")
    modes = evidence["modes"]
    if (
        not isinstance(modes, list)
        or not all(isinstance(item, dict) for item in modes)
        or [item.get("id") for item in modes] != list(RUNTIME_MODES)
    ):
        raise AssemblyError(
            f"installed runtime evidence must contain ordered modes {list(RUNTIME_MODES)}"
        )
    authorities = _object(evidence["authorities"], "installed runtime authorities")
    if set(authorities) != {"cell_local_time", "amr_rebalance_migration"}:
        raise AssemblyError("installed runtime evidence has incomplete authority receipts")
    for name, raw in authorities.items():
        authority = _object(raw, f"installed runtime authorities.{name}")
        if authority.get("consumed") is not True:
            raise AssemblyError(f"installed runtime authority {name} was not consumed")
    return evidence


def _load(path: Path) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AssemblyError(f"{path}:{line_number}: invalid JSON: {error}") from error
        measurement = _object(value, f"measurement[{line_number}]")
        if measurement.get("schema") != MEASUREMENT_SCHEMA:
            raise AssemblyError(f"{path}:{line_number}: unexpected measurement schema")
        measurements.append(measurement)
    if not measurements:
        raise AssemblyError("the ADC-757 campaign produced no measurements")
    return measurements


def _validate_measurement(value: dict[str, Any], *, revision: str) -> None:
    expected = {
        "schema",
        "status",
        "revision",
        "build_identity",
        "installed_wheel_sha256",
        "module_abi_sha256",
        "runtime_evidence_sha256",
        "execution_space",
        "mpi_ranks",
        "scenario",
        "route",
        "device_assignments",
        "streams",
        "metrics",
        "correctness",
    }
    if set(value) != expected:
        raise AssemblyError(f"measurement fields differ: {sorted(value)}")
    if value["status"] != "passed":
        raise AssemblyError("a hardware measurement did not pass its native checks")
    if value["revision"] != revision:
        raise AssemblyError("a hardware measurement belongs to another revision")
    if not isinstance(value["build_identity"], str) or not value["build_identity"]:
        raise AssemblyError("measurement build_identity must be non-empty")
    for field in (
        "installed_wheel_sha256",
        "module_abi_sha256",
        "runtime_evidence_sha256",
    ):
        digest = value[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise AssemblyError(f"measurement {field} must be lowercase sha256")
    if not isinstance(value["execution_space"], str) or not value["execution_space"]:
        raise AssemblyError("measurement execution_space must be non-empty")
    if isinstance(value["mpi_ranks"], bool) or not isinstance(value["mpi_ranks"], int):
        raise AssemblyError("measurement mpi_ranks must be an integer")
    if value["mpi_ranks"] < 2:
        raise AssemblyError("measurement requires at least two MPI ranks")
    if value["scenario"] not in SCENARIOS or value["route"] not in set(ROUTE_ORDER):
        raise AssemblyError("measurement has an unknown scenario or route")

    assignments = value["device_assignments"]
    if not isinstance(assignments, list) or len(assignments) != value["mpi_ranks"]:
        raise AssemblyError("device assignment count differs from mpi_ranks")
    ranks: list[int] = []
    uuids: list[str] = []
    for assignment in assignments:
        item = _object(assignment, "device assignment")
        if set(item) != {"rank", "uuid"}:
            raise AssemblyError("device assignment fields differ")
        ranks.append(item["rank"])
        uuids.append(item["uuid"])
    if sorted(ranks) != list(range(value["mpi_ranks"])) or len(set(uuids)) != len(uuids):
        raise AssemblyError("device assignments are incomplete or accelerator UUIDs alias")

    streams = _object(value["streams"], "measurement streams")
    if set(streams) != {
        "identities",
        "correctness_parity",
        "overlap_observed",
        "workspace_disjoint",
    }:
        raise AssemblyError("measurement stream fields differ")
    identities = streams["identities"]
    if not isinstance(identities, list) or len(identities) < 2:
        raise AssemblyError("measurement must contain at least two stream identities")
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise AssemblyError("stream identities must be non-empty strings")
    if len(set(identities)) != len(identities):
        raise AssemblyError("measurement stream identities alias")
    for field in ("correctness_parity", "overlap_observed", "workspace_disjoint"):
        if streams[field] is not True:
            raise AssemblyError(f"measurement did not prove streams.{field}")

    metrics = _object(value["metrics"], "measurement metrics")
    if set(metrics) != set(METRICS):
        raise AssemblyError("measurement metric fields differ")
    for name in METRICS:
        _finite(metrics[name], f"measurement metrics.{name}")
    if _finite(metrics["time_to_solution_seconds"], "time") <= 0.0:
        raise AssemblyError("measurement time must be positive")
    if _finite(metrics["throughput_cell_updates_per_second"], "throughput") <= 0.0:
        raise AssemblyError("measurement throughput must be positive")

    correctness = _object(value["correctness"], "measurement correctness")
    if set(correctness) != {"passed", *CORRECTNESS} or correctness["passed"] is not True:
        raise AssemblyError("measurement correctness is incomplete or failed")
    for name in CORRECTNESS:
        if _finite(correctness[name], f"correctness.{name}") > 1.0e-11:
            raise AssemblyError(f"measurement correctness.{name} exceeds 1e-11")


def _median_metrics(measurements: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: statistics.median(float(item["metrics"][name]) for item in measurements)
        for name in METRICS
    }


def assemble(
    measurements: list[dict[str, Any]],
    *,
    revision: str,
    minimum_speedup: float,
    runtime_evidence: Any,
) -> dict[str, Any]:
    if not math.isfinite(minimum_speedup) or minimum_speedup < 1.0:
        raise AssemblyError("minimum speedup must be finite and at least one")
    installed_runtime = _validate_runtime_evidence(runtime_evidence, revision=revision)
    runtime_evidence_sha256 = _canonical_sha256(installed_runtime)
    gpu_mpi = installed_runtime["modes"][-1]
    expected_wheel_sha256 = gpu_mpi["installation"].get("wheel_sha256")
    module_abi_key = gpu_mpi["artifact"].get("module_abi_key")
    if not isinstance(module_abi_key, str) or not module_abi_key:
        raise AssemblyError("installed GPU+MPI runtime has no module ABI key")
    expected_module_abi_sha256 = hashlib.sha256(module_abi_key.encode("utf-8")).hexdigest()
    for measurement in measurements:
        _validate_measurement(measurement, revision=revision)
        if measurement["runtime_evidence_sha256"] != runtime_evidence_sha256:
            raise AssemblyError(
                "a hardware measurement is not bound to the installed runtime evidence"
            )
        if measurement["installed_wheel_sha256"] != expected_wheel_sha256:
            raise AssemblyError("a hardware measurement used another installed wheel")
        if measurement["module_abi_sha256"] != expected_module_abi_sha256:
            raise AssemblyError("a hardware measurement used another installed module ABI")

    first = measurements[0]
    stable_fields = (
        "build_identity",
        "installed_wheel_sha256",
        "module_abi_sha256",
        "runtime_evidence_sha256",
        "execution_space",
        "mpi_ranks",
        "device_assignments",
    )
    for measurement in measurements[1:]:
        for field in stable_fields:
            if measurement[field] != first[field]:
                raise AssemblyError(f"measurement {field} changed during the campaign")

    reports: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        selected = [item for item in measurements if item["scenario"] == scenario]
        if len(selected) < 20 or len(selected) % 4 != 0:
            raise AssemblyError(f"{scenario} requires at least five complete ABBA blocks")
        blocks: list[list[float]] = []
        for offset in range(0, len(selected), 4):
            block = selected[offset : offset + 4]
            routes = tuple(item["route"] for item in block)
            if routes != ROUTE_ORDER:
                raise AssemblyError(f"{scenario} block {offset // 4} is not ordered A,B,B,A")
            blocks.append([float(item["metrics"]["time_to_solution_seconds"]) for item in block])
        baseline = [item for item in selected if item["route"] == "baseline"]
        candidate = [item for item in selected if item["route"] == "candidate"]
        correctness = {
            "passed": True,
            **{
                name: max(float(item["correctness"][name]) for item in selected)
                for name in CORRECTNESS
            },
        }
        reports.append(
            {
                "id": scenario,
                "baseline": _median_metrics(baseline),
                "candidate": _median_metrics(candidate),
                "correctness": correctness,
                "minimum_speedup": minimum_speedup,
                "abba_time_to_solution_seconds": blocks,
            }
        )

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    identities = [f"rank0:{identity}" for identity in first["streams"]["identities"]]
    topology = ";".join(
        f"rank={item['rank']},uuid={item['uuid']}" for item in first["device_assignments"]
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "provenance": {
            "revision": revision,
            "build_identity": first["build_identity"],
            "installed_wheel_sha256": first["installed_wheel_sha256"],
            "module_abi_sha256": first["module_abi_sha256"],
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "mpi_ranks": first["mpi_ranks"],
            "topology_identity": topology,
            "timestamp_utc": timestamp,
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
            "execution_space": first["execution_space"],
            "assignments": first["device_assignments"],
        },
        "streams": {
            "identities": identities,
            "correctness_parity": True,
            "overlap_observed": True,
            "workspace_disjoint": True,
        },
        "installed_runtime": installed_runtime,
        "scenarios": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-inventory-output", type=Path)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    args = parser.parse_args()
    try:
        runtime_evidence = json.loads(args.runtime_evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssemblyError(f"cannot load installed runtime evidence: {error}") from error
    report = assemble(
        _load(args.input),
        revision=args.expected_revision,
        minimum_speedup=args.minimum_speedup,
        runtime_evidence=runtime_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.device_inventory_output is not None:
        args.device_inventory_output.parent.mkdir(parents=True, exist_ok=True)
        assignments = report["device"]["assignments"]
        args.device_inventory_output.write_text(
            "".join(f"{item['rank']}\t{item['uuid']}\n" for item in assignments),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
