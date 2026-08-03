#!/usr/bin/env python3
"""Validate real heterogeneous ADC-757 numerics/performance evidence.

This verifier never manufactures hardware evidence.  It consumes one report produced by the
non-routine device campaign and refuses CPU runs, aliased streams, incomplete numerical parity,
or a candidate that moves less useful work without improving end-to-end time to solution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCHEMA = "pops.adc757.heterogeneous-numerics.v1"
DEVICE_BACKENDS = ("cuda", "hip", "sycl", "openmptarget")
SCENARIOS = ("prepared_local_time", "cost_aware_load_balance")
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


class EvidenceError(ValueError):
    """The supplied report is not closure-quality evidence."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{where} must be an object")
    return value


def _finite(value: Any, where: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{where} must be finite")
    if nonnegative and result < 0.0:
        raise EvidenceError(f"{where} must be non-negative")
    return result


def _positive(value: Any, where: str) -> float:
    result = _finite(value, where)
    if result <= 0.0:
        raise EvidenceError(f"{where} must be strictly positive")
    return result


def _exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        raise EvidenceError(
            f"{where} fields are {sorted(value)}, expected exactly {sorted(expected)}"
        )


def _validate_device(report: dict[str, Any], ranks: int) -> None:
    device = _mapping(report.get("device"), "device")
    _exact_keys(device, {"execution_space", "assignments"}, "device")
    execution_space = str(device["execution_space"])
    if not any(token in execution_space.lower() for token in DEVICE_BACKENDS):
        raise EvidenceError(
            f"device.execution_space must be a real accelerator backend, got {execution_space!r}"
        )
    assignments = device["assignments"]
    if not isinstance(assignments, list) or len(assignments) != ranks:
        raise EvidenceError("device.assignments must contain one entry per MPI rank")
    observed_ranks: list[int] = []
    identities: list[str] = []
    for index, raw in enumerate(assignments):
        item = _mapping(raw, f"device.assignments[{index}]")
        _exact_keys(item, {"rank", "uuid"}, f"device.assignments[{index}]")
        rank = item["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise EvidenceError(f"device.assignments[{index}].rank must be an integer")
        uuid = item["uuid"]
        if not isinstance(uuid, str) or not uuid:
            raise EvidenceError(f"device.assignments[{index}].uuid must be non-empty")
        observed_ranks.append(rank)
        identities.append(uuid)
    if sorted(observed_ranks) != list(range(ranks)):
        raise EvidenceError("device assignments do not cover every MPI rank exactly once")
    if len(set(identities)) != ranks:
        raise EvidenceError("each MPI rank must own one distinct accelerator UUID")


def _validate_streams(report: dict[str, Any]) -> None:
    streams = _mapping(report.get("streams"), "streams")
    _exact_keys(
        streams,
        {"identities", "correctness_parity", "overlap_observed", "workspace_disjoint"},
        "streams",
    )
    identities = streams["identities"]
    if not isinstance(identities, list) or len(identities) < 2:
        raise EvidenceError("streams.identities must contain at least two prepared streams")
    if any(not isinstance(value, str) or not value for value in identities):
        raise EvidenceError("every prepared stream identity must be a non-empty string")
    if len(set(identities)) != len(identities):
        raise EvidenceError("prepared stream identities alias one another")
    for field in ("correctness_parity", "overlap_observed", "workspace_disjoint"):
        if streams[field] is not True:
            raise EvidenceError(f"streams.{field} must be proved true")


def _validate_measurement(raw: Any, where: str) -> dict[str, float]:
    measurement = _mapping(raw, where)
    _exact_keys(measurement, set(METRICS), where)
    values = {name: _finite(measurement[name], f"{where}.{name}") for name in METRICS}
    _positive(values["time_to_solution_seconds"], f"{where}.time_to_solution_seconds")
    _positive(
        values["throughput_cell_updates_per_second"],
        f"{where}.throughput_cell_updates_per_second",
    )
    return values


def _validate_correctness(raw: Any, where: str) -> None:
    correctness = _mapping(raw, where)
    expected = {
        "passed",
        "mass_error",
        "restart_max_error",
        "rollback_max_error",
        "ledger_balance_error",
    }
    _exact_keys(correctness, expected, where)
    if correctness["passed"] is not True:
        raise EvidenceError(f"{where}.passed must be true")
    for name in expected - {"passed"}:
        value = _finite(correctness[name], f"{where}.{name}")
        if value > 1.0e-11:
            raise EvidenceError(f"{where}.{name}={value} exceeds 1e-11")


def _validate_scenario(raw: Any, expected_id: str) -> None:
    scenario = _mapping(raw, f"scenario[{expected_id}]")
    _exact_keys(
        scenario,
        {
            "id",
            "baseline",
            "candidate",
            "correctness",
            "minimum_speedup",
            "abba_time_to_solution_seconds",
        },
        f"scenario[{expected_id}]",
    )
    if scenario["id"] != expected_id:
        raise EvidenceError(
            f"scenario id {scenario['id']!r} appears where {expected_id!r} is required"
        )
    baseline = _validate_measurement(scenario["baseline"], f"{expected_id}.baseline")
    candidate = _validate_measurement(scenario["candidate"], f"{expected_id}.candidate")
    _validate_correctness(scenario["correctness"], f"{expected_id}.correctness")
    minimum_speedup = _positive(scenario["minimum_speedup"], f"{expected_id}.minimum_speedup")
    if minimum_speedup < 1.0:
        raise EvidenceError(f"{expected_id}.minimum_speedup must require a net benefit")
    blocks = scenario["abba_time_to_solution_seconds"]
    if not isinstance(blocks, list) or len(blocks) < 5:
        raise EvidenceError(f"{expected_id} requires at least five measured ABBA blocks")
    ratios: list[float] = []
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for index, raw_block in enumerate(blocks):
        if not isinstance(raw_block, list) or len(raw_block) != 4:
            raise EvidenceError(f"{expected_id} ABBA block {index} must contain A,B,B,A")
        a1, b1, b2, a2 = (
            _positive(value, f"{expected_id}.abba[{index}][{column}]")
            for column, value in enumerate(raw_block)
        )
        baseline_samples.extend((a1, a2))
        candidate_samples.extend((b1, b2))
        ratios.append(math.sqrt((a1 * a2) / (b1 * b2)))
    measured_baseline = statistics.median(baseline_samples)
    measured_candidate = statistics.median(candidate_samples)
    if not math.isclose(
        baseline["time_to_solution_seconds"], measured_baseline, rel_tol=1.0e-12
    ):
        raise EvidenceError(f"{expected_id} baseline summary differs from ABBA samples")
    if not math.isclose(
        candidate["time_to_solution_seconds"], measured_candidate, rel_tol=1.0e-12
    ):
        raise EvidenceError(f"{expected_id} candidate summary differs from ABBA samples")
    speedup = statistics.median(ratios)
    if speedup < minimum_speedup:
        raise EvidenceError(
            f"{expected_id} speedup {speedup:.6g} is below required {minimum_speedup:.6g}"
        )
    if candidate["throughput_cell_updates_per_second"] <= baseline[
        "throughput_cell_updates_per_second"
    ]:
        raise EvidenceError(f"{expected_id} does not improve measured throughput")
    if expected_id == "prepared_local_time":
        if candidate["useful_work_cell_updates"] >= baseline["useful_work_cell_updates"]:
            raise EvidenceError("prepared local time does not reduce useful-work updates")
        if candidate["fallback_count"] != 0.0:
            raise EvidenceError("prepared local time silently used a fallback")
    else:
        if candidate["imbalance_ratio"] >= baseline["imbalance_ratio"]:
            raise EvidenceError("cost-aware load balance does not reduce observed imbalance")
        if candidate["migration_bytes"] <= 0.0 or candidate["migration_seconds"] <= 0.0:
            raise EvidenceError("load-balance evidence must include measured migration cost")


def validate(report: Any, *, expected_revision: str) -> dict[str, Any]:
    root = _mapping(report, "report")
    _exact_keys(
        root,
        {"schema", "status", "provenance", "protocol", "device", "streams", "scenarios"},
        "report",
    )
    if root["schema"] != SCHEMA:
        raise EvidenceError(f"unexpected report schema {root['schema']!r}")
    if root["status"] != "passed":
        raise EvidenceError("hardware campaign did not pass")
    provenance = _mapping(root["provenance"], "provenance")
    _exact_keys(
        provenance,
        {"revision", "build_identity", "mpi_ranks", "topology_identity", "timestamp_utc"},
        "provenance",
    )
    if provenance["revision"] != expected_revision:
        raise EvidenceError("hardware evidence revision differs from the candidate revision")
    for name in ("build_identity", "topology_identity", "timestamp_utc"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise EvidenceError(f"provenance.{name} must be non-empty")
    ranks = provenance["mpi_ranks"]
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 2:
        raise EvidenceError("hardware evidence requires at least two MPI ranks")
    protocol = _mapping(root["protocol"], "protocol")
    expected_protocol = {
        "ordering": "ABBA",
        "clock": "steady_clock",
        "device_fence": "before_and_after",
        "mpi_barrier": "before_and_after",
        "rank_aggregation": "max",
        "warmups": 2,
    }
    if protocol != expected_protocol:
        raise EvidenceError(f"protocol must be exactly {expected_protocol}")
    _validate_device(root, ranks)
    _validate_streams(root)
    scenarios = root["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != len(SCENARIOS):
        raise EvidenceError(f"scenarios must contain exactly {list(SCENARIOS)}")
    for raw, expected_id in zip(scenarios, SCENARIOS, strict=True):
        _validate_scenario(raw, expected_id)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
        validate(report, expected_revision=args.expected_revision)
    except (EvidenceError, OSError, json.JSONDecodeError) as error:
        print(f"ADC-757 hardware evidence refused: {error}", file=sys.stderr)
        return 2
    print("ADC-757 hardware evidence: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
