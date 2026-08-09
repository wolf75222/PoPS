#!/usr/bin/env python3
"""Validate real heterogeneous ADC-757 numerics/performance evidence.

This verifier never manufactures hardware evidence.  It consumes one report produced by the
non-routine device campaign and refuses CPU runs, aliased streams, incomplete numerical parity,
or a candidate that moves less useful work without improving end-to-end time to solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any


SCHEMA = "pops.adc757.heterogeneous-numerics.v1"
RUNTIME_SCHEMA = "pops.adc757.installed-runtime-matrix.v1"
RUNTIME_SCENARIO = "adc757_amr_advection_runtime_v1"
RUNTIME_MODES = ("serial", "threaded", "gpu", "gpu_mpi")
RUNTIME_AUTHORITY_IDENTITIES = {
    "cell_local_time": "pops.local-time.runtime@1",
    "amr_rebalance_migration": "pops.amr.rebalance.runtime@1",
}
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


_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonempty_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{where} must be non-empty text")
    return value


def _sha256(value: Any, where: str) -> str:
    text = _nonempty_text(value, where)
    if _SHA256.fullmatch(text) is None:
        raise EvidenceError(f"{where} must be one lowercase sha256 digest")
    return text


def _integer(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{where} must be an integer >= {minimum}")
    return value


def _validate_installation(raw: Any, where: str) -> None:
    installation = _mapping(raw, where)
    expected = {
        "wheel_name",
        "wheel_sha256",
        "installed_tree_sha256",
        "native_sha256",
        "package_file",
        "native_extension",
        "python_executable",
        "outside_source_checkout",
    }
    _exact_keys(installation, expected, where)
    wheel_name = _nonempty_text(installation["wheel_name"], f"{where}.wheel_name")
    if not wheel_name.endswith(".whl") or "/" in wheel_name or "\\" in wheel_name:
        raise EvidenceError(f"{where}.wheel_name must name one retained wheel")
    for name in ("wheel_sha256", "installed_tree_sha256", "native_sha256"):
        _sha256(installation[name], f"{where}.{name}")
    for name in ("package_file", "native_extension", "python_executable"):
        path = _nonempty_text(installation[name], f"{where}.{name}")
        if not Path(path).is_absolute():
            raise EvidenceError(f"{where}.{name} must be an absolute installed path")
    if installation["outside_source_checkout"] is not True:
        raise EvidenceError(f"{where} did not prove an installation outside the source checkout")


def _validate_doctor(raw: Any, where: str) -> None:
    doctor = _mapping(raw, where)
    _exact_keys(doctor, {"passed", "checks_sha256"}, where)
    if doctor["passed"] is not True:
        raise EvidenceError(f"{where}.passed must be true")
    _sha256(doctor["checks_sha256"], f"{where}.checks_sha256")


def _validate_runtime_mode(raw: Any, expected_id: str, *, scenario_id: str) -> tuple[str, str]:
    where = f"installed_runtime.modes[{expected_id}]"
    mode = _mapping(raw, where)
    _exact_keys(
        mode, {"id", "scenario_id", "installation", "doctor", "artifact", "execution"}, where
    )
    if mode["id"] != expected_id or mode["scenario_id"] != scenario_id:
        raise EvidenceError(f"{where} does not execute the required scenario")
    _validate_installation(mode["installation"], f"{where}.installation")
    _validate_doctor(mode["doctor"], f"{where}.doctor")

    artifact = _mapping(mode["artifact"], f"{where}.artifact")
    _exact_keys(
        artifact,
        {"identity", "abi_key", "module_abi_key", "abi_compatible"},
        f"{where}.artifact",
    )
    identity = _nonempty_text(artifact["identity"], f"{where}.artifact.identity")
    abi_key = _nonempty_text(artifact["abi_key"], f"{where}.artifact.abi_key")
    _nonempty_text(artifact["module_abi_key"], f"{where}.artifact.module_abi_key")
    if artifact["abi_compatible"] is not True:
        raise EvidenceError(f"{where}.artifact did not prove ABI compatibility")

    execution = _mapping(mode["execution"], f"{where}.execution")
    _exact_keys(
        execution,
        {"backend", "mpi_ranks", "accepted_steps", "final_time", "solution_sha256"},
        f"{where}.execution",
    )
    backend = _nonempty_text(execution["backend"], f"{where}.execution.backend")
    ranks = _integer(execution["mpi_ranks"], f"{where}.execution.mpi_ranks", minimum=1)
    _integer(execution["accepted_steps"], f"{where}.execution.accepted_steps", minimum=1)
    _positive(execution["final_time"], f"{where}.execution.final_time")
    solution = _sha256(execution["solution_sha256"], f"{where}.execution.solution_sha256")

    lower_backend = backend.lower()
    if expected_id == "serial":
        if "serial" not in lower_backend or ranks != 1:
            raise EvidenceError("serial runtime mode requires a one-rank Serial backend")
    elif expected_id == "threaded":
        if not any(token in lower_backend for token in ("openmp", "threads")) or ranks != 1:
            raise EvidenceError("threaded runtime mode requires a one-rank threaded backend")
    elif expected_id == "gpu":
        if not any(token in lower_backend for token in DEVICE_BACKENDS) or ranks != 1:
            raise EvidenceError("gpu runtime mode requires a one-rank accelerator backend")
    elif not any(token in lower_backend for token in DEVICE_BACKENDS) or ranks < 2:
        raise EvidenceError("gpu_mpi runtime mode requires an accelerator and at least two ranks")
    return solution, f"{identity}\0{abi_key}"


def _validate_authority(
    raw: Any,
    where: str,
    *,
    expected_identity: str,
    gpu_mpi_artifact: str,
    migration: bool,
) -> None:
    authority = _mapping(raw, where)
    common = {"consumed", "identity", "artifact_identity", "abi_key", "receipt_sha256"}
    specific = (
        {"moved_patches", "migration_bytes", "post_migration_steps"}
        if migration
        else {"accepted_steps", "fallback_count"}
    )
    _exact_keys(authority, common | specific, where)
    if authority["consumed"] is not True:
        raise EvidenceError(f"{where} was not consumed by the installed runtime")
    if authority["identity"] != expected_identity:
        raise EvidenceError(f"{where}.identity must be {expected_identity!r}")
    artifact_identity = _nonempty_text(authority["artifact_identity"], f"{where}.artifact_identity")
    abi_key = _nonempty_text(authority["abi_key"], f"{where}.abi_key")
    if f"{artifact_identity}\0{abi_key}" != gpu_mpi_artifact:
        raise EvidenceError(f"{where} belongs to another artifact or ABI")
    _sha256(authority["receipt_sha256"], f"{where}.receipt_sha256")
    if migration:
        _integer(authority["moved_patches"], f"{where}.moved_patches", minimum=1)
        _integer(authority["migration_bytes"], f"{where}.migration_bytes", minimum=1)
        _integer(authority["post_migration_steps"], f"{where}.post_migration_steps", minimum=1)
    else:
        _integer(authority["accepted_steps"], f"{where}.accepted_steps", minimum=1)
        if authority["fallback_count"] != 0:
            raise EvidenceError(f"{where}.fallback_count must be zero")


def validate_installed_runtime(raw: Any, *, expected_revision: str) -> dict[str, Any]:
    runtime = _mapping(raw, "installed_runtime")
    _exact_keys(
        runtime,
        {"schema", "status", "revision", "scenario_id", "modes", "authorities"},
        "installed_runtime",
    )
    if runtime["schema"] != RUNTIME_SCHEMA or runtime["status"] != "passed":
        raise EvidenceError("installed runtime matrix did not pass its exact contract")
    if runtime["revision"] != expected_revision:
        raise EvidenceError("installed runtime matrix belongs to another revision")
    scenario_id = _nonempty_text(runtime["scenario_id"], "installed_runtime.scenario_id")
    if scenario_id != RUNTIME_SCENARIO:
        raise EvidenceError("installed runtime matrix used another scientific scenario")
    modes = runtime["modes"]
    if not isinstance(modes, list) or len(modes) != len(RUNTIME_MODES):
        raise EvidenceError(f"installed runtime matrix requires exactly {list(RUNTIME_MODES)}")
    solutions: list[str] = []
    gpu_mpi_artifact = ""
    for raw_mode, expected_id in zip(modes, RUNTIME_MODES, strict=True):
        solution, artifact = _validate_runtime_mode(raw_mode, expected_id, scenario_id=scenario_id)
        solutions.append(solution)
        if expected_id == "gpu_mpi":
            gpu_mpi_artifact = artifact
    if len(set(solutions)) != 1:
        raise EvidenceError("installed runtime modes do not produce the same solution digest")

    authorities = _mapping(runtime["authorities"], "installed_runtime.authorities")
    _exact_keys(
        authorities,
        {"cell_local_time", "amr_rebalance_migration"},
        "installed_runtime.authorities",
    )
    _validate_authority(
        authorities["cell_local_time"],
        "installed_runtime.authorities.cell_local_time",
        expected_identity=RUNTIME_AUTHORITY_IDENTITIES["cell_local_time"],
        gpu_mpi_artifact=gpu_mpi_artifact,
        migration=False,
    )
    _validate_authority(
        authorities["amr_rebalance_migration"],
        "installed_runtime.authorities.amr_rebalance_migration",
        expected_identity=RUNTIME_AUTHORITY_IDENTITIES["amr_rebalance_migration"],
        gpu_mpi_artifact=gpu_mpi_artifact,
        migration=True,
    )
    return runtime


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
    if not math.isclose(baseline["time_to_solution_seconds"], measured_baseline, rel_tol=1.0e-12):
        raise EvidenceError(f"{expected_id} baseline summary differs from ABBA samples")
    if not math.isclose(candidate["time_to_solution_seconds"], measured_candidate, rel_tol=1.0e-12):
        raise EvidenceError(f"{expected_id} candidate summary differs from ABBA samples")
    speedup = statistics.median(ratios)
    if speedup < minimum_speedup:
        raise EvidenceError(
            f"{expected_id} speedup {speedup:.6g} is below required {minimum_speedup:.6g}"
        )
    if (
        candidate["throughput_cell_updates_per_second"]
        <= baseline["throughput_cell_updates_per_second"]
    ):
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
        {
            "schema",
            "status",
            "provenance",
            "protocol",
            "device",
            "streams",
            "installed_runtime",
            "scenarios",
        },
        "report",
    )
    if root["schema"] != SCHEMA:
        raise EvidenceError(f"unexpected report schema {root['schema']!r}")
    if root["status"] != "passed":
        raise EvidenceError("hardware campaign did not pass")
    provenance = _mapping(root["provenance"], "provenance")
    _exact_keys(
        provenance,
        {
            "revision",
            "build_identity",
            "installed_wheel_sha256",
            "module_abi_sha256",
            "runtime_evidence_sha256",
            "mpi_ranks",
            "topology_identity",
            "timestamp_utc",
        },
        "provenance",
    )
    if provenance["revision"] != expected_revision:
        raise EvidenceError("hardware evidence revision differs from the candidate revision")
    installed_runtime = validate_installed_runtime(
        root["installed_runtime"], expected_revision=expected_revision
    )
    expected_runtime_digest = _canonical_sha256(installed_runtime)
    if provenance["runtime_evidence_sha256"] != expected_runtime_digest:
        raise EvidenceError("hardware evidence is not bound to its installed runtime matrix")
    gpu_mpi = installed_runtime["modes"][-1]
    expected_wheel_sha256 = gpu_mpi["installation"]["wheel_sha256"]
    expected_module_abi_sha256 = hashlib.sha256(
        gpu_mpi["artifact"]["module_abi_key"].encode("utf-8")
    ).hexdigest()
    if provenance["installed_wheel_sha256"] != expected_wheel_sha256:
        raise EvidenceError("hardware evidence used another installed wheel")
    if provenance["module_abi_sha256"] != expected_module_abi_sha256:
        raise EvidenceError("hardware evidence used another installed module ABI")
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
