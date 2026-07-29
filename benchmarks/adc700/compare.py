#!/usr/bin/env python3
"""Fail-closed ADC-700 device/performance campaign comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = "pops.adc700.program_cutover.measurement.v1"
REPORT_SCHEMA = "pops.adc700.program_cutover.report.v1"
BASELINE_ROUTE = "pre_cutover_native"
CANDIDATE_ROUTE = "program_only"
DEVICE_TOKENS = ("cuda", "hip", "sycl")
SIGNATURE_FIELDS = ("mass", "checksum", "checksum_square", "maximum")


class CampaignError(ValueError):
    """The campaign evidence is incomplete or not comparable."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"{label} must be finite")
    return result


def _read_measurements(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CampaignError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            raise CampaignError(f"{path}:{line_number}: unexpected measurement schema")
        rows.append(row)
    if not rows:
        raise CampaignError("campaign input contains no measurements")
    return rows


def _median_and_mad(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def _read_device_assignments(path: Path, mpi_ranks: int) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        columns = raw.split("\t")
        if len(columns) != 2:
            raise CampaignError(
                f"{path}:{line_number}: expected '<rank>\\t<device UUID>'"
            )
        try:
            rank = int(columns[0])
        except ValueError as error:
            raise CampaignError(
                f"{path}:{line_number}: device rank must be an integer"
            ) from error
        device_uuid = columns[1].strip()
        if not device_uuid:
            raise CampaignError(f"{path}:{line_number}: device UUID is empty")
        assignments.append({"rank": rank, "uuid": device_uuid})

    observed_ranks = [assignment["rank"] for assignment in assignments]
    expected_ranks = list(range(mpi_ranks))
    if sorted(observed_ranks) != expected_ranks:
        raise CampaignError(
            f"device assignments cover ranks {sorted(observed_ranks)}, expected {expected_ranks}"
        )
    device_uuids = [assignment["uuid"] for assignment in assignments]
    if len(set(device_uuids)) != mpi_ranks:
        raise CampaignError("each MPI rank must be assigned one distinct device UUID")
    return sorted(assignments, key=lambda assignment: int(assignment["rank"]))


def _validate_common(
    rows: list[dict[str, Any]],
    baseline_revision: str,
    candidate_revision: str,
) -> tuple[str, int]:
    expected_sequence = [BASELINE_ROUTE, CANDIDATE_ROUTE, CANDIDATE_ROUTE, BASELINE_ROUTE]
    if len(rows) < 4 or len(rows) % 4 != 0:
        raise CampaignError("campaign requires one or more complete ABBA blocks")
    for block in range(len(rows) // 4):
        observed = [row.get("route") for row in rows[4 * block : 4 * block + 4]]
        if observed != expected_sequence:
            raise CampaignError(
                f"ABBA block {block} route order is {observed}, expected {expected_sequence}"
            )

    execution_spaces = {str(row.get("execution_space", "")) for row in rows}
    if len(execution_spaces) != 1:
        raise CampaignError(f"execution-space mismatch: {sorted(execution_spaces)}")
    execution_space = next(iter(execution_spaces))
    if not any(token in execution_space.lower() for token in DEVICE_TOKENS):
        raise CampaignError(
            f"real device backend required; Kokkos reported '{execution_space}'"
        )

    mpi_rank_values = [row.get("mpi_ranks") for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in mpi_rank_values):
        raise CampaignError("MPI rank count must be an integer in every measurement")
    mpi_ranks = set(mpi_rank_values)
    if len(mpi_ranks) != 1:
        raise CampaignError("MPI rank count differs across measurements")
    rank_count = int(next(iter(mpi_ranks)))
    if rank_count < 1:
        raise CampaignError("MPI rank count must be positive")

    comparable = ("execution_concurrency", "real_bytes", "parameters", "topology")
    reference = rows[0]
    for index, row in enumerate(rows):
        expected_revision = (
            baseline_revision if row["route"] == BASELINE_ROUTE else candidate_revision
        )
        if row.get("revision") != expected_revision:
            raise CampaignError(
                f"measurement {index} revision {row.get('revision')!r} != {expected_revision!r}"
            )
        for field in comparable:
            if row.get(field) != reference.get(field):
                raise CampaignError(f"measurement {index} differs in comparable field '{field}'")
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            raise CampaignError(f"measurement {index} failed its numerical validation")
        timing = row.get("timing")
        if not isinstance(timing, dict):
            raise CampaignError(f"measurement {index} lacks timing evidence")
        if _finite_number(
            timing.get("per_step_seconds"), f"measurement {index} per-step time"
        ) <= 0.0:
            raise CampaignError(f"measurement {index} has non-positive per-step time")
        signature = row.get("signature")
        if not isinstance(signature, dict):
            raise CampaignError(f"measurement {index} lacks a signature")
        for field in SIGNATURE_FIELDS:
            _finite_number(signature.get(field), f"measurement {index} signature.{field}")
    return execution_space, rank_count


def _performance(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ratios: list[float] = []
    for block in range(len(rows) // 4):
        a1, b1, b2, a2 = rows[4 * block : 4 * block + 4]
        baseline_1 = float(a1["timing"]["per_step_seconds"])
        candidate_1 = float(b1["timing"]["per_step_seconds"])
        candidate_2 = float(b2["timing"]["per_step_seconds"])
        baseline_2 = float(a2["timing"]["per_step_seconds"])
        ratio = math.exp(
            0.5
            * (
                math.log(baseline_1)
                + math.log(baseline_2)
                - math.log(candidate_1)
                - math.log(candidate_2)
            )
        )
        ratios.append(ratio)
    median, mad = _median_and_mad(ratios)
    return {
        "metric": "candidate_throughput_over_pre_cutover",
        "protocol": "paired_ABBA_geometric_ratio",
        "blocks": len(ratios),
        "ratios": ratios,
        "median": median,
        "mad": mad,
        "threshold": threshold,
        "passed": median >= threshold,
    }


def _numerical_parity(
    rows: list[dict[str, Any]], relative_tolerance: float, absolute_tolerance: float
) -> dict[str, Any]:
    baseline_rows = [row for row in rows if row["route"] == BASELINE_ROUTE]
    candidate_rows = [row for row in rows if row["route"] == CANDIDATE_ROUTE]
    fields: dict[str, Any] = {}
    passed = True
    for field in SIGNATURE_FIELDS:
        baseline = statistics.median(
            float(row["signature"][field]) for row in baseline_rows
        )
        candidate = statistics.median(
            float(row["signature"][field]) for row in candidate_rows
        )
        difference = abs(candidate - baseline)
        limit = absolute_tolerance + relative_tolerance * max(
            abs(baseline), abs(candidate)
        )
        field_passed = difference <= limit
        passed = passed and field_passed
        fields[field] = {
            "baseline_median": baseline,
            "candidate_median": candidate,
            "absolute_difference": difference,
            "limit": limit,
            "passed": field_passed,
        }
    return {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "fields": fields,
        "passed": passed,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if not (0.0 < args.threshold <= 1.0):
        raise CampaignError("--threshold must be in (0, 1]")
    if args.signature_rtol < 0.0 or args.signature_atol < 0.0:
        raise CampaignError("signature tolerances must be non-negative")

    rows = _read_measurements(args.input)
    execution_space, mpi_ranks = _validate_common(
        rows, args.baseline_revision, args.candidate_revision
    )
    assignments = _read_device_assignments(args.device_inventory, mpi_ranks)

    performance = _performance(rows, args.threshold)
    numerical = _numerical_parity(rows, args.signature_rtol, args.signature_atol)
    device = {
        "passed": True,
        "execution_space": execution_space,
        "assignments": assignments,
        "mpi_ranks": mpi_ranks,
        "one_distinct_device_per_rank": True,
    }
    passed = performance["passed"] and numerical["passed"] and device["passed"]
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "provenance": {
            "baseline_revision": args.baseline_revision,
            "candidate_revision": args.candidate_revision,
            "raw_measurements": str(args.input),
            "ordering": "ABBA",
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        },
        "device": device,
        "performance": performance,
        "numerical_parity": numerical,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-inventory", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--signature-rtol", type=float, default=1.0e-11)
    parser.add_argument("--signature-atol", type=float, default=1.0e-12)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = compare(args)
        exit_code = 0 if report["status"] == "passed" else 1
    except (CampaignError, OSError) as error:
        report = {
            "schema": REPORT_SCHEMA,
            "schema_version": 1,
            "status": "invalid",
            "errors": [str(error)],
        }
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
