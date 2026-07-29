from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMPARE = ROOT / "benchmarks" / "adc700" / "compare.py"


def _measurement(route: str, revision: str, seconds: float, *, device: str = "Cuda") -> dict:
    return {
        "schema": "pops.adc700.program_cutover.measurement.v1",
        "route": route,
        "revision": revision,
        "execution_space": device,
        "mpi_ranks": 2,
        "execution_concurrency": 1,
        "real_bytes": 8,
        "parameters": {"n": 32, "warmups": 2, "measured_steps": 8, "dt": 0.001},
        "topology": {
            "levels": 2,
            "patches": 4,
            "boxes": "1:8,8,15,15;1:16,8,23,15",
        },
        "timing": {
            "seconds": 8 * seconds,
            "per_step_seconds": seconds,
            "rank_aggregation": "max",
            "device_fence": "before_and_after",
            "mpi_barrier": "before_and_after",
        },
        "signature": {
            "mass": 1.0,
            "initial_mass": 1.0,
            "mass_error": 0.0,
            "checksum": 7.0,
            "checksum_square": 13.0,
            "maximum": 2.0,
        },
        "validation": {"passed": True, "mass_tolerance": 1.0e-9},
    }


def _run(
    tmp_path: Path,
    rows: list[dict],
    *,
    assignments: str = "0\tGPU-0\n1\tGPU-1\n",
) -> tuple[subprocess.CompletedProcess[str], dict]:
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    inventory = tmp_path / "devices.txt"
    inventory.write_text(assignments, encoding="utf-8")
    report = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--input",
            str(raw),
            "--output",
            str(report),
            "--device-inventory",
            str(inventory),
            "--baseline-revision",
            "baseline",
            "--candidate-revision",
            "candidate",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return process, json.loads(report.read_text(encoding="utf-8"))


def test_adc700_comparator_accepts_real_device_abba_evidence(tmp_path: Path) -> None:
    rows = [
        _measurement("pre_cutover_native", "baseline", 1.0),
        _measurement("program_only", "candidate", 1.01),
        _measurement("program_only", "candidate", 1.01),
        _measurement("pre_cutover_native", "baseline", 1.0),
    ]
    process, report = _run(tmp_path, rows)

    assert process.returncode == 0, process.stderr
    assert report["status"] == "passed"
    assert report["device"]["execution_space"] == "Cuda"
    assert report["device"]["assignments"] == [
        {"rank": 0, "uuid": "GPU-0"},
        {"rank": 1, "uuid": "GPU-1"},
    ]
    assert report["performance"]["median"] >= 0.98


def test_adc700_comparator_refuses_cpu_or_slow_evidence(tmp_path: Path) -> None:
    slow = [
        _measurement("pre_cutover_native", "baseline", 1.0),
        _measurement("program_only", "candidate", 1.1),
        _measurement("program_only", "candidate", 1.1),
        _measurement("pre_cutover_native", "baseline", 1.0),
    ]
    process, report = _run(tmp_path, slow)
    assert process.returncode == 1
    assert report["status"] == "failed"
    assert report["performance"]["passed"] is False

    cpu = [{**row, "execution_space": "OpenMP"} for row in slow]
    process, report = _run(tmp_path, cpu)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "real device backend required" in report["errors"][0]


def test_adc700_comparator_refuses_different_amr_topology(tmp_path: Path) -> None:
    rows = [
        _measurement("pre_cutover_native", "baseline", 1.0),
        _measurement("program_only", "candidate", 1.0),
        _measurement("program_only", "candidate", 1.0),
        _measurement("pre_cutover_native", "baseline", 1.0),
    ]
    rows[1]["topology"]["boxes"] = "1:9,8,16,15"

    process, report = _run(tmp_path, rows)

    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "differs in comparable field 'topology'" in report["errors"][0]


def test_adc700_comparator_refuses_shared_or_missing_device_assignment(
    tmp_path: Path,
) -> None:
    rows = [
        _measurement("pre_cutover_native", "baseline", 1.0),
        _measurement("program_only", "candidate", 1.0),
        _measurement("program_only", "candidate", 1.0),
        _measurement("pre_cutover_native", "baseline", 1.0),
    ]

    process, report = _run(tmp_path, rows, assignments="0\tGPU-0\n1\tGPU-0\n")
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "one distinct device UUID" in report["errors"][0]

    process, report = _run(tmp_path, rows, assignments="0\tGPU-0\n")
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "expected [0, 1]" in report["errors"][0]
