from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMPARE = ROOT / "benchmarks" / "adc700" / "compare.py"
_TOOLCHAIN = {
    "nvcc_wrapper": {"path": "/toolchain/nvcc_wrapper", "sha256": "a" * 64, "version": "nvcc_wrapper"},
    "mpi": {"schema_version": 1, "compiler": "/toolchain/mpicxx", "compiler_version": "mpicxx",
            "compiler_sha256": "b" * 64, "abi_sha256": "c" * 64, "standard": "c++20",
            "compile_options": [], "compile_definitions": [], "link_options": [],
            "link_libraries": [], "include_dirs": ["/toolchain/mpi/include"], "headers": [], "libraries": []},
    "kokkos": {"schema_version": 1, "abi_sha256": "d" * 64,
                "include_dirs": ["/toolchain/kokkos/include"], "headers": [],
                "compile_options": [], "compile_definitions": [], "link_options": [],
                "link_libraries": []},
    "native_loader": {"schema_version": 1, "compile_definitions": ["POPS_NATIVE_DIM=2"]},
    "requested": {"cxx": "/toolchain/nvcc_wrapper", "include": "/toolchain/include", "std": "c++20",
                   "compile_flags": ["-DPOPS_NATIVE_DIM=2"], "link_flags": ["-ldl"]},
}
_TOOLCHAIN_RECEIPT = {"path": "/receipts/toolchain.json", "sha256": "e" * 64, "revision": "candidate"}


def _measurement(route: str, revision: str, seconds: float, *, device: str = "Cuda") -> dict:
    assignments = [{"rank": i, "uuid": "GPU-%d" % i} for i in range(4)]
    return {
        "schema": "pops.adc700.program_cutover.measurement.v1",
        "route": route,
        "revision": revision,
        "execution_space": device,
        "mpi_ranks": 4,
        "mpi_communicator": "MPI_COMM_WORLD",
        "toolchain_build_attested": True,
        "execution_concurrency": 1,
        "real_bytes": 8,
        "parameters": {"n": 32, "warmups": 2, "measured_steps": 8, "dt": 0.001},
        "topology": {
            "levels": 1,
            "patches": 0,
            "boxes": "",
            "distribute_coarse": True,
            "coarse_max_grid": 16,
            "coarse_local_boxes": 1,
            "coarse_total_boxes": 1,
        },
        "toolchain": _TOOLCHAIN,
        "toolchain_receipt": _TOOLCHAIN_RECEIPT,
        "gpu": {"rank": 0, "uuid": "GPU-0"},
        "gpu_uuid": "GPU-0",
        "gpu_assignments": assignments,
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
    assignments: str = "0\tGPU-0\n1\tGPU-1\n2\tGPU-2\n3\tGPU-3\n",
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
            "db3d390f43dfb14f12e88db31a9b3e631ff50488",
            "--candidate-revision",
            "candidate",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return process, json.loads(report.read_text(encoding="utf-8"))


def test_adc700_comparator_accepts_real_device_abba_evidence(tmp_path: Path) -> None:
    rows = []
    for _ in range(5):
        rows.extend([
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
            _measurement("program_only", "candidate", 1.01),
            _measurement("program_only", "candidate", 1.01),
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
        ])
    process, report = _run(tmp_path, rows)

    assert process.returncode == 0, process.stderr
    assert report["status"] == "passed"
    assert report["device"]["execution_space"] == "Cuda"
    assert report["device"]["assignments"] == [
        {"rank": 0, "uuid": "GPU-0"},
        {"rank": 1, "uuid": "GPU-1"},
        {"rank": 2, "uuid": "GPU-2"},
        {"rank": 3, "uuid": "GPU-3"},
    ]
    assert report["performance"]["median"] >= 0.98


def test_adc700_comparator_refuses_cpu_or_slow_evidence(tmp_path: Path) -> None:
    slow = []
    for _ in range(5):
        slow.extend([
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
            _measurement("program_only", "candidate", 1.1),
            _measurement("program_only", "candidate", 1.1),
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
        ])
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
    rows = []
    for _ in range(5):
        rows.extend([
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
            _measurement("program_only", "candidate", 1.0),
            _measurement("program_only", "candidate", 1.0),
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
        ])
    rows[1]["topology"]["boxes"] = "1:9,8,16,15"

    process, report = _run(tmp_path, rows)

    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "differs in comparable field 'topology'" in report["errors"][0]


def test_adc700_comparator_refuses_toolchain_or_coarse_layout_mismatch(tmp_path: Path) -> None:
    rows = []
    for _ in range(5):
        rows.extend([
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
            _measurement("program_only", "candidate", 1.0),
            _measurement("program_only", "candidate", 1.0),
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
        ])
    rows[1]["toolchain"] = {
        **rows[1]["toolchain"],
        "requested": {**rows[1]["toolchain"]["requested"], "std": "c++23"},
    }
    process, report = _run(tmp_path, rows)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "exact C++20" in report["errors"][0]

    rows[1]["toolchain"] = {
        **_TOOLCHAIN,
        "requested": {
            **_TOOLCHAIN["requested"],
            "compile_flags": ["-DPOPS_NATIVE_DIM=3"],
        },
    }
    process, report = _run(tmp_path, rows)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "comparable field 'toolchain'" in report["errors"][0]

    rows[1]["toolchain"] = _TOOLCHAIN
    rows[1]["toolchain_build_attested"] = False
    process, report = _run(tmp_path, rows)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "baseline CMake toolchain attestation" in report["errors"][0]

    rows[1]["toolchain_build_attested"] = True
    rows[1]["toolchain"] = _TOOLCHAIN
    rows[1]["topology"]["distribute_coarse"] = False
    process, report = _run(tmp_path, rows)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "distribute_coarse/coarse_max_grid" in report["errors"][0]


def test_adc700_comparator_refuses_kokkos_field_hash_or_flag_drift(tmp_path: Path) -> None:
    """Kokkos target fields and wrapper hashes are exact, not subset-presence evidence."""
    def rows() -> list[dict]:
        result = []
        for _ in range(5):
            result.extend([
                _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
                _measurement("program_only", "candidate", 1.0),
                _measurement("program_only", "candidate", 1.0),
                _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
            ])
        return result

    candidate = rows()
    candidate[1]["toolchain"] = {
        **candidate[1]["toolchain"],
        "kokkos": {
            key: value for key, value in candidate[1]["toolchain"]["kokkos"].items()
            if key != "link_libraries"
        },
    }
    process, report = _run(tmp_path, candidate)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "incomplete Kokkos provenance" in report["errors"][0]

    candidate = rows()
    candidate[1]["toolchain"] = {
        **candidate[1]["toolchain"],
        "kokkos": {
            key: value for key, value in candidate[1]["toolchain"]["kokkos"].items()
            if key != "abi_sha256"
        },
    }
    process, report = _run(tmp_path, candidate)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "incomplete Kokkos provenance" in report["errors"][0]

    candidate = rows()
    candidate[1]["toolchain"] = {
        **candidate[1]["toolchain"],
        "kokkos": {**candidate[1]["toolchain"]["kokkos"], "unexpected": "field"},
    }
    process, report = _run(tmp_path, candidate)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "incomplete Kokkos provenance" in report["errors"][0]

    candidate = rows()
    candidate[1]["toolchain"] = {
        **candidate[1]["toolchain"],
        "nvcc_wrapper": {
            **candidate[1]["toolchain"]["nvcc_wrapper"], "sha256": "f" * 64,
        },
    }
    process, report = _run(tmp_path, candidate)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "comparable field 'toolchain'" in report["errors"][0]

    candidate = rows()
    candidate[1]["toolchain"] = {
        **candidate[1]["toolchain"],
        "requested": {
            **candidate[1]["toolchain"]["requested"],
            "compile_flags": [],
        },
    }
    process, report = _run(tmp_path, candidate)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "incomplete production compile_flags" in report["errors"][0]

    candidate = rows()
    candidate[1]["toolchain"] = {
        **candidate[1]["toolchain"],
        "requested": {
            **candidate[1]["toolchain"]["requested"],
            "compile_flags": ["-DPOPS_NATIVE_DIM=2", "-DUNEXPECTED"],
        },
    }
    process, report = _run(tmp_path, candidate)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "comparable field 'toolchain'" in report["errors"][0]


def test_adc700_comparator_refuses_shared_or_missing_device_assignment(
    tmp_path: Path,
) -> None:
    rows = []
    for _ in range(5):
        rows.extend([
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
            _measurement("program_only", "candidate", 1.0),
            _measurement("program_only", "candidate", 1.0),
            _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
        ])

    process, report = _run(tmp_path, rows, assignments="0\tGPU-0\n1\tGPU-0\n2\tGPU-2\n3\tGPU-3\n")
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "one distinct device UUID" in report["errors"][0]

    process, report = _run(tmp_path, rows, assignments="0\tGPU-0\n1\tGPU-1\n2\tGPU-2\n")
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "expected [0, 1, 2, 3]" in report["errors"][0]


def test_adc700_comparator_refuses_short_abba_and_missing_per_run_uuid(tmp_path: Path) -> None:
    block = [
        _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
        _measurement("program_only", "candidate", 1.0),
        _measurement("program_only", "candidate", 1.0),
        _measurement("pre_cutover_native", "db3d390f43dfb14f12e88db31a9b3e631ff50488", 1.0),
    ]
    process, report = _run(tmp_path, block)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "at least 5 complete ABBA" in report["errors"][0]

    rows = block * 5
    rows[1] = {key: value for key, value in rows[1].items() if key != "gpu_assignments"}
    process, report = _run(tmp_path, rows)
    assert process.returncode == 2
    assert report["status"] == "invalid"
    assert "per-run GPU UUID" in report["errors"][0]
