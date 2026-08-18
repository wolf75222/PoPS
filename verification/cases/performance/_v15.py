"""Official PF mapping. Absent official benches are required fails, not wrappers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from verification.pops_verify.official_benchmark import (
    DEFAULT_EXECUTABLE,
    OFFICIAL_MANIFEST,
    OfficialBenchmarkUnavailable,
    official_case_ids,
    refuse_unofficial_pf,
)

OFFICIAL_PF_MAP = {
    "PF-01": "arith_halo",
    "PF-02": "scalar_mg",
}
MIN_WARMUPS = 5
MIN_SAMPLES = 10
INJECTED_NATIVE_ENV = "POPS_VERIFY_NATIVE_EXE"
ALLOWED_MPI_MODES = ("off", "on")
ALLOWED_SPACES = ("KokkosSerial", "KokkosOpenMP")


def official_authority(verification_id: str) -> dict:
    """Return the official bench for a PF id, or a required fail if absent."""
    mapped = OFFICIAL_PF_MAP.get(verification_id)
    if mapped is None:
        return {
            "verification_id": verification_id,
            "manifest": str(OFFICIAL_MANIFEST),
            "case_id": None,
            "status": "fail",
            "reason": refuse_unofficial_pf(verification_id),
        }
    return {
        "verification_id": verification_id,
        "manifest": str(OFFICIAL_MANIFEST),
        "case_id": mapped,
        "status": "official",
    }


def _injected_executable() -> Path | None:
    raw = os.environ.get(INJECTED_NATIVE_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    return Path(raw)


def _require_request(request) -> None:
    if request is None:
        return
    mode = getattr(request, "mpi_mode", "off")
    if mode not in ALLOWED_MPI_MODES:
        raise OfficialBenchmarkUnavailable(f"invalid mpi mode {mode!r}")
    space = getattr(request, "execution_space", "KokkosSerial")
    if space == "KokkosCuda":
        raise OfficialBenchmarkUnavailable(
            "unsupported execution space KokkosCuda; official PF-01/PF-02 have no public CUDA space"
        )
    if space not in ALLOWED_SPACES:
        raise OfficialBenchmarkUnavailable(f"unsupported execution space {space!r}")
    if mode == "on":
        try:
            from pops._native_selector import selected_native_module

            module = selected_native_module(required=False)
        except Exception:
            module = None
        if module is None or getattr(module, "__has_mpi__", False) is not True:
            raise OfficialBenchmarkUnavailable(
                "mpi_mode=on requires an authenticated native MPI communicator"
            )
    warmups = int(getattr(request, "warmups", MIN_WARMUPS) or MIN_WARMUPS)
    samples = int(
        getattr(request, "repetitions", None)
        or getattr(request, "samples", MIN_SAMPLES)
        or MIN_SAMPLES
    )
    if warmups < MIN_WARMUPS:
        raise OfficialBenchmarkUnavailable(
            f"warmup count {warmups} is below the official floor of {MIN_WARMUPS}"
        )
    if samples < MIN_SAMPLES:
        raise OfficialBenchmarkUnavailable(
            f"sample/repetition count {samples} is below the official floor of {MIN_SAMPLES}"
        )


def _resolution(request) -> int | None:
    if request is None:
        return None
    if request.min_resolution is not None:
        return int(request.min_resolution)
    resolutions = getattr(getattr(request, "resources", None), "resolutions", ()) or ()
    if resolutions:
        return int(resolutions[0])
    return None


def _output_path(request, case_id: str) -> Path:
    if request is not None and getattr(request, "output_dir", None) is not None:
        out_dir = Path(request.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{case_id}.jsonl"
    return Path("benchmarks.jsonl")


def _parse_jsonl(path: Path) -> dict:
    records = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    statistics = None
    validation = {"passed": False, "records": 0}
    samples = 0
    for record in records:
        stats = record.get("stats") or record.get("statistics")
        if isinstance(stats, dict) and statistics is None:
            statistics = stats
            samples = int(stats.get("count") or 0)
        if record.get("record_type") == "validation" or "validation" in record:
            payload = record.get("validation") if isinstance(record.get("validation"), dict) else record
            validation = {
                "passed": bool(payload.get("passed", True)),
                "records": validation["records"] + 1,
            }
        elif record.get("record_type") == "measurement":
            validation["records"] += 1
            validation["passed"] = True
    if statistics is None:
        statistics = {"count": samples}
    return {
        "records": records,
        "statistics": statistics,
        "validation": validation,
        "samples": samples or int(statistics.get("count") or 0),
    }


def run_official_for_request(case_id: str, request=None) -> dict:
    """Run pops_benchmark with CampaignRequest fields. No sibling timer."""
    known = official_case_ids()
    if case_id not in known:
        raise OfficialBenchmarkUnavailable(
            f"{case_id!r} is not in {OFFICIAL_MANIFEST} (have {known})"
        )
    injected = _injected_executable()
    exe = injected if injected is not None else Path(DEFAULT_EXECUTABLE)
    if not exe.is_file():
        raise OfficialBenchmarkUnavailable(
            f"official {exe} is missing; build benchmarks/ "
            "(cmake -S benchmarks -B build/benchmarks && cmake --build "
            "build/benchmarks --target pops_benchmark)"
        )
    _require_request(request)
    warmups = int(getattr(request, "warmups", MIN_WARMUPS) or MIN_WARMUPS) if request else MIN_WARMUPS
    samples = (
        int(
            getattr(request, "repetitions", None)
            or getattr(request, "samples", MIN_SAMPLES)
            or MIN_SAMPLES
        )
        if request
        else MIN_SAMPLES
    )
    out = _output_path(request, case_id)
    resolution = _resolution(request)
    command = [
        str(exe),
        f"--case={case_id}",
        f"--output={out}",
        f"--warmups={warmups}",
        f"--repetitions={samples}",
    ]
    if resolution is not None:
        if case_id == "arith_halo":
            command.append(f"--arith-n={resolution}")
        elif case_id == "scalar_mg":
            command.append(f"--mg-n={resolution}")
    completed = subprocess.run(
        command,
        cwd=Path(OFFICIAL_MANIFEST).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OfficialBenchmarkUnavailable(
            f"pops_benchmark --case={case_id} failed rc={completed.returncode}: "
            f"{completed.stderr[-400:]}"
        )
    parsed = _parse_jsonl(out)
    return {
        "case_id": case_id,
        "manifest": str(OFFICIAL_MANIFEST),
        "executable": str(exe),
        "output": str(out),
        "output_dir": str(out.parent),
        "stdout": completed.stdout,
        "returncode": completed.returncode,
        "resolution": [resolution] if resolution is not None else None,
        "n": resolution,
        "timing": {
            "warmups": warmups,
            "samples": samples,
            "repetitions": samples,
        },
        "statistics": parsed["statistics"],
        "validation": parsed["validation"],
        "records": parsed["records"],
    }


def run_mapped_or_refuse(verification_id: str, request=None) -> dict:
    """Run an official case or raise a required-fail official-absence reason."""
    injected = _injected_executable()
    if injected is not None and not injected.is_file():
        raise OfficialBenchmarkUnavailable(f"injected native executable missing: {injected}")
    _require_request(request)
    authority = official_authority(verification_id)
    if authority["status"] != "official":
        raise OfficialBenchmarkUnavailable(authority["reason"])
    result = run_official_for_request(authority["case_id"], request)
    result.update(authority)
    return result
