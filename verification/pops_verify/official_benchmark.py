"""Invoke the official ``benchmarks/manifest.toml`` harness.

This is not a second benchmark stack. Timed PF work belongs to
``arith_halo``, ``scalar_mg``, and the ADC campaigns already listed there.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MANIFEST = REPO_ROOT / "benchmarks" / "manifest.toml"
DEFAULT_EXECUTABLE = REPO_ROOT / "build" / "benchmarks" / "bin" / "pops_benchmark"
OFFICIAL_CASE_IDS = ("arith_halo", "scalar_mg")
OFFICIAL_CAMPAIGNS = (
    "adc700_program_cutover",
    "adc757_heterogeneous_numerics",
)


class OfficialBenchmarkUnavailable(RuntimeError):
    """Raised when the official harness cannot run."""


def load_official_manifest(path: Path = OFFICIAL_MANIFEST) -> dict:
    """Return the official benchmark manifest."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def official_case_ids(path: Path = OFFICIAL_MANIFEST) -> tuple[str, ...]:
    """Return ``[[case]]`` ids from the official benchmark manifest."""
    manifest = load_official_manifest(path)
    return tuple(str(case["id"]) for case in manifest.get("case", []) if "id" in case)


def official_campaign_ids(path: Path = OFFICIAL_MANIFEST) -> tuple[str, ...]:
    """Return campaign keys from the official benchmark manifest."""
    manifest = load_official_manifest(path)
    campaigns = manifest.get("campaigns", {})
    if not isinstance(campaigns, dict):
        return ()
    return tuple(str(name) for name in campaigns)


def refuse_unofficial_pf(case_id: str) -> str:
    """Return why a verification/performance wrapper is not an official bench."""
    cases = ", ".join(official_case_ids())
    campaigns = ", ".join(official_campaign_ids() or OFFICIAL_CAMPAIGNS)
    return (
        f"{case_id} is not an official timed case; use benchmarks/manifest.toml "
        f"cases [{cases}] or campaigns [{campaigns}]"
    )


def run_official_benchmark(
    case_id: str,
    *,
    executable: Path = DEFAULT_EXECUTABLE,
    output: Path | None = None,
) -> dict:
    """Run ``pops_benchmark --case=<id>`` from the official harness."""
    known = official_case_ids()
    if case_id not in known:
        raise OfficialBenchmarkUnavailable(
            f"{case_id!r} is not in {OFFICIAL_MANIFEST} (have {known})"
        )
    exe = Path(executable)
    if not exe.is_file():
        raise OfficialBenchmarkUnavailable(
            f"official {exe} is missing; build benchmarks/ "
            "(cmake -S benchmarks -B build/benchmarks && cmake --build "
            "build/benchmarks --target pops_benchmark)"
        )
    out = Path(output) if output is not None else Path("benchmarks.jsonl")
    completed = subprocess.run(
        [str(exe), f"--case={case_id}", f"--output={out}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OfficialBenchmarkUnavailable(
            f"pops_benchmark --case={case_id} failed rc={completed.returncode}: "
            f"{completed.stderr[-400:]}"
        )
    return {
        "case_id": case_id,
        "manifest": str(OFFICIAL_MANIFEST),
        "executable": str(exe),
        "output": str(out),
        "stdout": completed.stdout,
        "returncode": completed.returncode,
    }


def dumps_json(payload: dict) -> str:
    """Stable JSON for campaign logs."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "arith_halo"
    print(dumps_json(run_official_benchmark(target)))