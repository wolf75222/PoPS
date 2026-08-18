"""Write truthful Phase 8 status/visual_data from an EvidenceBundle only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError

ALLOWED_VERDICTS = frozenset({"pass", "fail", "not-run"})


def write_run_status(
    run_dir: Path,
    *,
    case_id: str,
    run_id: str,
    verdict: str,
    scientific_pass: bool,
) -> None:
    """Write status.json. scientific/live equal scientific_pass, never bundle auth."""
    if verdict not in ALLOWED_VERDICTS:
        raise EvidenceError(f"illegal status verdict {verdict!r}")
    if scientific_pass and verdict != "pass":
        raise EvidenceError("scientific_pass requires verdict pass")
    if verdict == "pass" and not scientific_pass:
        raise EvidenceError("verdict pass requires scientific_pass")
    run_dir.mkdir(parents=True, exist_ok=True)
    passed = bool(scientific_pass)
    payload = {
        "case_id": case_id,
        "run_id": run_id,
        "verdict": verdict,
        "data_kind": "campaign",
        "scientific": passed,
        "live": passed,
    }
    (run_dir / "status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _result_series(bundle: EvidenceBundle) -> tuple[list[float], list[float]]:
    if not bundle.records:
        raise EvidenceError("no EvidenceBundle records for live visual_data")
    result = bundle.records[-1]["result"]
    y = [float(value) for value in np.ravel(np.asarray(result, dtype=np.float64))]
    x = [float(index) for index in range(len(y))]
    return x, y


def _convergence_from_analysis(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    orders = (analysis or {}).get("orders") if isinstance(analysis, dict) else None
    if not isinstance(orders, list) or not orders:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for row in orders:
        if not isinstance(row, dict):
            continue
        spacing = row.get("spacing") or row.get("h") or row.get("n")
        error = row.get("error") or row.get("linf") or row.get("l2")
        if spacing is None or error is None:
            continue
        xs.append(float(spacing))
        ys.append(float(error))
    if len(xs) < 2:
        return None
    return {
        "figure_id": "spatial_convergence",
        "kind": "spatial_convergence",
        "data_kind": "campaign",
        "units": {"x": "1/h", "y": "L2 error"},
        "variables": ["scalar"],
        "series": [{"name": "L2", "x": xs, "y": ys, "unit": "1"}],
        "times": [1.0],
        "step_numbers": [0],
        "title": "spatial convergence from EvidenceBundle analysis",
    }


def _pin_identity_files(run_dir: Path) -> None:
    """Copy bundle identity files to the series root for Phase 8 digest pins."""
    series_file = run_dir / "series.json"
    if not series_file.is_file():
        raise EvidenceError("series.json missing for live identity pins")
    series = json.loads(series_file.read_text(encoding="utf-8"))
    jobs = [str(name) for name in series.get("jobs") or []]
    if not jobs:
        raise EvidenceError("series.json has no jobs")
    job_dir = run_dir / jobs[-1]
    for name in ("resolved_case.json", "native_artifact.json"):
        src = job_dir / name
        dest = run_dir / name
        if src.is_file():
            dest.write_bytes(src.read_bytes())
    program = job_dir / "program.bin"
    if not program.is_file():
        raise EvidenceError("program.bin missing from EvidenceBundle job")
    payload = {
        "job": jobs[-1],
        "path": f"{jobs[-1]}/program.bin",
        "sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
    }
    (run_dir / "program.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_live_visual_data(
    run_dir: Path,
    bundle: EvidenceBundle,
    *,
    case_id: str,
    analysis: dict[str, Any] | None = None,
) -> None:
    """Emit visual_data from authenticated bundle records. Never reread disk arrays."""
    if not isinstance(bundle, EvidenceBundle):
        raise EvidenceError("live visual_data requires an authenticated EvidenceBundle")
    x, y = _result_series(bundle)
    visual_dir = run_dir / "analysis" / "visual_data"
    visual_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "figure_id": "reference_profile",
        "kind": "reference_profile",
        "data_kind": "campaign",
        "units": {"x": "cell", "y": "scalar"},
        "variables": ["scalar"],
        "series": [{"name": "numerical", "x": x, "y": y, "unit": "1"}],
        "times": [1.0],
        "step_numbers": [0],
        "title": f"{case_id} numerical profile from EvidenceBundle",
    }
    report = {
        "figure_id": "report_figure",
        "kind": "report_figure",
        "data_kind": "campaign",
        "units": {"x": "cell", "y": "scalar"},
        "variables": ["scalar"],
        "series": [{"name": "numerical", "x": x, "y": y, "unit": "1"}],
        "times": [1.0],
        "step_numbers": [0],
        "title": f"{case_id} report figure from EvidenceBundle",
    }
    payloads = [profile, report]
    convergence = _convergence_from_analysis(analysis)
    if convergence is not None:
        payloads.append(convergence)
    for payload in payloads:
        dest = visual_dir / f"{payload['figure_id']}.json"
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_live_run_visuals(
    run_dir: Path,
    *,
    case_id: str,
    run_id: str,
    scientific_pass: bool,
    verdict: str,
    analysis: dict[str, Any] | None = None,
    bundle: EvidenceBundle | None = None,
) -> None:
    """Status plus optional visual_data. Bundle auth is not a scientific pass."""
    write_run_status(
        run_dir,
        case_id=case_id,
        run_id=run_id,
        verdict=verdict,
        scientific_pass=scientific_pass,
    )
    if bundle is None:
        return
    _pin_identity_files(run_dir)
    write_live_visual_data(run_dir, bundle, case_id=case_id, analysis=analysis)
