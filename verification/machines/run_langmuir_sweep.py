"""Langmuir frequency sweep driver (CP-02 + CP-03).

Compiles each Case once, advances through the probe times, and prints
ω_num / E_ω for kL/(2π) = 1, 2, 4, 8 (CP-03) plus the cold k=1 reference.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.langmuir_sweep import (
    analyze_probe,
    cold_point,
    probe_index,
    warm_points,
)


def _load(rel: str):
    return load_sibling_module(ROOT / rel)


def _probe_density(run, point):
    snapshots = run.run_native_series(point.times, point.n_cells, **_cycles_kw(run, point))
    index = probe_index(point.n_cells)
    return np.asarray(snapshots[:, 0, index], dtype=np.float64)


def _cycles_kw(run, point) -> dict:
    if "cycles" in inspect.signature(run.run_native_series).parameters:
        return {"cycles": point.cycles}
    return {}


def main() -> int:
    results = []
    cp02 = _load("verification/cases/euler_poisson/langmuir_cold/run.py")
    cold = cold_point()
    print(f"=== {cold.case_id} cycles={cold.cycles} n={cold.n_cells} ω_ref={cold.omega_ref:.8f} ===")
    samples = _probe_density(cp02, cold)
    row = {"case_id": cold.case_id, "cycles": cold.cycles, "n_cells": cold.n_cells}
    row.update(analyze_probe(cold.times, samples, cold.omega_ref))
    results.append(row)
    print(
        f"  ω_fft={row['omega_fft']:.8f} ω_fit={row['omega_fit']:.8f} "
        f"E_fft={row['error_fft']:.3e} E_fit={row['error_fit']:.3e}"
    )

    cp03 = _load("verification/cases/euler_poisson/langmuir_warm/run.py")
    for point in warm_points():
        print(
            f"=== {point.case_id} cycles={point.cycles} n={point.n_cells} "
            f"ω_ref={point.omega_ref:.8f} ==="
        )
        samples = _probe_density(cp03, point)
        row = {"case_id": point.case_id, "cycles": point.cycles, "n_cells": point.n_cells}
        row.update(analyze_probe(point.times, samples, point.omega_ref))
        results.append(row)
        print(
            f"  ω_fft={row['omega_fft']:.8f} ω_fit={row['omega_fit']:.8f} "
            f"E_fft={row['error_fft']:.3e} E_fit={row['error_fit']:.3e}"
        )

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
