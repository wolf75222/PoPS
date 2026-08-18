"""Phase 5 close-out: native oracles, positivity, polar gates."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors


def _load(rel: str):
    return load_sibling_module(ROOT / rel)


def _euler_primitives_1d(field, gamma: float):
    rho = np.asarray(field[0], dtype=np.float64)
    mom = np.asarray(field[1], dtype=np.float64)
    energy = np.asarray(field[2], dtype=np.float64)
    vel = mom / rho
    pressure = (float(gamma) - 1.0) * (energy - 0.5 * rho * vel * vel)
    return rho, vel, pressure


def _one_1d(name, rel, n, t_end, *, gamma, exact_t=None):
    print(f"=== {name} ===")
    run = _load(rel)
    exact = _load(str(Path(rel).with_name("exact.py")))
    try:
        field = np.asarray(run.run_native(n, t_end=t_end), dtype=np.float64)
    except Exception as exc:
        print(name, "FAILED", type(exc).__name__, exc)
        return {
            "name": name,
            "finite": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    field = np.reshape(field, (3, n))
    rho, _, pressure = _euler_primitives_1d(field, gamma)
    pos_rho = bool(np.all(rho > 0.0))
    pos_p = bool(np.all(pressure > 0.0))
    finite = bool(np.isfinite(field).all())
    row = {
        "name": name,
        "shape": tuple(int(s) for s in field.shape),
        "finite": finite,
        "rho_min": float(np.min(rho)),
        "p_min": float(np.min(pressure)),
        "positive_rho": pos_rho,
        "positive_p": pos_p,
    }
    if exact_t is not None and hasattr(exact, "conserved_1d"):
        centers, width = run.cell_centers(n)
        oracle = exact.conserved_1d(centers, exact_t)
        volumes = np.full(centers.shape, width, dtype=np.float64)
        err = reference_errors(field[0], oracle[0], volumes)
        row["rho_l1"] = float(err.l1)
        row["rho_linf"] = float(err.linf)
        print(name, field.shape, "rho_L1", err.l1, "rho_Linf", err.linf,
              "rho_min", row["rho_min"], "p_min", row["p_min"], finite)
    else:
        print(name, field.shape, "rho_min", row["rho_min"], "p_min", row["p_min"], finite)
    return row


def _one_2d(name, rel, n, t_end, *, gamma=1.4, ge03_oracle=False):
    print(f"=== {name} ===")
    run = _load(rel)
    raw = run.run_native(n, t_end=t_end)
    if isinstance(raw, dict):
        field = run.pack_conserved(raw)
    else:
        field = np.ascontiguousarray(raw, dtype=np.float64)
    rho = field[0]
    finite = bool(np.isfinite(field).all())
    row = {
        "name": name,
        "shape": tuple(int(s) for s in field.shape),
        "finite": finite,
        "rho_min": float(np.min(rho)),
        "positive_rho": bool(np.all(rho > 0.0)),
    }
    if ge03_oracle:
        exact = _load(str(Path(rel).with_name("exact.py")))
        x, y, width = run.cell_centers(n)[:3]
        prim = exact.primitives(x, y, t_end)
        volumes = np.full(rho.shape, float(width) * float(width), dtype=np.float64)
        err = reference_errors(rho, prim["rho"], volumes)
        row["rho_l1"] = float(err.l1)
        row["rho_linf"] = float(err.linf)
        print(name, field.shape, "rho_L1", err.l1, "rho_Linf", err.linf,
              "rho_min", row["rho_min"], finite)
    else:
        print(name, field.shape, "rho_min", row["rho_min"], finite)
    return row


def _polar_gate(name, rel):
    print(f"=== {name} polar gate ===")
    run = _load(rel)
    reason = run.refuse_public_polar_runtime()
    try:
        run.run_native()
        ok = False
        detail = "UNEXPECTED_OK"
    except Exception as exc:
        ok = exc.__class__.__name__ == "NativeUnavailable"
        detail = f"{type(exc).__name__}: {exc}"
    print(name, "refuse", reason, "run_native", detail)
    return {"name": name, "refusal": reason, "gated": ok, "detail": detail}


def main() -> int:
    dim = os.environ.get("POPS_NATIVE_DIM", "")
    print(f"host-dim={dim!r} suite=phase5-close")
    rows = []
    if dim in ("", "1"):
        rows.append(_one_1d(
            "RB-01", "verification/cases/robustness/sod/run.py", 64, 0.2,
            gamma=1.4, exact_t=0.2,
        ))
        rows.append(_one_1d(
            "RB-02", "verification/cases/robustness/double_rarefaction/run.py",
            64, 0.15, gamma=1.4, exact_t=0.15,
        ))
        rows.append(_one_1d(
            "RB-03", "verification/cases/robustness/strong_shock/run.py",
            32, 0.004, gamma=1.4,
        ))
        rows.append(_one_1d(
            "RB-04", "verification/cases/robustness/shu_osher/run.py",
            64, 0.4, gamma=1.4,
        ))
        rows.append(_one_1d(
            "RB-06", "verification/cases/robustness/noh/run.py",
            64, 0.3, gamma=5.0 / 3.0, exact_t=0.3,
        ))
        rows.append(_one_1d(
            "RB-09", "verification/cases/robustness/blast_waves/run.py",
            64, 0.01, gamma=1.4,
        ))
    if dim == "2":
        rows.append(_one_2d(
            "GE-03", "verification/cases/geometry/radial_acoustic/run.py",
            16, 0.02, ge03_oracle=True,
        ))
        rows.append(_one_2d("RB-05", "verification/cases/robustness/sedov/run.py", 16, 0.01))
        rows.append(_one_2d(
            "RB-07", "verification/cases/robustness/liska_implosion/run.py", 16, 0.05
        ))
        rows.append(_one_2d(
            "GE-06", "verification/cases/geometry/diocotron_amr/run.py", 16, 0.02
        ))
    for name, rel in (
        ("GE-01", "verification/cases/geometry/polar_poisson/run.py"),
        ("GE-02", "verification/cases/geometry/solid_rotation/run.py"),
        ("GE-04", "verification/cases/geometry/cartesian_polar_oracle/run.py"),
        ("GE-05", "verification/cases/geometry/polar_axis/run.py"),
    ):
        rows.append(_polar_gate(name, rel))
    failed = [row for row in rows if row.get("gated") is False]
    native = [row for row in rows if "finite" in row]
    if any(not row.get("finite") for row in native):
        print("FAILED native non-finite", [row["name"] for row in native if not row.get("finite")])
        return 1
    if failed:
        print("FAILED polar gate", failed)
        return 1
    print("DONE", len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
