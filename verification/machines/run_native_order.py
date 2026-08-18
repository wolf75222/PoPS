#!/usr/bin/env python3
"""ROMEO native order campaign: TR-01 Dim1 Serial/OpenMP resolutions.

Requires a Dim1 `_pops.so` + variants.json and POPS_CACHE_DIR on scratch
(home quota cannot hold the DSL cache).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors

RESOLUTIONS = (16, 32, 64, 128, 256)
T_END = 0.25
THRESHOLD = 1.8


def _run_space(label: str, nthreads: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(int(nthreads))
    os.environ.setdefault("POPS_NATIVE_DIM", "1")
    run = load_sibling_module(
        ROOT / "verification" / "cases" / "transport" / "advection_sine" / "run.py"
    )
    exact = load_sibling_module(
        ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
    )
    errors = []
    rows = []
    for n in RESOLUTIONS:
        field = run.run_native(n, t_end=T_END)
        centers, volumes = exact.uniform_cell_centers(n)
        oracle = exact.exact_sine(centers, T_END)
        norms = reference_errors(field, oracle, volumes)
        errors.append(float(norms.l2))
        rows.append({"n": n, "l1": norms.l1, "l2": norms.l2, "linf": norms.linf})
    spacings = [1.0 / float(n) for n in RESOLUTIONS]
    orders = [float(v) for v in observed_order(errors, spacings)]
    return {
        "space": label,
        "omp_num_threads": int(nthreads),
        "t_end": T_END,
        "resolutions": list(RESOLUTIONS),
        "spacings": spacings,
        "errors": rows,
        "observed_orders": orders,
        "threshold": THRESHOLD,
        "pass": bool(orders) and float(orders[-1]) >= THRESHOLD,
    }


def main() -> int:
    out_dir = Path(os.environ.get("POPS_ORDER_OUT", ROOT / "verification" / "out" / "native_order"))
    out_dir.mkdir(parents=True, exist_ok=True)
    spaces = (("KokkosSerial", 1), ("KokkosOpenMP", int(os.environ.get("POPS_ORDER_OMP", "8"))))
    payload = {"schema": "pops.verification.native_order.v1", "cases": ["TR-01"], "results": []}
    failed = False
    for label, threads in spaces:
        print(f"=== {label} OMP_NUM_THREADS={threads} ===")
        result = _run_space(label, threads)
        payload["results"].append(result)
        print(json.dumps(result, indent=2))
        if not result["pass"]:
            failed = True
    (out_dir / "tr01_dim1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out_dir / "tr01_dim1.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
