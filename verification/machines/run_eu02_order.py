#!/usr/bin/env python3
"""ROMEO Dim2 native order campaign: EU-02 Serial/OpenMP.

Requires Dim2 `_pops.so` + variants.json and POPS_CACHE_DIR on scratch.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors

RESOLUTIONS = (16, 32, 48)
T_END = 0.25
THRESHOLD = 1.5
PERIOD = 10.0


def _density_error(run, exact, conserved, n_cells, t):
    count = int(n_cells)
    width = PERIOD / float(count)
    centers = (np.arange(count, dtype=float) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    volumes = np.full((count, count), width * width)
    primitives = run.conserved_to_primitives(conserved)
    oracle = exact.exact_vortex(x, y, t)
    return reference_errors(primitives["rho"], oracle["rho"], volumes)


def _run_space(label: str, nthreads: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(int(nthreads))
    os.environ.setdefault("POPS_NATIVE_DIM", "2")
    run = load_sibling_module(
        ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "run.py"
    )
    exact = load_sibling_module(
        ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "exact.py"
    )
    errors = []
    rows = []
    for n in RESOLUTIONS:
        conserved = run.run_native(n, t_end=T_END)
        norms = _density_error(run, exact, conserved, n, T_END)
        errors.append(float(norms.l2))
        rows.append({"n": n, "l1": norms.l1, "l2": norms.l2, "linf": norms.linf})
    spacings = [PERIOD / float(n) for n in RESOLUTIONS]
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
    out_dir = Path(
        os.environ.get("POPS_ORDER_OUT", ROOT / "verification" / "out" / "native_order")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    spaces = (
        ("KokkosSerial", 1),
        ("KokkosOpenMP", int(os.environ.get("POPS_ORDER_OMP", "8"))),
    )
    payload = {"schema": "pops.verification.native_order.v1", "cases": ["EU-02"], "results": []}
    failed = False
    for label, threads in spaces:
        print(f"=== {label} OMP_NUM_THREADS={threads} ===")
        result = _run_space(label, threads)
        payload["results"].append(result)
        print(json.dumps(result, indent=2))
        if not result["pass"]:
            failed = True
    (out_dir / "eu02_dim2.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out_dir / "eu02_dim2.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
