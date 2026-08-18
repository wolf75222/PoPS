"""Phase 7 campaign: compile once, warmup, timed samples, thread strong scale.

GPU is reported unavailable. Two-node rank scaling is a separate SLURM job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.tr01_runtime import advance, prepare


def _median(values):
    return float(statistics.median(values)) if values else None


def time_tr01(n_cells: int, t_end: float, *, warmups: int, samples: int) -> dict:
    prepared = prepare(n_cells)
    for _ in range(int(warmups)):
        advance(prepared, t_end)
    elapsed = []
    for _ in range(int(samples)):
        started = time.perf_counter()
        field = advance(prepared, t_end)
        elapsed.append(time.perf_counter() - started)
    median = _median(elapsed)
    cells = float(n_cells)
    return {
        "n_cells": int(n_cells),
        "t_end": float(t_end),
        "warmups": int(warmups),
        "samples": int(samples),
        "elapsed_s": elapsed,
        "median_s": median,
        "cells_per_second": None if not median else cells / median,
        "finite": bool(__import__("numpy").isfinite(field).all()),
    }


def main() -> int:
    print("suite=phase7-campaign")
    report = {
        "schema": "pops.verification.phase7.v1",
        "gpu": "no public CUDA space",
        "max_nodes": 2,
        "results": {},
    }
    failed = False
    try:
        report["results"]["TR-01_n32"] = time_tr01(32, 0.25, warmups=2, samples=5)
        print("TR-01 n=32 median_s", report["results"]["TR-01_n32"]["median_s"])
        previous = os.environ.get("OMP_NUM_THREADS")
        strong = {}
        for threads in (1, 8):
            os.environ["OMP_NUM_THREADS"] = str(threads)
            strong[str(threads)] = time_tr01(32, 0.25, warmups=1, samples=3)
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous
        report["results"]["TR-01_thread_strong"] = strong
        t1 = strong["1"]["median_s"]
        t8 = strong["8"]["median_s"]
        report["results"]["thread_speedup_1_to_8"] = None if not t1 or not t8 else t1 / t8
        print("thread strong 1->8", report["results"]["thread_speedup_1_to_8"])
    except Exception as exc:
        print("FAILED", type(exc).__name__, exc)
        failed = True
    out = Path(os.environ.get("POPS_PHASE7_OUT", ROOT / "verification" / "out"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase7_campaign.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", path)
    print("FAILED" if failed else "DONE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
