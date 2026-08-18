"""Phase 6 close-out: live IF-01..08/10 plus IF-09 N/A."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module


def _load(rel: str):
    return load_sibling_module(ROOT / rel)


def main() -> int:
    dim = os.environ.get("POPS_NATIVE_DIM", "1")
    print(f"host-dim={dim!r} suite=phase6-final")
    failed = False

    def _ok(name, fn):
        nonlocal failed
        print(f"=== {name} ===")
        try:
            result = fn()
            print(name, "OK", result)
            return result
        except Exception as exc:
            print(name, "FAILED", type(exc).__name__, exc)
            failed = True
            return None

    if dim == "1":
        _ok("IF-01", lambda: _load("verification/cases/infrastructure/mpi_invariance/run.py").run_native(n_cells=32, t_end=0.25))
        _ok("IF-02", lambda: _load("verification/cases/infrastructure/thread_invariance/run.py").run_native_threads((1, 8), n_cells=32, t_end=0.25))
        _ok("IF-03", lambda: list(_load("verification/cases/infrastructure/space_parity/run.py").run_native_spaces(32, t_end=0.25)))
        _ok("IF-04", lambda: _load("verification/cases/infrastructure/checkpoint_restart/run.py").run_native(n_cells=16, t=0.1)["linf"])
        _ok("IF-05", lambda: _load("verification/cases/infrastructure/output_cadence/run.py").run_native(16, t_end=0.1)["linf"])
        _ok("IF-06", lambda: _load("verification/cases/infrastructure/deterministic_reductions/run.py").run_native(16, t_end=0.1)["linf"])
        _ok("IF-07", lambda: _load("verification/cases/infrastructure/path_parity/run.py").run_native(16, t_end=0.1)["path"])
        if08 = _load("verification/cases/infrastructure/native_dim_guard/run.py")
        try:
            if08.present_dim2_case()
            print("IF-08 Dim2-under-Dim1 UNEXPECTED")
            failed = True
        except if08.NativeUnavailable as exc:
            print("IF-08 Dim2-under-Dim1", exc)
        _ok("IF-08 Dim1", lambda: if08.run_native_dim1(16, t_end=0.1).shape)
        print("=== IF-09 ===")
        if09 = _load("verification/cases/infrastructure/float_precision/run.py")
        try:
            if09.run_native()
            print("IF-09 UNEXPECTED_OK")
            failed = True
        except if09.NativeUnavailable as exc:
            print("IF-09 N/A", exc)
        _ok("IF-10", lambda: _load("verification/cases/infrastructure/hdf5_reread/run.py").run_native(8, t_end=0.05)["path"])
    elif dim == "2":
        if08 = _load("verification/cases/infrastructure/native_dim_guard/run.py")
        try:
            if08.present_dim1_case()
            print("IF-08 Dim1-under-Dim2 UNEXPECTED")
            failed = True
        except if08.NativeUnavailable as exc:
            print("IF-08 Dim1-under-Dim2", exc)
        _ok("IF-08 Dim2", lambda: getattr(if08.run_native(8, t_end=0.01), "shape", "ok"))

    print("FAILED" if failed else "DONE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
