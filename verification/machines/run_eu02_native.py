#!/usr/bin/env python3
"""ROMEO/local driver: EU-02 native smoke (Dim2)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

RUN = ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("eu02_run", RUN)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    try:
        conserved = module.run_native(16, t_end=0.05)
    except module.NativeUnavailable as exc:
        print("UNAVAILABLE", exc)
        return 2
    density = conserved["rho"]
    print(
        "OK shape",
        density.shape,
        "rho_min",
        float(density.min()),
        "rho_max",
        float(density.max()),
    )
    if density.size == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
