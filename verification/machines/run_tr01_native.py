#!/usr/bin/env python3
"""ROMEO/local driver: TR-01 native smoke (no pytest)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

RUN = ROOT / "verification" / "cases" / "transport" / "advection_sine" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("tr01_run", RUN)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    try:
        field = module.run_native(16, t_end=0.05)
    except module.NativeUnavailable as exc:
        print("UNAVAILABLE", exc)
        return 2
    print("OK shape", field.shape, "min", float(field.min()), "max", float(field.max()))
    if not field.size:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
