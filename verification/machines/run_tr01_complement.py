#!/usr/bin/env python3
"""Run the TR-01 complement catalog on one exact-rank native leaf."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _installed_native_root() -> Path | None:
    try:
        import pops
    except ImportError:
        return None
    for item in getattr(pops, "__path__", ()):
        candidate = Path(item).resolve() / "_native"
        if (candidate / "variants.json").is_file():
            return candidate
    return None


_INSTALLED_NATIVE = _installed_native_root()
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
if _INSTALLED_NATIVE is not None and not os.environ.get("POPS_NATIVE_VARIANTS_ROOT"):
    os.environ["POPS_NATIVE_VARIANTS_ROOT"] = str(_INSTALLED_NATIVE)

COMPLEMENT = (
    ROOT / "verification" / "cases" / "transport" / "advection_sine" / "complement.py"
)
ANALYZE = ROOT / "verification" / "cases" / "transport" / "advection_sine" / "analyze.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "verification" / "tr01-complement",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    complement = _load(COMPLEMENT, "tr01_complement")
    analyze = _load(ANALYZE, "tr01_analyze")
    output = Path(args.out) / f"dim{args.dim}"
    payload = complement.run_campaign(
        dim=args.dim, output_dir=output, smoke=args.smoke
    )
    writer = getattr(analyze, "write_complement_markdown", None)
    if writer is not None:
        writer(output, payload)
    summary = payload["summary"]
    print(
        f"done dim={args.dim} ok={summary['n_ok']}/{summary['n_planned']} "
        f"order_ge_1.8={summary['spatial_pairs_ge_1_8']}/{summary['spatial_pairs']}",
        flush=True,
    )
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
