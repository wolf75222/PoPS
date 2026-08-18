#!/usr/bin/env python3
"""Render Phase 8 verification figures from an existing run.

This script does not run the solver. It reads metrics.json, provenance.json,
status.json, and analysis/visual_data, then writes analysis/figures and
visual_manifest.json. Missing scientific data fails closed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.pops_verify.visualization.catalog import P0_STATIC_CASES
from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.fixtures import write_fixture_run
from verification.pops_verify.visualization.gallery import render_release_gallery
from verification.pops_verify.visualization.render import render_run


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(item.strip() for item in raw.split(",") if item.strip())
    allowed = {"png", "pdf", "svg", "mp4", "gif"}
    unknown = [item for item in formats if item not in allowed]
    if unknown:
        raise SystemExit(f"unsupported formats: {', '.join(unknown)}")
    return formats or ("svg", "png", "pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="build/verification/<case-id>/<run-id>")
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "verification" / "manifest.toml",
        help="Campaign manifest; visual requirements come from the Phase 8 catalog.",
    )
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--suite", default="pr", choices=["pr", "nightly", "weekly", "release", "two_node"])
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument(
        "--examples",
        type=Path,
        help="Write labeled fixture plots under this gitignored directory (use build/verification/).",
    )
    parser.add_argument("--gallery", type=Path, help="Campaign report directory for §40.7 dashboards.")
    args = parser.parse_args(argv)
    formats = _parse_formats(args.formats)
    try:
        if args.examples is not None:
            for case_id, dimension in (("TR-01", 1), ("TR-01", 2), ("PO-01", 1), ("PO-01", 2)):
                run = write_fixture_run(args.examples, case_id, dimension=dimension)
                render_run(run, suite="pr", formats=formats, strict=args.strict)
            for case_id in P0_STATIC_CASES:
                if case_id in {"TR-01", "PO-01"}:
                    continue
                from verification.pops_verify.visualization.catalog import catalog_entry

                codes = catalog_entry(case_id).dimension_codes
                dimension = 1 + next(i for i, code in enumerate(codes) if code == "R")
                run = write_fixture_run(args.examples, case_id, dimension=dimension)
                render_run(run, suite="pr", formats=formats, strict=args.strict)
            return 0
        if args.gallery is not None:
            render_release_gallery(args.gallery, formats=tuple(fmt for fmt in formats if fmt in {"svg", "png", "pdf"}))
            return 0
        if args.run is None:
            parser.error("one of --run, --examples, or --gallery is required")
        render_run(args.run, suite=args.suite, formats=formats, strict=args.strict)
        return 0
    except VisualsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
