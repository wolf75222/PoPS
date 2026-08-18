#!/usr/bin/env python3
"""Fail-closed Phase 8 visual catalog and optional campaign gate.

This script does not run the solver. It validates the §40.6.11 catalog
contracts and, when given a campaign directory, the visual manifests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from verification.pops_verify.visualization.catalog import (
    SCIENTIFIC_CASE_IDS,
    visual_contract_for,
)
from verification.pops_verify.visualization.gates import (
    VisualsGateError,
    check_release_completeness,
)

SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"


def check_catalog() -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for case_id in SCIENTIFIC_CASE_IDS:
        validator.validate(visual_contract_for(case_id))
    print(f"ok: {len(SCIENTIFIC_CASE_IDS)} visual contracts")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, help="Optional campaign output directory")
    parser.add_argument("--suite", default="release")
    parser.add_argument("--executed-only", action="store_true")
    args = parser.parse_args(argv)
    check_catalog()
    if args.campaign is not None:
        try:
            report = check_release_completeness(
                args.campaign,
                suite=args.suite,
                executed_only=args.executed_only,
            )
        except VisualsGateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"ok: checked {report['cells_checked']} executed cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
