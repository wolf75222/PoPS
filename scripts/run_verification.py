#!/usr/bin/env python3
"""Plan and optionally execute a verification campaign.

Validates the scientific campaign manifest, refuses more than two nodes, expands
selected cases into single-dimension jobs, and writes output/plan.json.
With ``--execute``, each planned job imports the case ``run.py`` and calls
``run_native`` in-process. Multi-rank MPI is the native PoPS communicator:
launch this script under ``srun``/``mpiexec``. This script does not spawn
ranks and is not a second runner beside the case pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_verification_manifest import (  # noqa: E402
    DEFAULT_MANIFEST,
    VerificationManifestError,
    check_verification_manifest,
)
from verification.pops_verify.campaign import (  # noqa: E402
    ALLOWED_DIMENSIONS,
    MAX_NODES_LIMIT,
    CampaignError,
    expand_jobs,
    resolve_artifact_dim,
)
from verification.pops_verify.case_authoring import load_sibling_module  # noqa: E402

ALLOWED_SUITES = ("pr", "nightly", "weekly", "release", "two_node")


class VerificationRunnerError(RuntimeError):
    pass


def parse_dimensions(raw: str) -> list[int]:
    if not raw or not raw.strip():
        raise VerificationRunnerError(
            "invalid --dimensions (expected comma-separated 1, 2, and/or 3)"
        )
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(part == "" for part in parts):
        raise VerificationRunnerError(
            f"invalid --dimensions {raw!r} (expected comma-separated 1, 2, and/or 3)"
        )
    dimensions: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise VerificationRunnerError(
                f"invalid --dimensions {raw!r} (expected comma-separated 1, 2, and/or 3)"
            ) from exc
        if value not in ALLOWED_DIMENSIONS:
            raise VerificationRunnerError(
                f"invalid --dimensions {raw!r} (expected comma-separated 1, 2, and/or 3)"
            )
        if value not in dimensions:
            dimensions.append(value)
    return dimensions


def select_cases(manifest: dict, suite: str, dimensions: list[int]) -> list[dict]:
    requested = set(dimensions)
    selected = []
    for case in manifest.get("case", []):
        if suite not in case.get("suites", []):
            continue
        native = set(case.get("native_dimensions", []))
        if native.intersection(requested):
            selected.append(case)
    return selected


def write_plan(output: Path, plan: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )


def execute_jobs(jobs, cases: list[dict], output: Path) -> list[dict]:
    """Run each planned job's public ``run_native`` in this process."""
    import os
    import traceback

    by_id = {case["id"]: case for case in cases}
    results: list[dict] = []
    for job in jobs:
        case = by_id.get(job.case_id)
        if case is None:
            results.append(
                {
                    "case_id": job.case_id,
                    "pops_native_dim": job.pops_native_dim,
                    "status": "missing",
                }
            )
            continue
        path = REPO_ROOT / case["path"]
        env_dim = os.environ.get("POPS_NATIVE_DIM")
        os.environ["POPS_NATIVE_DIM"] = str(job.pops_native_dim)
        record = {
            "case_id": job.case_id,
            "pops_native_dim": job.pops_native_dim,
            "path": str(path),
        }
        try:
            module = load_sibling_module(path)
            runner = getattr(module, "run_native", None)
            if not callable(runner):
                record["status"] = "no_run_native"
            else:
                runner()
                record["status"] = "ok"
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "NativeUnavailable":
                record["status"] = "skipped"
                record["reason"] = str(exc)
            else:
                record["status"] = "failed"
                record["reason"] = f"{name}: {exc}"
                record["traceback"] = traceback.format_exc()
        finally:
            if env_dim is None:
                os.environ.pop("POPS_NATIVE_DIM", None)
            else:
                os.environ["POPS_NATIVE_DIM"] = env_dim
        results.append(record)
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--dimensions", required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pops-native-dim", type=int, choices=ALLOWED_DIMENSIONS)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run each planned job's run_native in-process (no rank launcher)",
    )
    args = parser.parse_args(argv)

    try:
        instance = check_verification_manifest(args.manifest)

        if args.max_nodes > MAX_NODES_LIMIT:
            raise VerificationRunnerError(
                f"--max-nodes {args.max_nodes} exceeds the two-node limit"
            )
        if args.suite not in ALLOWED_SUITES:
            raise VerificationRunnerError(
                f"unknown --suite {args.suite!r} (expected {', '.join(ALLOWED_SUITES)})"
            )
        dimensions = parse_dimensions(args.dimensions)
        if args.max_nodes < 1:
            raise VerificationRunnerError("--max-nodes must be >= 1")

        artifact_dim = resolve_artifact_dim(args.pops_native_dim)
        cases = select_cases(instance, args.suite, dimensions)
        jobs = expand_jobs(cases, dimensions, artifact_dim=artifact_dim)
        write_plan(
            args.output,
            {
                "suite": args.suite,
                "dimensions": dimensions,
                "max_nodes": args.max_nodes,
                "manifest": str(args.manifest.resolve()),
                "cases": cases,
                "jobs": [
                    {
                        "case_id": job.case_id,
                        "pops_native_dim": job.pops_native_dim,
                    }
                    for job in jobs
                ],
            },
        )
        if args.execute:
            results = execute_jobs(jobs, cases, args.output)
            failed = sum(1 for row in results if row.get("status") == "failed")
            print(f"planned {len(cases)} cases")
            print(f"executed {len(results)} jobs ({failed} failed)")
            return 1 if failed else 0
    except (VerificationRunnerError, VerificationManifestError, CampaignError) as exc:
        print(f"VERIFICATION: FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"planned {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
