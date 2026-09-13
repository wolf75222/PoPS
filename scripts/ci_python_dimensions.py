"""Run selected Python files in processes with their declared native specialization.

The default CI contract is Dim2. Exceptions are explicit data, never inferred from
source text or whichever native artifact happens to be present. No file is excluded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/python/native_dimensions.json"


def partition(paths: list[str], contract: dict) -> dict[int, list[str]]:
    default, overrides = contract["default"], contract["files"]
    if any(type(dim) is not int or dim not in (1, 2, 3)
           for dim in (default, *overrides.values())):
        raise ValueError("native dimensions must be explicit integers 1, 2 or 3")
    if len(paths) != len(set(paths)):
        raise ValueError("selected Python files must not be duplicated")
    groups: dict[int, list[str]] = {}
    for path in paths:
        groups.setdefault(overrides.get(path, default), []).append(path)
    return dict(sorted(groups.items()))


def run_groups(groups: dict[int, list[str]], packages: Path, timings: Path) -> int:
    status = 0
    for dimension, paths in groups.items():
        print(f"Running {len(paths)} selected Python files with native Dim{dimension}", flush=True)
        # Each downloaded package has its own authenticated variants.json; do not merge
        # packages or inherit another process's selected extension/search path.
        package = (packages / f"dim{dimension}").resolve()
        if not (package / "pops/_native/variants.json").is_file():
            raise FileNotFoundError(f"missing declared Dim{dimension} native package: {package}")
        receipts = (timings / f"dim{dimension}").resolve()
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / "selected.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(POPS_NATIVE_DIM=str(dimension), PYTHONNOUSERSITE="1",
                           PYTHONPATH=os.pathsep.join((str(ROOT / "scripts"), str(package))),
                           POPS_CI_PYTEST_TIMINGS_DIR=str(receipts))
        environment.pop("POPS_NATIVE_VARIANTS_ROOT", None)
        verified = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_installed_native.py"),
             "--expect-dim", str(dimension), "--expect-serial"],
            cwd=ROOT, env=environment, check=False)
        if verified.returncode:
            status = status or verified.returncode
            continue
        result = subprocess.run(
            [sys.executable, "-u", "-m", "pytest", "-v", "-ra", "--durations=20",
             "-p", "ci_pytest_timings", f"--junitxml={receipts / 'junit.xml'}", *paths],
            cwd=ROOT, env=environment, check=False)
        status = status or result.returncode
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--packages-root", type=Path)
    parser.add_argument("--timings-dir", type=Path)
    args = parser.parse_args()
    paths = [line.strip() for line in args.selected_file.read_text().splitlines() if line.strip()]
    groups = partition(paths, json.loads(CONTRACT.read_text()))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            for dimension in (1, 2, 3):
                output.write(f"dim{dimension}_count={len(groups.get(dimension, []))}\n")
        return 0
    if args.packages_root is None or args.timings_dir is None:
        parser.error("execution requires --packages-root and --timings-dir")
    return run_groups(groups, args.packages_root, args.timings_dir)


if __name__ == "__main__":
    raise SystemExit(main())
