#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "test_manifest.toml"
UNAVAILABLE = ROOT / "tests" / "python" / "unavailable_verification_tests.json"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"missing manifest: {MANIFEST.relative_to(ROOT)}")
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_cpp_sources(manifest: dict) -> set[str]:
    sources: set[str] = set()
    for suite in manifest.get("cpp", {}).get("suite", []):
        for source in suite.get("sources", []):
            sources.add(str(source))
    return sources


def manifest_python_files(manifest: dict) -> set[str]:
    files: set[str] = set()
    for suite in manifest.get("python", {}).get("suite", []):
        path = ROOT / suite["path"]
        # A Python suite owns its complete subtree, exactly like
        # scripts/ci_select_tests.py and the manifest coverage fence.
        files.update(str(p.relative_to(ROOT)) for p in path.rglob("test_*.py"))
    return files


def historically_unavailable_python_files() -> set[str]:
    """Test files that the historical catalogue records as unavailable, byte for byte.

    Seven tests were committed without their dependencies before M0 (bcb73f4a). The catalogue
    records them as unavailable, never run; the coverage fence of
    tests/python/architecture/test_manifest_suite_coverage.py pins it and refuses that a suite
    claims them. A file whose bytes no longer match its record is not exempted.
    """
    if not UNAVAILABLE.exists():
        return set()
    catalog = json.loads(UNAVAILABLE.read_text(encoding="utf-8"))
    files: set[str] = set()
    for record in catalog.get("tests", []):
        path = ROOT / record["path"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            files.add(str(record["path"]))
    return files


def actual_cpp_sources() -> set[str]:
    return set(str(p.relative_to(ROOT)) for p in (ROOT / "tests/cpp").rglob("test_*.cpp"))


def actual_python_files() -> set[str]:
    return set(str(p.relative_to(ROOT)) for p in (ROOT / "tests/python").rglob("test_*.py"))


def check_manifest() -> int:
    manifest = load_manifest()
    expected_cpp = actual_cpp_sources()
    expected_py = actual_python_files()
    manifest_cpp = manifest_cpp_sources(manifest)
    manifest_py = manifest_python_files(manifest)
    historical = historically_unavailable_python_files()

    missing = sorted((expected_cpp - manifest_cpp) | (expected_py - manifest_py - historical))
    stale = sorted((manifest_cpp - expected_cpp) | (manifest_py - expected_py))
    claimed = sorted(historical & manifest_py)

    print(f"C++ tests: {len(expected_cpp)} files")
    print(f"Python tests: {len(expected_py)} files")
    print(f"Manifest entries: {len(manifest_cpp)} C++ sources, {len(manifest_py)} Python files")
    print(f"Historically unavailable, never run: {len(historical)} Python files "
          f"({UNAVAILABLE.relative_to(ROOT)})")
    for path in missing:
        print(f"MISSING {path}")
    for path in stale:
        print(f"STALE {path}")
    for path in claimed:
        print(f"CLAIMED {path} (historically unavailable, yet under a suite path)")
    return 1 if missing or stale or claimed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-matrix", action="store_true", help="validate test_manifest.toml")
    args = parser.parse_args()
    if args.check_matrix:
        return check_manifest()
    manifest = load_manifest()
    print(f"cpp_suites={len(manifest.get('cpp', {}).get('suite', []))}")
    print(f"python_suites={len(manifest.get('python', {}).get('suite', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
