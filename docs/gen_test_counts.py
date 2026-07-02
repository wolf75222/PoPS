#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "test_manifest.toml"


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
        files.update(str(p.relative_to(ROOT)) for p in path.glob("test_*.py"))
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

    missing = sorted((expected_cpp - manifest_cpp) | (expected_py - manifest_py))
    stale = sorted((manifest_cpp - expected_cpp) | (manifest_py - expected_py))

    print(f"C++ tests: {len(expected_cpp)} files")
    print(f"Python tests: {len(expected_py)} files")
    print(f"Manifest entries: {len(manifest_cpp)} C++ sources, {len(manifest_py)} Python files")
    for path in missing:
        print(f"MISSING {path}")
    for path in stale:
        print(f"STALE {path}")
    return 1 if missing or stale else 0


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
