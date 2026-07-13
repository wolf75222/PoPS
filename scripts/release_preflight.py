#!/usr/bin/env python3
"""Fail-closed PoPS release preflight.

Development mode checks every static version/generator contract. ``--release`` additionally requires
an exact tag, installed native package, clean checkout and authenticated evidence for every expensive
build/example/IO gate. The evidence is machine output from the final gate, never a boolean CLI escape.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "python" / "pops" / "_generated_release_contract.py"
REQUIRED_GATES = (
    "official_build",
    "doctor",
    "codesign",
    "native_conformance",
    "python_conformance",
    "examples",
    "artifact_reopen",
    "strict_restart",
    "documentation",
    "generated_products",
)


class PreflightError(RuntimeError):
    pass


def _generated() -> Any:
    spec = importlib.util.spec_from_file_location("_pops_release_contract", GENERATED)
    if spec is None or spec.loader is None:
        raise PreflightError("cannot load generated release contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-20:])
        raise PreflightError("command failed (%s):\n%s" % (" ".join(args), tail))
    return result.stdout.strip()


def _project_version() -> str:
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    found = re.findall(r"(?m)^\s*VERSION\s+(\d+\.\d+\.\d+)\b", text)
    if len(found) != 1:
        raise PreflightError("CMake must contain exactly one project VERSION")
    if "PROJECT_VERSION_MAJOR EQUAL 0" not in text or "SameMinorVersion" not in text:
        raise PreflightError("CMake pre-1.0 compatibility policy is not fail-closed")
    return found[0]


def _static_contract(contract: Any) -> list[str]:
    package_version = _project_version()
    if contract.PACKAGE_VERSION != package_version or package_version == "unknown":
        raise PreflightError("generated/package CMake versions disagree")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"].get("dynamic") != ["version"]:
        raise PreflightError("wheel version must remain dynamic from CMakeLists.txt")
    provider = project["tool"]["scikit-build"]["metadata"]["version"]
    if provider.get("input") != "CMakeLists.txt" or not re.search(
            provider["regex"], (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")):
        raise PreflightError("wheel version provider does not resolve the CMake version")

    source = json.loads((ROOT / "schemas" / "release_contract.v1.json").read_text())
    catalog = json.loads((ROOT / "schemas" / "component_catalog.v2.json").read_text())
    exact = {
        "component_catalog_schema_version": catalog["catalog_schema_version"],
        "component_manifest_schema_version": catalog["component_manifest_schema_version"],
        "component_registry_version": catalog["route_registry_version"],
        "capability_vocabulary_version": catalog["capability_vocabulary_version"],
    }
    for name, value in exact.items():
        if source[name] != value:
            raise PreflightError("release contract %s drifted from component catalog" % name)
    native = (ROOT / "include" / "pops" / "runtime" / "module_capabilities.hpp").read_text()
    match = re.search(r"kAbiVersion\s*=\s*(\d+)", native)
    if match is None or int(match.group(1)) != source["native_abi_version"]:
        raise PreflightError("release native ABI drifted from module capability ABI")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    if 'POPS_KOKKOS_FETCH_VERSION "' + source["supported_matrix"]["kokkos"]["version"] + '"' \
            not in cmake:
        raise PreflightError("supported Kokkos version drifted from CMake")
    if contract.RELEASE_CONTRACT_SHA256 != hashlib.sha256(json.dumps(
            {"package_version": package_version, **source}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode()).hexdigest():
        raise PreflightError("generated release contract digest is not canonical")
    _run(sys.executable, "scripts/generate_release_contract.py", "--check")
    _run(sys.executable, "scripts/generate_component_catalog.py", "--check")
    return ["version", "wheel_metadata", "schemas", "abi", "matrix", "generated"]


def _tag_contract(version: str, tag: str) -> None:
    if tag != "v" + version:
        raise PreflightError("release tag %r must equal v%s" % (tag, version))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(r"(?m)^## \[%s\](?:\s+-\s+\d{4}-\d{2}-\d{2})?\s*$" % re.escape(version),
                 changelog) is None:
        raise PreflightError("CHANGELOG has no exact release section for %s" % version)


def _installed_contract(contract: Any) -> None:
    import pops
    from pops import _pops

    if pops.__version__ != contract.PACKAGE_VERSION or _pops.__version__ != contract.PACKAGE_VERSION:
        raise PreflightError("installed Python/native/package versions disagree")
    if _pops.__abi_version__ != contract.NATIVE_ABI_VERSION:
        raise PreflightError("installed native ABI disagrees with release contract")
    if _pops.__release_contract_sha256__ != contract.RELEASE_CONTRACT_SHA256:
        raise PreflightError("installed native release digest disagrees with Python")


def _evidence(path: Path, contract: Any, commit: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "commit_sha", "package_version", "contract_sha256", "gates"}
    if not isinstance(payload, dict) or set(payload) != expected or payload["schema_version"] != 1:
        raise PreflightError("release evidence has an unknown or incomplete schema")
    if payload["commit_sha"] != commit or payload["package_version"] != contract.PACKAGE_VERSION \
            or payload["contract_sha256"] != contract.RELEASE_CONTRACT_SHA256:
        raise PreflightError("release evidence belongs to another build")
    gates = payload["gates"]
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES):
        raise PreflightError("release evidence gate set must be exactly %s" % (REQUIRED_GATES,))
    for name in REQUIRED_GATES:
        row = gates[name]
        if not isinstance(row, dict) or set(row) != {"status", "command", "evidence"}:
            raise PreflightError("release evidence %s has an invalid row" % name)
        if row["status"] != "passed" or not row["command"] or not row["evidence"]:
            raise PreflightError("release gate %s did not produce passing evidence" % name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    try:
        contract = _generated()
        checks = _static_contract(contract)
        if args.release:
            if not args.tag or not args.installed or args.evidence is None:
                raise PreflightError("--release requires --tag, --installed and --evidence")
            _tag_contract(contract.PACKAGE_VERSION, args.tag)
            commit = _run("git", "rev-parse", "HEAD")
            if _run("git", "status", "--porcelain"):
                raise PreflightError("release checkout is dirty")
            _installed_contract(contract)
            _evidence(args.evidence, contract, commit)
            checks.extend(("tag", "changelog", "installed", "evidence", "clean"))
        elif args.tag:
            _tag_contract(contract.PACKAGE_VERSION, args.tag)
            checks.extend(("tag", "changelog"))
        print(json.dumps({"status": "passed", "package_version": contract.PACKAGE_VERSION,
                          "contract_sha256": contract.RELEASE_CONTRACT_SHA256,
                          "checks": checks}, sort_keys=True))
        return 0
    except (PreflightError, OSError, ValueError, KeyError) as exc:
        print("release preflight failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
