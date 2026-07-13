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

from final_release_contract import (
    FINAL_EXAMPLES,
    PYTHON_REQUIRED_SELECTION,
    REQUIRED_PROOF_MARKERS,
    REQUIRED_RELEASE_GATES,
    require_source_contract,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "python" / "pops" / "_generated_release_contract.py"
REQUIRED_GATES = REQUIRED_RELEASE_GATES
EVIDENCE_SCHEMA_VERSION = 2


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
    require_source_contract(ROOT)
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
    return ["version", "wheel_metadata", "schemas", "abi", "matrix", "generated",
            "final_specification", "final_examples"]


def _tag_contract(version: str, tag: str) -> None:
    if tag != "v" + version:
        raise PreflightError("release tag %r must equal v%s" % (tag, version))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(r"(?m)^## \[%s\](?:\s+-\s+\d{4}-\d{2}-\d{2})?\s*$" % re.escape(version),
                 changelog) is None:
        raise PreflightError("CHANGELOG has no exact release section for %s" % version)


def _installed_contract(contract: Any) -> dict[str, str]:
    import pops
    from pops import _pops

    if pops.__version__ != contract.PACKAGE_VERSION or _pops.__version__ != contract.PACKAGE_VERSION:
        raise PreflightError("installed Python/native/package versions disagree")
    if _pops.__abi_version__ != contract.NATIVE_ABI_VERSION:
        raise PreflightError("installed native ABI disagrees with release contract")
    if _pops.__release_contract_sha256__ != contract.RELEASE_CONTRACT_SHA256:
        raise PreflightError("installed native release digest disagrees with Python")
    extension = Path(_pops.__file__).resolve()
    if not extension.is_file():
        raise PreflightError("installed native extension has no readable origin")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "pops_file": str(Path(pops.__file__).resolve()),
        "native_extension": str(extension),
        "native_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }


def _inside(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _command_evidence(directory: Path, rows: Any, *, gate: str) -> list[Path]:
    if not isinstance(rows, list) or not rows:
        raise PreflightError("release evidence %s has no command transcripts" % gate)
    logs: list[Path] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"argv", "log", "sha256"}:
            raise PreflightError("release evidence %s command %d is malformed" % (gate, index))
        if not isinstance(row["argv"], list) or not row["argv"] \
                or not all(isinstance(value, str) and value for value in row["argv"]):
            raise PreflightError("release evidence %s command %d has invalid argv" % (gate, index))
        relative = Path(row["log"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PreflightError("release evidence %s command %d escapes its directory" % (gate, index))
        log = (directory / relative).resolve()
        if not _inside(directory, log) or not log.is_file():
            raise PreflightError("release evidence %s command %d has no transcript" % (gate, index))
        actual = hashlib.sha256(log.read_bytes()).hexdigest()
        if row["sha256"] != actual:
            raise PreflightError("release evidence %s command %d transcript hash drifted" %
                                (gate, index))
        logs.append(log)
    return logs


def _artifact_file(root: Path, relative: Any, digest: Any, *, label: str) -> None:
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise PreflightError("release evidence %s is malformed" % label)
    path = (root / relative).resolve()
    if not _inside(root, path) or not path.is_file():
        raise PreflightError("release evidence %s is absent" % label)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise PreflightError("release evidence %s hash drifted" % label)


def _checkpoint_tree(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise PreflightError("release evidence checkpoint is empty: %s" % path)
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _examples_evidence(directory: Path, gates: dict[str, Any]) -> None:
    examples = gates["examples"]["evidence"]
    reopen = gates["artifact_reopen"]["evidence"]
    restart = gates["strict_restart"]["evidence"]
    expected = {path.as_posix() for path in FINAL_EXAMPLES}
    if set(examples) != {"examples"} or set(reopen) != {"examples"} or set(restart) != {"examples"}:
        raise PreflightError("final-example evidence has an unknown schema")
    if set(examples["examples"]) != expected or set(reopen["examples"]) != expected \
            or set(restart["examples"]) != expected:
        raise PreflightError("final-example evidence does not cover exactly the final examples")
    command_rows = gates["examples"]["commands"]
    logs = _command_evidence(directory, command_rows, gate="examples")
    if len(logs) != len(FINAL_EXAMPLES):
        raise PreflightError("final examples must have one execution transcript each")
    for example in FINAL_EXAMPLES:
        key = example.as_posix()
        row = examples["examples"][key]
        if not isinstance(row, dict) or set(row) != {"source_sha256", "stdout_sha256", "output_root"}:
            raise PreflightError("release evidence %s is malformed" % key)
        if row["source_sha256"] != hashlib.sha256((ROOT / example).read_bytes()).hexdigest():
            raise PreflightError("release evidence source drifted for %s" % key)
        if not isinstance(row["output_root"], str):
            raise PreflightError("release evidence output root is invalid for %s" % key)
        matching = [log for log, command in zip(logs, command_rows)
                    if key in " ".join(command["argv"])]
        if len(matching) != 1:
            raise PreflightError("release evidence has no unique command transcript for %s" % key)
        transcript = matching[0].read_text(encoding="utf-8")
        if row["stdout_sha256"] != hashlib.sha256(transcript.encode("utf-8")).hexdigest():
            raise PreflightError("release evidence stdout hash drifted for %s" % key)
        if any(marker not in transcript for marker in REQUIRED_PROOF_MARKERS):
            raise PreflightError("release evidence lacks restart/reopen proof output for %s" % key)
        output_root = (directory / row["output_root"]).resolve()
        if not _inside(directory, output_root) or not output_root.is_dir():
            raise PreflightError("release evidence output root is absent for %s" % key)
        reopened = reopen["examples"][key]
        if not isinstance(reopened, dict) or set(reopened) != {"hdf5", "paraview"}:
            raise PreflightError("release evidence reopen record is malformed for %s" % key)
        for format_name in ("hdf5", "paraview"):
            artifacts = reopened[format_name]
            if not isinstance(artifacts, list) or not artifacts:
                raise PreflightError("release evidence has no %s output for %s" % (format_name, key))
            for artifact in artifacts:
                if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
                    raise PreflightError("release evidence %s output is malformed for %s" %
                                        (format_name, key))
                _artifact_file(output_root, artifact["path"], artifact["sha256"],
                               label="%s %s" % (format_name, key))
        restarted = restart["examples"][key]
        if not isinstance(restarted, dict) or set(restarted) != {
                "checkpoint", "tree_sha256", "proof_markers"}:
            raise PreflightError("release evidence restart record is malformed for %s" % key)
        checkpoint = Path(restarted["checkpoint"]).resolve()
        if not _inside(directory, checkpoint) or not checkpoint.exists():
            raise PreflightError("release evidence checkpoint is absent for %s" % key)
        if restarted["tree_sha256"] != _checkpoint_tree(checkpoint):
            raise PreflightError("release evidence checkpoint hash drifted for %s" % key)
        if restarted["proof_markers"] != list(REQUIRED_PROOF_MARKERS):
            raise PreflightError("release evidence restart proof markers drifted for %s" % key)


def _evidence(path: Path, contract: Any, commit: str, runtime: dict[str, str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "producer", "commit_sha", "package_version", "contract_sha256",
                "artifact_directory", "runtime", "gates"}
    if not isinstance(payload, dict) or set(payload) != expected \
            or payload["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise PreflightError("release evidence has an unknown or incomplete schema")
    if payload["commit_sha"] != commit or payload["package_version"] != contract.PACKAGE_VERSION \
            or payload["contract_sha256"] != contract.RELEASE_CONTRACT_SHA256:
        raise PreflightError("release evidence belongs to another build")
    producer = payload["producer"]
    expected_producer = {
        "script": "scripts/run_final_gate.py",
        "sha256": hashlib.sha256((ROOT / "scripts" / "run_final_gate.py").read_bytes()).hexdigest(),
    }
    if producer != expected_producer:
        raise PreflightError("release evidence was not produced by this final gate")
    if payload["runtime"] != runtime:
        raise PreflightError("release evidence belongs to another installed native extension")
    gates = payload["gates"]
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES):
        raise PreflightError("release evidence gate set must be exactly %s" % (REQUIRED_GATES,))
    for name in REQUIRED_GATES:
        row = gates[name]
        if not isinstance(row, dict) or set(row) != {"status", "commands", "evidence"}:
            raise PreflightError("release evidence %s has an invalid row" % name)
        if row["status"] != "passed":
            raise PreflightError("release gate %s did not produce passing evidence" % name)
    artifact_relative = Path(payload["artifact_directory"])
    if artifact_relative.is_absolute() or len(artifact_relative.parts) != 1 \
            or artifact_relative.name in {"", ".", ".."}:
        raise PreflightError("release evidence artifact directory is invalid")
    directory = (path.resolve().parent / artifact_relative).resolve()
    if not _inside(path.resolve().parent, directory) or not directory.is_dir():
        raise PreflightError("release evidence artifact directory is absent")
    for name in REQUIRED_GATES:
        commands = gates[name]["commands"]
        if name == "strict_restart":
            if commands != []:
                raise PreflightError("derived release gate %s must not invent a command" % name)
        else:
            _command_evidence(directory, commands, gate=name)
    for name in ("native_conformance", "python_conformance"):
        evidence = gates[name]["evidence"]
        expected = {"required_lane"} if name == "native_conformance" \
            else {"required_lane", "selection"}
        if not isinstance(evidence, dict) or set(evidence) != expected:
            raise PreflightError("release evidence %s lane is malformed" % name)
        lane = evidence["required_lane"]
        if not isinstance(lane, dict) or set(lane) != {
                "path", "sha256", "tests", "failures", "skips_or_xfails"}:
            raise PreflightError("release evidence %s JUnit summary is malformed" % name)
        if not isinstance(lane["tests"], int) or lane["tests"] <= 0 \
                or lane["failures"] != 0 or lane["skips_or_xfails"] != 0:
            raise PreflightError("release evidence %s required lane is not all-pass" % name)
        report = Path(lane["path"]).resolve()
        if not _inside(directory, report):
            raise PreflightError("release evidence %s JUnit path escapes its directory" % name)
        _artifact_file(directory, report.relative_to(directory), lane["sha256"],
                       label="%s JUnit" % name)
    if gates["python_conformance"]["evidence"]["selection"] != PYTHON_REQUIRED_SELECTION:
        raise PreflightError("release evidence Python required-lane selection drifted")
    _examples_evidence(directory, gates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    try:
        if args.release and (not args.tag or not args.installed or args.evidence is None):
            raise PreflightError("--release requires --tag, --installed and --evidence")
        contract = _generated()
        checks = _static_contract(contract)
        if args.release:
            _tag_contract(contract.PACKAGE_VERSION, args.tag)
            commit = _run("git", "rev-parse", "HEAD")
            if _run("git", "status", "--porcelain"):
                raise PreflightError("release checkout is dirty")
            runtime = _installed_contract(contract)
            _evidence(args.evidence, contract, commit, runtime)
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
