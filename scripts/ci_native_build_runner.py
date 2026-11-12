#!/usr/bin/env python3
"""Record and compare the runner that built a CI native Python module.

The native module bakes the exact compiler and implicit include trees it was built
with (``pops.native-compiler-includes``). Native compilation refuses to run when
the current runner differs, and this script never relaxes that refusal. A GitHub
image rollout can put the job that consumes a module artifact on another image
than the job that built it; every native test of that job would then fail with
the same contract error. ``check`` answers, before any test starts and with the
contract's own fingerprints, whether the downloaded module's baked contract holds
on this runner, so CI can rebuild the module on this runner instead. It also names
the two images and the Debian packages whose files differ.

``record`` runs where the module was built and authenticated. It reads the
contract from the loaded native leaf itself, never from the build tree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pops.ci-native-build-runner@1"
RECORD_NAME = "native-build-runner.json"
#: Files listed per changed package; the full receipt keeps every changed package.
FILES_PER_PACKAGE = 25


def _tree_module() -> Any:
    source = ROOT / "python/pops/codegen/_compiler_include_tree.py"
    spec = importlib.util.spec_from_file_location("pops_ci_runner_include_tree", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("compiler include fingerprint implementation is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def single_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("pops-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(f"exactly one pops wheel is required in {directory}")
    return wheels[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runner_identity(environment: dict[str, str] | None = None) -> dict[str, str | None]:
    environment = dict(os.environ) if environment is None else environment
    return {"image_os": environment.get("ImageOS"),
            "image_version": environment.get("ImageVersion"),
            "runner_name": environment.get("RUNNER_NAME")}


def installed_packages() -> dict[str, str] | None:
    """Exact Debian package versions, or ``None`` where dpkg does not exist."""
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f", "${binary:Package}\t${Version}\n"],
            capture_output=True, text=True, check=True, env={**os.environ, "LC_ALL": "C"})
    except FileNotFoundError:
        return None
    packages = {}
    for line in result.stdout.splitlines():
        name, _, version = line.partition("\t")
        if name:
            packages[name] = version
    return packages


def package_files(package: str) -> list[str]:
    result = subprocess.run(["dpkg-query", "-L", package], capture_output=True, text=True,
                            check=False, env={**os.environ, "LC_ALL": "C"})
    return [line for line in result.stdout.splitlines() if line.startswith("/")]


def module_contract(dimension: int) -> dict:
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(dimension)
    raw = getattr(native, "__compiler_include_contract__", None)
    if type(raw) is not str:
        raise RuntimeError("the native leaf exposes no compiler include contract")
    contract = json.loads(raw)
    _validate_contract(contract, "native leaf")
    return contract


def _validate_contract(contract: Any, where: str) -> None:
    if (type(contract) is not dict or type(contract.get("compiler")) is not dict
            or type(contract.get("include_trees")) is not list
            or any(type(tree) is not dict or set(tree) != {"root", "sha256", "file_count"}
                   for tree in contract["include_trees"])):
        raise ValueError(f"{where} compiler include contract is malformed")


def record(dimension: int, wheel_dir: Path, output: Path, *,
           contract: dict | None = None, packages: dict[str, str] | None = None,
           environment: dict[str, str] | None = None) -> dict:
    wheel = single_wheel(wheel_dir)
    data = {
        "schema": SCHEMA,
        "dimension": dimension,
        "wheel": {"name": wheel.name, "sha256": sha256_file(wheel)},
        "contract": module_contract(dimension) if contract is None else contract,
        "runner": runner_identity(environment),
        "packages": installed_packages() if packages is None else packages,
    }
    _validate_contract(data["contract"], "recorded")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def load_record(package_dir: Path, dimension: int) -> dict:
    path = package_dir / RECORD_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"module artifact carries no {RECORD_NAME}: {package_dir}") from exc
    if type(data) is not dict or data.get("schema") != SCHEMA:
        raise ValueError(f"{path} has an unsupported schema")
    if data.get("dimension") != dimension:
        raise ValueError(f"{path} records Dim={data.get('dimension')!r}, expected Dim={dimension}")
    wheel = single_wheel(package_dir)
    if data.get("wheel") != {"name": wheel.name, "sha256": sha256_file(wheel)}:
        raise ValueError(f"{path} does not describe the wheel beside it")
    _validate_contract(data.get("contract"), "recorded")
    return data


def compare(built: dict, local: dict,
            snapshot: Callable[[str], dict] | None = None) -> dict:
    """Return the compiler and include roots of ``built`` that differ on this runner.

    ``local`` is this runner's identity receipt contract. A built root absent from it is
    fingerprinted here with the contract's own implementation.
    """
    local_trees = {tree["root"]: tree for tree in local["include_trees"]}
    take = snapshot if snapshot is not None else _tree_module().snapshot_include_tree
    differing, matching = [], []
    for tree in built["include_trees"]:
        root = tree["root"]
        here = local_trees.get(root)
        if here is None:
            try:
                taken = take(root)
                here = {"root": root, "sha256": taken["sha256"], "file_count": taken["file_count"]}
            except (OSError, ValueError):
                here = None
        if here is not None and (here["sha256"], here["file_count"]) == (
                tree["sha256"], tree["file_count"]):
            matching.append(root)
        else:
            differing.append({"root": root, "built": tree, "here": here})
    compiler_keys = ("path", "size", "sha256")
    compiler_changed = any(built["compiler"].get(key) != local["compiler"].get(key)
                           for key in compiler_keys)
    return {"compiler_changed": compiler_changed, "differing_roots": differing,
            "matching_roots": matching}


def package_changes(built: dict[str, str] | None, here: dict[str, str] | None) -> list[dict]:
    if built is None or here is None:
        return []
    return [{"package": name, "built": built.get(name), "here": here.get(name)}
            for name in sorted(set(built) | set(here)) if built.get(name) != here.get(name)]


def files_in_differing_roots(paths: list[str], differing: list[str], matching: list[str]) -> list[str]:
    def under(path: str, root: str) -> bool:
        return path == root or path.startswith(root.rstrip("/") + "/")

    selected = []
    for path in paths:
        owners = [root for root in differing if under(path, root)]
        if not owners:
            continue
        deepest = max(len(root) for root in owners)
        # A file under a deeper root that still matches cannot explain the difference.
        if any(under(path, root) and len(root) > deepest for root in matching):
            continue
        selected.append(path)
    return selected


def check(dimension: int, package_dir: Path, identity: Path, *,
          packages: dict[str, str] | None = None,
          environment: dict[str, str] | None = None,
          files_of: Callable[[str], list[str]] = package_files,
          snapshot: Callable[[str], dict] | None = None) -> dict:
    built = load_record(package_dir, dimension)
    receipt = json.loads(identity.read_text(encoding="utf-8"))
    local = receipt.get("contract") if type(receipt) is dict else None
    _validate_contract(local, "this runner's")
    assert local is not None
    difference = compare(built["contract"], local, snapshot)
    report = {"schema": "pops.ci-native-runner-check@1", "dimension": dimension,
              "built_runner": built["runner"], "this_runner": runner_identity(environment),
              **difference, "rebuild": bool(difference["compiler_changed"]
                                            or difference["differing_roots"])}
    if report["rebuild"]:
        here = installed_packages() if packages is None else packages
        changes = package_changes(built.get("packages"), here)
        roots = [row["root"] for row in difference["differing_roots"]]
        for change in changes:
            files = files_in_differing_roots(files_of(change["package"]), roots,
                                             difference["matching_roots"])
            change["include_files"] = files
        report["package_changes"] = changes
    return report


def _image(runner: dict) -> str:
    return "%s %s" % (runner.get("image_os") or "unknown-image",
                      runner.get("image_version") or "unknown-version")


def describe(report: dict) -> list[str]:
    dimension = report["dimension"]
    if not report["rebuild"]:
        return [f"Dim{dimension} native module contract holds on this runner "
                f"({_image(report['this_runner'])}); the module is used as is."]
    lines = [f"Dim{dimension} native module was built on {_image(report['built_runner'])}; "
             f"this runner is {_image(report['this_runner'])}."]
    if report["compiler_changed"]:
        lines.append("The native compiler binary differs.")
    for row in report["differing_roots"]:
        here = row["here"]
        lines.append("Include tree %s: built %s files sha256 %s; here %s." % (
            row["root"], row["built"]["file_count"], row["built"]["sha256"][:16],
            "unavailable" if here is None else "%s files sha256 %s" % (
                here["file_count"], here["sha256"][:16])))
    changes = report.get("package_changes", [])
    explaining = [change for change in changes if change.get("include_files")]
    for change in explaining:
        files = change["include_files"]
        shown = ", ".join(files[:FILES_PER_PACKAGE])
        more = "" if len(files) <= FILES_PER_PACKAGE else f" (+{len(files) - FILES_PER_PACKAGE} more)"
        lines.append(f"Package {change['package']} {change['built']} -> {change['here']}: "
                     f"{shown}{more}")
    if changes and not explaining:
        lines.append(f"{len(changes)} Debian packages differ; none owns a file in a differing tree.")
    elif len(changes) > len(explaining):
        lines.append(f"{len(changes) - len(explaining)} other Debian packages differ "
                     "without files in a differing tree.")
    if not changes:
        lines.append("No Debian package version differs (or dpkg is unavailable).")
    return lines


def _write_outputs(report: dict, github_output: Path | None, summary: Path | None,
                   receipt: Path | None) -> None:
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as stream:
            stream.write("rebuild=%s\n" % ("true" if report["rebuild"] else "false"))
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary is not None and report["rebuild"]:
        with summary.open("a", encoding="utf-8") as stream:
            stream.write(f"### Dim{report['dimension']} native module rebuilt for this runner\n\n")
            stream.writelines(f"- {line}\n" for line in describe(report))
            stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record")
    rec.add_argument("--dimension", type=int, choices=(1, 2, 3), required=True)
    rec.add_argument("--wheel-dir", type=Path, required=True)
    rec.add_argument("--output", type=Path, required=True)
    chk = commands.add_parser("check")
    chk.add_argument("--dimension", type=int, choices=(1, 2, 3), required=True)
    chk.add_argument("--package-dir", type=Path, required=True)
    chk.add_argument("--identity", type=Path, required=True)
    chk.add_argument("--github-output", type=Path)
    chk.add_argument("--summary", type=Path)
    chk.add_argument("--receipt", type=Path)
    chk.add_argument("--require-match", action="store_true",
                     help="fail with a named error instead of requesting a rebuild")
    args = parser.parse_args(argv)
    if args.command == "record":
        data = record(args.dimension, args.wheel_dir, args.output)
        print(f"Dim{args.dimension} native module recorded on {_image(data['runner'])}")
        return 0
    report = check(args.dimension, args.package_dir, args.identity)
    lines = describe(report)
    _write_outputs(report, args.github_output, args.summary, args.receipt)
    if not report["rebuild"]:
        print(lines[0])
        return 0
    title = "Native module built on another runner image"
    if args.require_match:
        print(f"::error title={title}::" + " ".join(lines))
        return 1
    print(f"::warning title={title}::" + " ".join(lines)
          + f" Rebuilding Dim{args.dimension} on this runner; the include contract is unchanged.")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    # ``record`` imports the authenticated package named by the caller's PYTHONPATH,
    # never the source tree.
    raise SystemExit(main())
