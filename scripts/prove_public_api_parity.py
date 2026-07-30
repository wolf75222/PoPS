#!/usr/bin/env python3
"""Prove that the release wheel and source checkout expose one pure-Python API."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "python" / "pops"
PROOF_SCHEMA_VERSION = 2
TYPED_PAYLOAD_SUFFIXES = (".py", ".pyi")
PUBLIC_ROOT = (
    "Model",
    "Program",
    "Case",
    "RunReport",
    "RunStopReason",
    "ExecutionContext",
    "set_threads",
    "validate",
    "inspect",
    "explain",
    "resolve",
    "compile",
    "bind",
    "run",
    "__version__",
)

_SNAPSHOT_PROGRAM = r"""
import hashlib
import inspect as _inspect
import json
from pathlib import Path
import sys

package_parent = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(package_parent))
import pops

expected_retired = (
    "Problem",
    "RuntimePolicies",
    "OutputPolicy",
    "CheckpointPolicy",
    "System",
    "AmrSystem",
    "ModelSpec",
    "BindInputs",
    "SystemConfig",
    "AmrSystemConfig",
    "CompiledTime",
    "compile_library",
    "read_library_manifest",
    "LibraryManifest",
)
expected_public = (
    "Model",
    "Program",
    "Case",
    "RunReport",
    "RunStopReason",
    "ExecutionContext",
    "set_threads",
    "validate",
    "inspect",
    "explain",
    "resolve",
    "compile",
    "bind",
    "run",
    "__version__",
)
if tuple(pops.__all__) != expected_public:
    raise RuntimeError("root public API does not match the final contract")
if "pops._pops" in sys.modules:
    raise RuntimeError("root import loaded pops._pops")
if not isinstance(pops.Case, type) or "__getattr__" in pops.Case.__dict__:
    raise RuntimeError("Case is not one explicit public type")
if "__getattr__" in pops.__dict__:
    raise RuntimeError("root package uses a dynamic public facade")
if any(hasattr(pops, name) for name in expected_retired):
    raise RuntimeError("root package still exposes a replaced public name")
if not (Path(pops.__file__).resolve().parent / "py.typed").is_file():
    raise RuntimeError("package has no py.typed marker")

model = pops.Model("parity")
state = model.state("U", components=("u",))
case = pops.Case("two_instances")
left = case.block("left", model)
right = case.block("right", model)
left_state = case.qualify(state, block=left)
right_state = case.qualify(state, block=right)
if left_state == right_state or left_state.block_ref != left or right_state.block_ref != right:
    raise RuntimeError("qualified handles do not disambiguate repeated Model instances")
if pops.validate(case) is not case or not case.frozen:
    raise RuntimeError("pure-Python validation did not freeze the exact Case")
report = pops.inspect(case)
if report["name"] != "two_instances" or set(report["blocks"]) != {"left", "right"}:
    raise RuntimeError("pure-Python inspection did not preserve qualified blocks")
if "pops._pops" in sys.modules:
    raise RuntimeError("authoring, validation, or inspection loaded pops._pops")

def _annotation(value):
    if isinstance(value, str):
        return value
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return module + "." + qualname
    return repr(value)

def _symbol(name):
    value = getattr(pops, name)
    if _inspect.isclass(value):
        kind = "class"
    elif _inspect.isfunction(value):
        kind = "function"
    else:
        kind = type(value).__name__
    try:
        call_signature = str(_inspect.signature(value, eval_str=False))
    except (TypeError, ValueError):
        call_signature = None
    annotations = getattr(value, "__annotations__", {})
    return {
        "kind": kind,
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "signature": call_signature,
        "annotations": {
            key: _annotation(annotation)
            for key, annotation in sorted(annotations.items())
        },
    }

public = list(pops.__all__)
snapshot = {
    "public": public,
    "symbols": {name: _symbol(name) for name in public},
    "case_is_explicit_type": True,
    "qualified_handles": True,
    "pure_authoring": True,
    "py_typed": True,
}
print(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))
"""


class PublicApiParityError(RuntimeError):
    """The source checkout and release wheel do not expose one exact public API."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_typed_payload(relative: str) -> bool:
    path = PurePosixPath(relative)
    return path.name == "py.typed" or path.suffix in TYPED_PAYLOAD_SUFFIXES


def _typed_manifest(package: Path, *, label: str) -> dict[str, str]:
    if not package.is_dir():
        raise PublicApiParityError("%s package is absent: %s" % (label, package))
    manifest = {
        path.relative_to(package).as_posix(): _sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and _is_typed_payload(path.relative_to(package).as_posix())
    }
    required = {"__init__.py", "_pops.pyi", "py.typed"}
    if not required.issubset(manifest):
        raise PublicApiParityError("%s package lacks its root API or typing payload" % label)
    return manifest


def _wheel_manifest(archive: zipfile.ZipFile) -> dict[str, str]:
    members = [
        info
        for info in archive.infolist()
        if not info.is_dir() and info.filename.startswith("pops/")
    ]
    names = [info.filename for info in members]
    if len(names) != len(set(names)):
        raise PublicApiParityError("release wheel contains duplicate pops package members")
    manifest = {
        info.filename.removeprefix("pops/"): _sha256_bytes(archive.read(info))
        for info in members
        if _is_typed_payload(info.filename.removeprefix("pops/"))
    }
    required = {"__init__.py", "_pops.pyi", "py.typed"}
    if not required.issubset(manifest):
        raise PublicApiParityError("release wheel lacks its root API or typing payload")
    return manifest


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    for info in archive.infolist():
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise PublicApiParityError("release wheel contains an unsafe member path")
    archive.extractall(destination)


def _snapshot(package_parent: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _SNAPSHOT_PROGRAM, str(package_parent.resolve())],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode:
        raise PublicApiParityError(
            "public API snapshot failed for %s:\n%s"
            % (package_parent, completed.stdout[-4000:])
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PublicApiParityError(
            "public API snapshot was not JSON for %s" % package_parent
        ) from exc
    if not isinstance(payload, dict):
        raise PublicApiParityError("public API snapshot is not an object")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_manifest_parity(
    reference: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    label: str,
) -> None:
    if candidate == reference:
        return
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    changed = sorted(
        name
        for name in set(reference) & set(candidate)
        if reference[name] != candidate[name]
    )
    raise PublicApiParityError(
        "%s Python/typing payload differs from source "
        "(missing=%s, extra=%s, changed=%s)"
        % (label, missing[:8], extra[:8], changed[:8])
    )


def _installed_package_from_distribution() -> Path:
    try:
        distribution = importlib.metadata.distribution("PoPS")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PublicApiParityError("the PoPS distribution is not installed") from exc
    files = distribution.files
    if files is None:
        raise PublicApiParityError("the installed PoPS distribution has no file inventory")
    package_initializers = [
        row for row in files if PurePosixPath(str(row)).as_posix() == "pops/__init__.py"
    ]
    if len(package_initializers) != 1:
        raise PublicApiParityError(
            "the installed PoPS distribution has no unique pops/__init__.py")
    package = Path(distribution.locate_file(package_initializers[0])).resolve().parent
    if not package.is_dir():
        raise PublicApiParityError("the installed PoPS package directory is absent")
    try:
        package.relative_to(ROOT)
    except ValueError:
        return package
    raise PublicApiParityError(
        "the installed-package proof resolved inside the source checkout: %s" % package)


def build_proof(
    wheel: Path,
    *,
    installed_package: Path | None = None,
) -> dict[str, Any]:
    """Compare one exact wheel archive with the current source checkout."""
    retained = wheel.expanduser().resolve()
    if retained.suffix != ".whl" or not retained.is_file():
        raise PublicApiParityError("release artifact is not one readable wheel")
    source_manifest = _typed_manifest(SOURCE_PACKAGE, label="source")
    installed = None if installed_package is None else installed_package.expanduser().resolve()
    if installed is not None:
        try:
            installed.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise PublicApiParityError(
                "the installed-package proof resolved inside the source checkout: %s" % installed)
    try:
        with tempfile.TemporaryDirectory(prefix="pops-public-api-") as temporary:
            extracted = Path(temporary)
            with zipfile.ZipFile(retained) as archive:
                wheel_manifest = _wheel_manifest(archive)
                _require_manifest_parity(
                    source_manifest, wheel_manifest, label="wheel")
                _safe_extract(archive, extracted)
            source_snapshot = _snapshot(SOURCE_PACKAGE.parent)
            wheel_snapshot = _snapshot(extracted)
            if installed is not None:
                installed_manifest = _typed_manifest(installed, label="installed")
                _require_manifest_parity(
                    source_manifest, installed_manifest, label="installed")
                installed_snapshot = _snapshot(installed.parent)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PublicApiParityError("release wheel is unreadable: %s" % exc) from exc
    if wheel_snapshot != source_snapshot:
        raise PublicApiParityError("wheel and source public API snapshots differ")
    if installed is not None and installed_snapshot != source_snapshot:
        raise PublicApiParityError("installed and source public API snapshots differ")
    if tuple(source_snapshot["public"]) != PUBLIC_ROOT:
        raise PublicApiParityError("public API snapshot differs from the final root contract")
    proof = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "wheel_path": str(retained),
        "wheel_sha256": _sha256(retained),
        "typed_payload_files": len(source_manifest),
        "typed_payload_sha256": _canonical_sha256(source_manifest),
        "public_api_sha256": _canonical_sha256(source_snapshot),
        "public_names": source_snapshot["public"],
        "pure_authoring": source_snapshot["pure_authoring"],
        "qualified_handles": source_snapshot["qualified_handles"],
        "py_typed": source_snapshot["py_typed"],
        "installed": installed is not None,
    }
    if installed is not None:
        proof.update({
            "installed_package": str(installed),
            "installed_typed_payload_sha256": _canonical_sha256(installed_manifest),
            "installed_public_api_sha256": _canonical_sha256(installed_snapshot),
        })
    return proof


def _write_evidence(path: Path, proof: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    try:
        destination.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise PublicApiParityError("evidence path must be outside the checkout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PublicApiParityError("refusing to overwrite public API evidence: %s" % destination)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        json.dump(proof, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument(
        "--installed",
        action="store_true",
        help="also prove the importlib.metadata-resolved installed distribution outside checkout",
    )
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        installed = _installed_package_from_distribution() if args.installed else None
        proof = build_proof(args.wheel, installed_package=installed)
        if args.evidence is not None:
            _write_evidence(args.evidence, proof)
    except (PublicApiParityError, OSError, ValueError) as exc:
        print("public API parity proof failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
