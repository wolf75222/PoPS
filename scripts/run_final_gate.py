#!/usr/bin/env python3
"""Produce reproducible, integrity-checked evidence for the final PoPS release.

The gate has no success switches.  It first verifies the exact final source
contract, then builds the installed package, exercises complete native conformance and the exact
M4/final-example Python release ledger, executes every final example, independently reopens
their scientific artifacts, checks their restart evidence, and writes an
attestation outside the checkout.  ``release_preflight.py --release`` verifies
the attestation again against the live installed extension.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    exact_dimensions,
)

from final_release_contract import (  # noqa: E402
    FINAL_EXAMPLES,
    FINAL_EXAMPLE_REQUIRED_TESTS,
    FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS,
    FINAL_SPECIFICATION,
    INSTALLED_COMPONENT_PACKAGE_NODEID,
    PYTHON_REQUIRED_SELECTION,
    REQUIRED_PROOF_MARKERS,
    REQUIRED_RELEASE_GATES,
    required_python_conformance_nodeids,
    require_release_matrix_source_contract,
    require_source_contract,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SCHEMA_VERSION = 12
REQUIRED_GATES = REQUIRED_RELEASE_GATES


class FinalGateError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False)
    if completed.returncode:
        raise FinalGateError("git %s failed:\n%s" % (" ".join(args), completed.stdout[-4000:]))
    return completed.stdout.strip()


def _require_cpp_duration_catalogs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/ci_select_tests.py",
            "verify-cpp-duration-catalogs",
            "--shard-total",
            "7",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise FinalGateError(
            "C++ duration catalog contract failed:\n" + completed.stdout[-4000:]
        )


def _outside_checkout(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise FinalGateError("--evidence must be outside the checkout: %s" % resolved)


def _conda_command(
    arguments: Sequence[str],
    *,
    pops_include: Path | None = ROOT / "include",
) -> list[str]:
    """Run inside the same conda installation selected by the gate process.

    A login shell is deliberately forbidden here: user startup files may rewrite ``PATH`` and
    select another conda base, making the wheel build in one environment and the conformance gates
    import an older installation from another. ``conda run`` is non-interactive and preserves one
    explicit environment identity for every phase.
    """
    explicit = os.environ.get("POPS_CONDA_EXE")
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FinalGateError(
                "POPS_CONDA_EXE is not an executable file: %s" % candidate)
        conda = str(candidate.resolve())
    else:
        active = os.environ.get("CONDA_EXE")
        candidates = [Path(active).expanduser()] if active else []
        discovered = shutil.which("conda")
        if discovered:
            candidates.append(Path(discovered))
        candidates.extend(
            Path.home() / relative for relative in (
                "miniforge3/bin/conda",
                "mambaforge/bin/conda",
                "anaconda3/bin/conda",
                "miniconda3/bin/conda",
            )
        )
        conda = next(
            (str(candidate.resolve()) for candidate in candidates
             if candidate.is_file() and os.access(candidate, os.X_OK)),
            None,
        )
    if conda is None:
        raise FinalGateError("conda is unavailable on PATH; run scripts/setup_env.sh first")
    environment = os.environ.get("POPS_ENV_NAME", "pops")
    if not environment:
        raise FinalGateError("POPS_ENV_NAME must not be empty")
    return [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "/usr/bin/env",
        "PYTHONPATH=",
        "PYTHONNOUSERSITE=1",
        "POPS_REQUIRE_NATIVE_TESTS=1",
        "POPS_INCLUDE=" + ("" if pops_include is None else str(pops_include.resolve())),
        *arguments,
    ]


def _resolve_ctest_dir(requested: Path | None) -> Path:
    candidate = (ROOT / "build") if requested is None else requested.resolve()
    if not (candidate / "CTestTestfile.cmake").is_file() \
            or not (candidate / "CMakeCache.txt").is_file():
        label = "native preset build" if requested is None else "--ctest-dir"
        raise FinalGateError("%s is not a configured top-level CTest tree: %s" % (label, candidate))
    cache = (candidate / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    if "POPS_BUILD_TESTS:BOOL=ON" not in cache:
        raise FinalGateError("CTest tree was configured without POPS_BUILD_TESTS=ON: %s" % candidate)
    return candidate


def _runtime_provenance(dimension: int) -> dict[str, Any]:
    code = """
import hashlib
import json
from pathlib import Path
import pops
from pops._native_selector import select_native_dimension
select_native_dimension(%d)
from pops import _pops
extension = Path(_pops.__file__).resolve()
digest = hashlib.sha256(extension.read_bytes()).hexdigest()
print(json.dumps({
    'python_executable': str(Path(__import__('sys').executable).resolve()),
    'pops_file': str(Path(pops.__file__).resolve()),
    'native_dimension': _pops.__native_dimension__,
    'native_extension': str(extension),
    'native_sha256': digest,
}, sort_keys=True))
""" % dimension
    completed = subprocess.run(_conda_command(["python", "-c", code]), cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise FinalGateError("cannot obtain installed runtime provenance:\n%s" % completed.stdout[-4000:])
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FinalGateError("runtime provenance was not JSON: %s" % completed.stdout) from exc
    strings = {"python_executable", "pops_file", "native_extension", "native_sha256"}
    expected = {*strings, "native_dimension"}
    if set(payload) != expected or not all(isinstance(payload[name], str) and payload[name]
                                           for name in strings) \
            or payload["native_dimension"] != dimension:
        raise FinalGateError("runtime provenance is incomplete")
    return payload


def _json_evidence(stdout: str, *, gate: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FinalGateError("%s evidence was not JSON: %s" % (gate, stdout[-4000:])) from exc
    if not isinstance(payload, dict):
        raise FinalGateError("%s evidence must be a JSON object" % gate)
    return payload


def _signed_runtime_sha256(
    evidence: dict[str, Any], *, retained_variants: Sequence[dict[str, Any]],
    runtime_dimension: int,
) -> str:
    if set(evidence) != {"schema_version", "platform", "extensions"} \
            or evidence["schema_version"] != 2 or evidence["platform"] != "darwin":
        raise FinalGateError("codesign evidence is not the Darwin release proof")
    extensions = evidence["extensions"]
    retained = {
        row.get("dimension"): (row.get("sha256"), row.get("extension"))
        for row in retained_variants
        if isinstance(row, dict)
    }
    if not isinstance(extensions, list) or len(extensions) != len(retained) \
            or set(retained) != {row.get("dimension") for row in extensions
                                if isinstance(row, dict)}:
        raise FinalGateError("codesign evidence does not cover the retained native variant set")
    active_digest = None
    for extension in extensions:
        if not isinstance(extension, dict) or set(extension) != {
                "dimension", "path", "sha256", "signature"} \
                or extension["signature"] != "adhoc":
            raise FinalGateError("codesign extension evidence is malformed")
        digest = extension["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(character not in "0123456789abcdef" for character in digest):
            raise FinalGateError("codesign extension sha256 is malformed")
        retained_digest, retained_path = retained.get(extension["dimension"], (None, None))
        if retained_digest != digest or retained_path != extension["path"]:
            raise FinalGateError(
                "codesign changed retained wheel native bytes; refusing to publish "
                "an artifact different from the validated variant set"
            )
        if extension["dimension"] == runtime_dimension:
            active_digest = digest
    if active_digest is None:
        raise FinalGateError("codesign evidence omits the active runtime dimension")
    return active_digest


def _contract() -> tuple[str, str]:
    generated = ROOT / "python" / "pops" / "_generated_release_contract.py"
    specification = importlib.util.spec_from_file_location("_final_release_contract", generated)
    if specification is None or specification.loader is None:
        raise FinalGateError("cannot load generated release contract")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.PACKAGE_VERSION, module.RELEASE_CONTRACT_SHA256


def _check_clean_checkout() -> None:
    dirty = _git("status", "--porcelain")
    if dirty:
        raise FinalGateError("final gate requires a clean checkout before execution:\n%s" % dirty)


@dataclass
class Recorder:
    root: Path
    rows: dict[str, dict[str, Any]]
    sequence: int = 0

    def run(self, name: str, command: Sequence[str], *, evidence: dict[str, Any] | None = None) -> str:
        self.sequence += 1
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
        log = self.root / "logs" / ("%02d_%s.log" % (self.sequence, name))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
        command_row = {
            "argv": list(command),
            "log": str(log.relative_to(self.root)),
            "sha256": _sha256(log),
        }
        if completed.returncode:
            tail = "\n".join(completed.stdout.splitlines()[-40:])
            raise FinalGateError("%s failed (exit %d):\n%s" % (name, completed.returncode, tail))
        row = self.rows.setdefault(name, {
            "status": "passed", "commands": [], "evidence": evidence if evidence is not None else {},
        })
        if row["evidence"] != (evidence if evidence is not None else row["evidence"]):
            raise FinalGateError("gate %s attempted to write conflicting evidence" % name)
        row["commands"].append(command_row)
        return completed.stdout

    def derived(self, name: str, evidence: dict[str, Any]) -> None:
        if name in self.rows:
            raise FinalGateError("gate %s already has command evidence" % name)
        self.rows[name] = {"status": "passed", "commands": [], "evidence": evidence}


def _parse_checkpoint(stdout: str, *, example: Path) -> Path:
    matches = [line.split(":", 1)[1].strip() for line in stdout.splitlines()
               if line.strip().lower().startswith("checkpoint:")]
    if len(matches) != 1:
        raise FinalGateError("%s did not print one checkpoint path" % example)
    candidate = Path(matches[0])
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    if not candidate.exists():
        raise FinalGateError("%s reported a missing checkpoint: %s" % (example, candidate))
    return candidate


def _tree_hash(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FinalGateError("checkpoint directory is empty: %s" % path)
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(_sha256(item).encode("ascii"))
    return digest.hexdigest()


def _junit_summary(path: Path) -> dict[str, Any]:
    """Read one JUnit report and reject an empty, failed, skipped or xfailed lane."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise FinalGateError("invalid JUnit report %s: %s" % (path, exc)) from exc
    cases = tuple(root.iter("testcase"))
    if not cases:
        raise FinalGateError("JUnit report contains no executed tests: %s" % path)
    skipped = tuple(case for case in cases if case.find("skipped") is not None)
    failed = tuple(case for case in cases
                   if case.find("failure") is not None or case.find("error") is not None)
    if failed or skipped:
        raise FinalGateError(
            "required conformance lane is not all-pass: tests=%d failures=%d skips_or_xfails=%d" %
            (len(cases), len(failed), len(skipped)))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "tests": len(cases),
        "failures": 0,
        "skips_or_xfails": 0,
    }


def _require_junit_nodeids(
    path: Path,
    required: Sequence[str],
) -> list[str]:
    """Authenticate exact pytest tests inside an already all-pass JUnit lane."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise FinalGateError("invalid JUnit report %s: %s" % (path, exc)) from exc
    cases = tuple(root.iter("testcase"))
    authenticated = []
    for nodeid in required:
        relative, function_name = nodeid.split("::", 1)
        expected_class = str(Path(relative).with_suffix("")).replace("/", ".")
        matches = [
            case
            for case in cases
            if case.attrib.get("name", "").split("[", 1)[0] == function_name
            and case.attrib.get("classname", "").endswith(expected_class)
        ]
        if len(matches) != 1:
            raise FinalGateError(
                "required final-example test %s appears %d times in %s"
                % (nodeid, len(matches), path)
            )
        authenticated.append(nodeid)
    return authenticated


def _require_no_hidden_skip(stdout: str) -> None:
    """Reject script-style tests which print a skip reason but return success."""
    matches = [line.strip() for line in stdout.splitlines()
               if re.search(r"\bskip(?:ped|s)?\b", line, flags=re.IGNORECASE)]
    if matches:
        raise FinalGateError(
            "required Python conformance printed a hidden skip while returning success:\n%s" %
            "\n".join(matches[:20]))


def _reopen_outputs(
    output_dir: Path, *, example: Path,
) -> tuple[dict[str, Any], tuple[Path, ...], tuple[Path, ...]]:
    expectations = FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS.get(example)
    if expectations is None:
        raise FinalGateError("%s has no scientific-output release ledger" % example)
    output_root = output_dir.resolve()

    def roots(format_name: str) -> tuple[Path, ...]:
        resolved = tuple(
            (output_dir / expectation["artifact_root"]).resolve()
            for expectation in expectations[format_name]
        )
        if any(not root.is_relative_to(output_root) for root in resolved):
            raise FinalGateError(
                "%s has an escaping %s scientific artifact root"
                % (example, format_name)
            )
        return resolved

    def files(format_name: str, suffix: str) -> list[Path]:
        paths = []
        for expectation, artifact_root in zip(
            expectations[format_name], roots(format_name), strict=True,
        ):
            matches = sorted(
                path
                for path in artifact_root.rglob("*" + suffix)
                if path.is_file() and path.stat().st_size
            )
            if not matches:
                raise FinalGateError(
                    "%s did not produce a non-empty %s scientific artifact %s in %s"
                    % (example, format_name, suffix, expectation["artifact_root"])
                )
            paths.extend(matches)
        return paths

    hdf5_paths = files("hdf5", ".h5")
    npz_paths = files("npz", ".npz")
    paraview_vtu_paths = files("paraview", ".vtu")
    paraview_pvd_paths = files("paraview", ".pvd")
    paraview_paths = sorted((*paraview_vtu_paths, *paraview_pvd_paths))
    for path in hdf5_paths:
        if path.read_bytes()[:8] != b"\x89HDF\r\n\x1a\n":
            raise FinalGateError("HDF5 artifact has an invalid signature: %s" % path)
    for path in paraview_vtu_paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise FinalGateError("invalid ParaView XML %s: %s" % (path, exc)) from exc
        if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
            raise FinalGateError("ParaView artifact is not an UnstructuredGrid VTKFile: %s" % path)
    expected_vtu_paths = {path.resolve() for path in paraview_vtu_paths}
    paraview_roots = roots("paraview")
    for path in paraview_pvd_paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise FinalGateError("invalid ParaView collection XML %s: %s" % (path, exc)) from exc
        collection = root.find("Collection")
        datasets = () if collection is None else tuple(collection.findall("DataSet"))
        if root.tag != "VTKFile" or root.attrib.get("type") != "Collection" or not datasets:
            raise FinalGateError("ParaView artifact is not a non-empty PVD collection: %s" % path)
        containing_roots = tuple(
            artifact_root for artifact_root in paraview_roots
            if path.resolve().is_relative_to(artifact_root)
        )
        if len(containing_roots) != 1:
            raise FinalGateError("ParaView collection has an ambiguous artifact root: %s" % path)
        for dataset in datasets:
            relative = dataset.attrib.get("file")
            timestep = dataset.attrib.get("timestep")
            try:
                time_value = float(timestep) if timestep is not None else float("nan")
            except ValueError:
                time_value = float("nan")
            if not relative or not math.isfinite(time_value):
                raise FinalGateError("ParaView collection has an invalid DataSet row: %s" % path)
            referenced = (path.parent / relative).resolve()
            if not referenced.is_relative_to(containing_roots[0]) \
                    or referenced not in expected_vtu_paths:
                raise FinalGateError(
                    "ParaView collection references an absent or escaping VTU: %s" % referenced
                )
    for path in npz_paths:
        if path.read_bytes()[:4] != b"PK\x03\x04":
            raise FinalGateError("NPZ artifact has an invalid ZIP signature: %s" % path)
    return {
        "hdf5": [{"path": str(path.relative_to(output_dir)), "sha256": _sha256(path)}
                 for path in hdf5_paths],
        "npz": [{"path": str(path.relative_to(output_dir)), "sha256": _sha256(path)}
                for path in npz_paths],
        "paraview": [{"path": str(path.relative_to(output_dir)), "sha256": _sha256(path)}
                     for path in paraview_paths],
    }, tuple(hdf5_paths), tuple(npz_paths)


def _reopen_hdf5_with_installed_runtime(recorder: Recorder, paths: Sequence[Path]) -> None:
    code = (
        "import h5py, sys; "
        "[h5py.File(path, 'r').close() for path in sys.argv[1:]]; "
        "print('reopened_hdf5=' + str(len(sys.argv) - 1))"
    )
    recorder.run("artifact_reopen", _conda_command(
        ["python", "-c", code, *(str(path) for path in paths)]))


def _reopen_npz_with_installed_runtime(recorder: Recorder, paths: Sequence[Path]) -> None:
    code = """
import sys
import numpy as np

arrays = 0
for path in sys.argv[1:]:
    with np.load(path, allow_pickle=False) as payload:
        if not payload.files:
            raise RuntimeError("empty NPZ archive: " + path)
        for name in payload.files:
            np.asarray(payload[name])
            arrays += 1
print("reopened_npz=%d arrays=%d" % (len(sys.argv) - 1, arrays))
"""
    recorder.run("artifact_reopen", _conda_command(
        ["python", "-c", code, *(str(path) for path in paths)]))


def _run_examples(
    recorder: Recorder,
    *,
    runtime_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    reopened: dict[str, Any] = {}
    restarted: dict[str, Any] = {}
    examples_root = recorder.root / "examples"
    for example in FINAL_EXAMPLES:
        destination = examples_root / example.stem
        stdout = recorder.run(
            "examples",
            _conda_command([
                "python",
                "scripts/run_installed_example.py",
                "--runtime-sha256",
                runtime_sha256,
                "--example",
                str(example),
                "--",
                "--output-dir",
                str(destination),
            ]),
        )
        missing = [marker for marker in REQUIRED_PROOF_MARKERS if marker not in stdout]
        if missing:
            raise FinalGateError("%s did not emit required runtime proof markers %s" % (example, missing))
        checkpoint = _parse_checkpoint(stdout, example=example)
        reopened[example.as_posix()], hdf5_paths, npz_paths = _reopen_outputs(
            destination, example=example)
        _reopen_hdf5_with_installed_runtime(recorder, hdf5_paths)
        if npz_paths:
            _reopen_npz_with_installed_runtime(recorder, npz_paths)
        restarted[example.as_posix()] = {
            "checkpoint": str(checkpoint),
            "tree_sha256": _tree_hash(checkpoint),
            "proof_markers": list(REQUIRED_PROOF_MARKERS),
        }
        results[example.as_posix()] = {
            "source_sha256": _sha256(ROOT / example),
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "output_root": str(destination.relative_to(recorder.root)),
            "runtime_sha256": runtime_sha256,
        }
    return results, reopened, restarted


def _write_evidence(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="output JSON path outside this checkout")
    parser.add_argument("--ctest-dir", type=Path,
                        help="configured CTest build tree; auto-detected otherwise")
    parser.add_argument(
        "--dim", required=True, type=int, choices=(1, 2, 3),
        help="active dimension for doctor, conformance and final examples",
    )
    parser.add_argument(
        "--wheel-dim", action="append", required=True, type=int, choices=(1, 2, 3),
        help="exact wheel variant set; repeat (fat release: 1, 2, 3)",
    )
    parser.add_argument(
        "--wheel", type=Path,
        help="validate and install this already-built release wheel instead of rebuilding it; "
             "all native, Python, example, restart, documentation and provenance gates still run",
    )
    args = parser.parse_args(argv)
    try:
        wheel_dimensions = exact_dimensions(
            args.wheel_dim, where="final gate wheel dimensions"
        )
        if args.dim not in wheel_dimensions:
            raise FinalGateError(
                "active --dim must be included in the explicit --wheel-dim set"
            )
        if args.wheel is None and wheel_dimensions != (args.dim,):
            raise FinalGateError(
                "the current build path produces one leaf; use --wheel for an explicit fat wheel"
            )
        require_source_contract(ROOT)
        require_release_matrix_source_contract(ROOT)
        _require_cpp_duration_catalogs()
        _check_clean_checkout()
        evidence_path = _outside_checkout(args.evidence)
        evidence_root = evidence_path.parent.resolve() / (evidence_path.stem + ".artifacts")
        if evidence_path.exists() or evidence_root.exists():
            raise FinalGateError(
                "refusing to overwrite release evidence or its artifact directory: %s / %s" %
                (evidence_path, evidence_root))
        package_version, contract_sha256 = _contract()
        recorder = Recorder(evidence_root, {})

        recorder.run("official_build", ["bash", "scripts/setup_env.sh"])
        wheel_directory = evidence_root / "wheels"
        if args.wheel is None:
            recorder.run("official_build", [
                "bash", "scripts/build_python.sh", "--dim", str(args.dim),
                "--wheel-dir", str(wheel_directory),
            ])
        else:
            supplied_wheel = args.wheel.expanduser().resolve()
            if not supplied_wheel.is_file() or supplied_wheel.suffix != ".whl":
                raise FinalGateError("--wheel must name one readable wheel artifact")
            try:
                supplied_wheel.relative_to(ROOT)
            except ValueError:
                pass
            else:
                raise FinalGateError("--wheel must be outside the checkout")
            wheel_directory.mkdir(parents=True)
            retained_wheel = wheel_directory / supplied_wheel.name
            shutil.copy2(supplied_wheel, retained_wheel)
            recorder.run("official_build", _conda_command([
                "python", "-m", "pip", "install", "--force-reinstall", "--no-deps",
                str(retained_wheel),
            ]))
        wheels = tuple(wheel_directory.glob("pops-*.whl"))
        if len(wheels) != 1:
            raise FinalGateError("official build did not retain exactly one PoPS wheel")
        wheel = wheels[0]
        recorder.rows["official_build"]["evidence"] = {
            "wheel": {
                "path": str(wheel.relative_to(evidence_root)),
                "sha256": _sha256(wheel),
                "size": wheel.stat().st_size,
            },
        }
        recorder.run("installed_wheel", _conda_command([
            "python", "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel),
        ]))
        installed_wheel_stdout = recorder.run(
            "installed_wheel",
            _conda_command([
                "python", "scripts/prove_installed_wheel.py", "--wheel", str(wheel),
                *[
                    item
                    for dimension in wheel_dimensions
                    for item in ("--expect-dim", str(dimension))
                ],
            ]),
        )
        recorder.rows["installed_wheel"]["evidence"] = _json_evidence(
            installed_wheel_stdout, gate="installed_wheel"
        )
        codesign_stdout = recorder.run(
            "codesign",
            _conda_command([
                "python", "scripts/codesign_pops_extensions.py", "--json",
                *[
                    item
                    for dimension in wheel_dimensions
                    for item in ("--expect-dim", str(dimension))
                ],
            ]),
        )
        recorder.rows["codesign"]["evidence"] = _json_evidence(
            codesign_stdout, gate="codesign"
        )
        recorder.run("official_build", _conda_command(["cmake", "--preset", "serial"]))
        recorder.run("official_build", _conda_command(["cmake", "--build", "--preset", "serial"]))
        recorder.run("doctor", _conda_command([
            "python", "scripts/verify_installed_native.py",
            "--expect-dim", str(args.dim),
        ]))
        doctor_code = (
            "import pops; from pops._native_selector import select_native_dimension; "
            "select_native_dimension(%d); from pops.runtime.doctor import doctor; "
            "report = doctor(verbose=False); "
            "failed = {name: detail for name, (ok, detail) in report.items() if not ok}; "
            "assert not failed, failed; "
            "print('doctor package=' + pops.__version__)"
        ) % args.dim
        recorder.run("doctor", _conda_command(["python", "-c", doctor_code]))

        ctest_dir = _resolve_ctest_dir(args.ctest_dir)
        native_junit = evidence_root / "reports" / "native-conformance.xml"
        native_junit.parent.mkdir(parents=True, exist_ok=True)
        recorder.run("native_conformance", _conda_command([
            "ctest", "--test-dir", str(ctest_dir), "--output-on-failure",
            "--output-junit", str(native_junit),
        ]))
        recorder.rows["native_conformance"]["evidence"] = {
            "required_lane": _junit_summary(native_junit),
        }
        python_nodeids = required_python_conformance_nodeids(ROOT)
        python_junit = evidence_root / "reports" / "python-required-conformance.xml"
        required_stdout = recorder.run("python_conformance", _conda_command([
            "python", "-m", "pytest", "-q", "-s",
            "-o", "xfail_strict=true",
            *python_nodeids,
            "--junitxml", str(python_junit),
        ]))
        _require_no_hidden_skip(required_stdout)
        authenticated_python_nodeids = _require_junit_nodeids(
            python_junit, python_nodeids
        )
        installed_component_junit = (
            evidence_root / "reports" / "installed-component-package.xml"
        )
        installed_component_stdout = recorder.run(
            "python_conformance",
            _conda_command(
                [
                    "POPS_PROVE_INSTALLED_COMPONENT_PACKAGE=1",
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    "-o",
                    "xfail_strict=true",
                    INSTALLED_COMPONENT_PACKAGE_NODEID,
                    "--junitxml",
                    str(installed_component_junit),
                ],
                pops_include=None,
            ),
        )
        _require_no_hidden_skip(installed_component_stdout)
        recorder.rows["python_conformance"]["evidence"] = {
            "required_lane": _junit_summary(python_junit),
            "selection": PYTHON_REQUIRED_SELECTION,
            "nodeids": authenticated_python_nodeids,
            "final_example_nodeids": list(FINAL_EXAMPLE_REQUIRED_TESTS),
            "installed_component_package": {
                "nodeid": INSTALLED_COMPONENT_PACKAGE_NODEID,
                "headers": "installed-wheel",
                "lane": _junit_summary(installed_component_junit),
            },
        }
        signed_runtime_sha256 = _signed_runtime_sha256(
            recorder.rows["codesign"]["evidence"],
            retained_variants=(
                recorder.rows["installed_wheel"]["evidence"]["native_variants"]
            ),
            runtime_dimension=args.dim,
        )
        examples, reopened, restarted = _run_examples(
            recorder, runtime_sha256=signed_runtime_sha256)
        recorder.rows["examples"]["evidence"] = {"examples": examples}
        recorder.rows["artifact_reopen"]["evidence"] = {"examples": reopened}
        recorder.derived("strict_restart", {"examples": restarted})
        recorder.run("documentation", _conda_command(["python", "docs/check_docs.py"]))
        recorder.run("generated_products", _conda_command(
            ["python", "scripts/generate_release_contract.py", "--check"]))
        recorder.run("generated_products", _conda_command(
            ["python", "scripts/generate_component_catalog.py", "--check"]))
        recorder.run("diff", ["git", "diff", "--check"])
        recorder.run("diff", ["git", "diff", "--cached", "--check"])
        _check_clean_checkout()
        runtime = _runtime_provenance(args.dim)

        if tuple(recorder.rows) != REQUIRED_GATES:
            raise FinalGateError("internal evidence gate mismatch: %s" % tuple(recorder.rows))
        payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "producer": {
                "script": "scripts/run_final_gate.py",
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "commit_sha": _git("rev-parse", "HEAD"),
            "package_version": package_version,
            "contract_sha256": contract_sha256,
            "native_variant_set": list(wheel_dimensions),
            "artifact_directory": evidence_root.name,
            "runtime": runtime,
            "gates": recorder.rows,
        }
        _write_evidence(evidence_path, payload)
        print(json.dumps({
            "status": "passed",
            "evidence": str(evidence_path),
            "commit_sha": payload["commit_sha"],
            "package_version": package_version,
            "contract_sha256": contract_sha256,
            "final_specification": str(FINAL_SPECIFICATION),
            "examples": [str(item) for item in FINAL_EXAMPLES],
        }, sort_keys=True))
        return 0
    except (
        FinalGateError,
        NativeVariantManifestError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print("final release gate failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
