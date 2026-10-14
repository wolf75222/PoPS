"""Record construction, compilation and execution separately for every corpus member.

Run with the newly built, installed PoPS package (PYTHONPATH unset). This driver is a
reception harness only: equations and numerical steps execute in the generated artifact.
An unsupported case is an unsuccessful reception, never an expected-pass substitute.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pops

from .common import MissingCapabilityError


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("unserializable reception evidence: " + type(value).__name__)


def package_identity():
    from pops._native_selector import selected_native_module
    native = selected_native_module(required=True)
    package = Path(pops.__file__).resolve().parent
    files = sorted(package.rglob("*.py"))
    hashes = {str(p.relative_to(package)): digest(p) for p in files}
    # The compiler can use the development checkout's SDK. Authenticate both
    # trees, not just the extension that is eventually loaded by Python.
    repository = Path(__file__).resolve().parents[2]
    header_trees = {}
    for label, root in (("installed", package / "include"),
                        ("source", repository / "include")):
        header_trees[label] = {str(p.relative_to(root)): digest(p)
                              for p in sorted(root.rglob("*")) if p.is_file()}
    variants = package / "_native" / "variants.json"
    return {"package": str(package), "python_sources_sha256": hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "native": str(native.__file__), "native_sha256": digest(native.__file__),
        "python_files": hashes, "sdk_headers": header_trees,
        "native_variants": json.loads(variants.read_text())}


def artifact_sources(artifact, destination):
    rows = []
    objects = [("program", artifact.program)] + [
        ("block_" + block.name, block.model) for block in artifact.blocks]
    for name, compiled in objects:
        library = Path(compiled.so_path).resolve()
        cpp = getattr(compiled, "_generated_cpp", None)
        if not isinstance(cpp, str) or not cpp:
            raise RuntimeError("reception requires the retained generated C++: " + name)
        source = destination / (name + ".cpp")
        source.write_text(cpp)
        rows.append({"name": name, "source": str(source), "source_sha256": digest(source),
                     "library": str(library), "library_sha256": digest(library),
                     "compile_command": getattr(compiled, "compile_command", None)})
        if name == "program":
            maps = compiled.source_maps
            if not maps or any(not data["complete"] or
                               data["source_sha256"] != rows[-1]["source_sha256"]
                               for data in maps.values()):
                raise RuntimeError("reception requires complete source correspondence for this C++")
            mapping = destination / (name + ".source_map.json")
            mapping.write_text(json.dumps(maps, sort_keys=True, indent=2) + "\n")
            rows[-1]["source_map"] = str(mapping)
            rows[-1]["source_map_sha256"] = digest(mapping)
    return rows


def build(identifier):
    if identifier.startswith("M"):
        module = "models_M01_M14" if int(identifier[1:]) <= 14 else "models_M15_M28"
        return importlib.import_module("examples.api_v02." + module).build_model(identifier)
    if identifier.startswith("T"):
        return importlib.import_module("examples.api_v02.methods_T01_T12").build_method(identifier)
    return importlib.import_module("examples.api_v02.algebra_checks").build_algebra_check(identifier)


def run_one(identifier, destination):
    directory = destination / identifier
    directory.mkdir(parents=True, exist_ok=True)
    row = {"id": identifier, "status": "running", "stage": "build"}
    row["scientific_sources"] = {p.name: digest(p) for p in sorted(
        Path(__file__).parent.glob("*.py"))}
    start = time.perf_counter()
    try:
        spec = build(identifier)
        row["profile"] = spec.profile
        from .coverage import OBLIGATIONS, COMMON_SCOPE_NOTES
        row["unqualified_obligations"] = OBLIGATIONS.get(identifier, ())
        row["scope_notes"] = COMMON_SCOPE_NOTES
        row["stage"] = "validate_resolve"
        plan = pops.resolve(pops.validate(spec.case), layout=spec.layout,
                            compile_options={"debug": True})
        row["stage"] = "compile"
        artifact = pops.compile(plan)
        artifact.verify()
        row["artifact"] = artifact_sources(artifact, directory)
        row["program_hash"] = artifact.program_hash
        row["dimension"] = artifact.resolved_dimension
        row["stage"] = "bind"
        context = pops.ExecutionContext.mpi_world(artifact)
        runtime = pops.bind(artifact, initial_values=dict(spec.initial_values),
                            resources={"execution_context": context})
        row["stage"] = "run"
        report = pops.run(runtime, t_end=spec.t_end, max_steps=10000, console=False)
        row["time"] = runtime.time()
        row["accepted_steps"] = report.accepted_steps
        row["rejected_steps"] = report.rejected_steps
        row["stage"] = "assess"
        row["metrics"] = spec.invariants(runtime)
        row["native_levels"] = runtime.n_levels()
        snapshots = {f"{block}_level{level}": np.asarray(runtime.block_level_state_global(block, level))
                     for block in spec.subjects for level in range(row["native_levels"])}
        if row["native_levels"] > 1:
            row["patch_boxes"] = runtime.patch_boxes()
        np.savez(directory / "states.npz", **snapshots)
        row["states_sha256"] = digest(directory / "states.npz")
        if not row["metrics"].get("passed", False):
            raise AssertionError("native acceptance properties failed")
        if abs(runtime.time() - spec.t_end) > 2e-13:
            raise AssertionError("native execution did not reach the requested endpoint")
        row.update(status="executed_and_assessed", stage="complete")
    except MissingCapabilityError as error:
        row.update(status="unsupported", required=error.required, profile=error.profile,
                   error=str(error))
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error),
                   traceback=traceback.format_exc())
    row["elapsed_seconds"] = time.perf_counter() - start
    (directory / "result.json").write_text(json.dumps(row, indent=2, default=json_default) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", required=True,
                        help="explicit profiles matching --dimension; campaign dispatches the complete corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(1,2,3), default=2)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    from pops._native_selector import select_native_dimension, selected_native_module
    select_native_dimension(args.dimension)
    native = selected_native_module(required=True)
    world = native.mpi_world()
    if int(world.size) != 1:
        raise RuntimeError("this evidence recorder currently requires one MPI process")
    identity = package_identity()
    repository = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    rows = []
    report = {"source_base": head, "package_before": identity,
              "scope": f"authentic PoPS; selected Dim{args.dimension}; Kokkos CPU; MPI world size 1",
              "not_qualified": ["GPU", "multi-rank MPI", "AMR unless explicitly exercised",
                                "scientific convergence beyond the stated case checks"],
              "results": rows}
    for identifier in args.ids:
        row = run_one(identifier, destination)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("id", "status", "stage", "elapsed_seconds")}), flush=True)
        (destination / "result.json").write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    report["package_after"] = package_identity()
    report["package_unchanged"] = report["package_after"] == identity
    report["all_executed_and_assessed"] = report["package_unchanged"] and all(
        row["status"] == "executed_and_assessed" for row in rows)
    # Passing a finite profile is never a proof of every model/method variant.
    report["all_corpus_requirements_qualified"] = False
    report["qualification_policy"] = "only the precise profiles and checks recorded per row are qualified"
    (destination / "result.json").write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    return 0 if report["all_executed_and_assessed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
