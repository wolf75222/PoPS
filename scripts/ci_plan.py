#!/usr/bin/env python3
"""Plan PR jobs from the same affected-test inventory their runners consume.

No native imports or builds: this entry point uses only stdlib and the existing
manifest/import/include selectors. Uncertain changes select the complete matrix.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from fnmatch import fnmatchcase
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import ci_route_mode
import ci_select_tests as select
import ci_shard_binpack
import ci_python_dimensions
from ci_pytest_timings import SHARD_TOTAL as PYTHON_SHARDS

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "scripts/ci_components.toml"
CPP_SHARDS = 13
UNKNOWN_DIFF = "__unresolved_pr_change_scope__"


def changed_files(base: str, head: str, *, root: Path = ROOT) -> list[str]:
    """Diff PR head against its merge base; include both sides of a rename."""
    if not base or not head:
        raise ValueError("PR base and head revisions are required")
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)
    ancestor = git("merge-base", base, head).decode().strip()
    fields = git("diff", "--name-status", "-z", "--find-renames", ancestor, head).decode().split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        count = 2 if status[0] in "RC" else 1
        paths.update(fields[index + 1:index + 1 + count])
        index += 1 + count
    # The existing selectors use newline-delimited manifests. Never truncate an
    # unusual filename into a trustworthy-looking partial selection.
    if any("\n" in path or "\r" in path for path in paths):
        return [UNKNOWN_DIFF]
    return sorted(paths)


def classify(changed: list[str]) -> tuple[dict, list[str]]:
    policy = tomllib.loads(POLICY.read_text(encoding="utf-8"))
    impact = {}
    full_reasons = []
    for path in changed:
        matches = [rule for rule in policy["component"]
                   if any(fnmatchcase(path, pattern) for pattern in rule["paths"])]
        impact[path] = {
            "components": [rule["name"] for rule in matches],
            **{flag: any(rule.get(flag, False) for rule in matches)
               for flag in ("full", "mpi", "architecture", "metadata")},
        }
        if not matches:
            full_reasons.append(f"unmapped-path:{path}")
        elif impact[path]["full"]:
            full_reasons.append(f"shared-build-or-core:{path}")
        elif not impact[path]["metadata"] and not (ROOT / path).is_file():
            full_reasons.append(f"deleted-or-missing-source:{path}")
    return impact, full_reasons


def write_lines(path: Path, rows: list[str]) -> None:
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def make_plan(changed: list[str], *, event_name: str, output_dir: Path,
              ci_full: bool = False, ci_kokkos: bool = False,
              force_full: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(output_dir / "changed-files.txt", changed)
    impact, full_reasons = classify(changed)
    if event_name != "pull_request" or ci_full or ci_kokkos or force_full:
        full_reasons.append("event-or-explicit-full-override")
    full = bool(full_reasons)
    manifest = select.load_manifest()
    mpi_paths = {entry["path"] for entry in
                 select.manifest_python_mpi_entrypoints(manifest)
                 + select.manifest_python_mpi_orchestrators(manifest)}
    # Architecture and documentation edits do not change functional behavior.
    # Native-only test edits likewise cannot change the installed Python module.
    source_inputs = [path for path in changed if not impact[path]["metadata"]
                     and not path.startswith("tests/python/architecture/")]
    python_inputs = [path for path in source_inputs
                     if not path.startswith("tests/cpp/") and path not in mpi_paths]
    cpp_inputs_path = output_dir / "cpp-changed-files.txt"
    python_inputs_path = output_dir / "python-changed-files.txt"
    write_lines(cpp_inputs_path, source_inputs)
    write_lines(python_inputs_path, python_inputs)

    def selections(all_tests: bool) -> tuple[dict, select.PythonSelection]:
        with redirect_stdout(io.StringIO()):
            select.plan_cpp(argparse.Namespace(
                changed_files=str(cpp_inputs_path), force_all=all_tests,
                shard_index=0, shard_total=CPP_SHARDS, github_output=None,
                explain_file=str(output_dir / "cpp-plan.json")))
        cpp = json.loads((output_dir / "cpp-plan.json").read_text())
        python = select.compute_python_selection(str(python_inputs_path), all_tests)
        return cpp, python

    cpp, python = selections(full)
    # A failure to resolve one language's dependencies is uncertainty for the PR,
    # not permission to omit the other language or the parallel capability lanes.
    if not full and (cpp["full_reasons"] or python.full_reasons):
        full_reasons.extend(cpp["full_reasons"] + python.full_reasons)
        full = True
        cpp, python = selections(True)

    excluded = set(ci_shard_binpack.EXCLUDED_FROM_SHARDS)
    ordinary_python = [path for path in python.selected_tests if path not in excluded]
    python_shards = ci_shard_binpack.assign_shards(
        ordinary_python, PYTHON_SHARDS, ci_shard_binpack.load_durations())
    ci_shard_binpack.verify_partition(ordinary_python, python_shards)
    compile_cache = sorted(set(python.selected_tests) & excluded)
    dimensions = ci_python_dimensions.partition(
        python.selected_tests, json.loads(ci_python_dimensions.CONTRACT.read_text()))
    if set(dimensions) - {1, 2}:
        raise ValueError("CI package downloads support only declared native Dim1/Dim2")
    # Installed serial shards and MPI entrypoints have separate manifest ownership.
    # Their import graph still exposes MPI consumers of a changed Python module.
    pops_changes = [path for path in source_inputs if path.startswith("python/pops/")]
    mpi_imports = []
    if pops_changes and not full:
        affected = select.ci_import_closure.impacted_tests(pops_changes, repo_root=ROOT)
        mpi_imports = sorted(affected & mpi_paths)

    all_architecture = sorted(path.relative_to(ROOT).as_posix()
                              for path in (ROOT / "tests/python/architecture").glob("test_*.py"))
    # Source-reading architecture fences are a coherent cross-cutting contract.
    # A test-only edit needs only that test; broad source changes retain the fences.
    architecture = all_architecture if full or any(row["architecture"] for row in impact.values()) else [
        path for path in changed if path in all_architecture]
    write_lines(output_dir / "architecture-tests.txt", architecture)

    try:
        _, runtime_consumers = select._runtime_object_lib_map()
        shared_targets = set().union(*runtime_consumers.values()) if runtime_consumers else set()
    except select.ci_include_graph.GraphError:
        # An uncertain linkage graph cannot justify skipping the shared build.
        shared_targets = set(cpp["selected"])
    outputs = {
        "full": full,
        "cpp_required": bool(cpp["selected"]),
        "cpp_prewarm_required": bool(set(cpp["selected"]) & shared_targets),
        "python_required": bool(ordinary_python),
        "compile_cache_required": bool(compile_cache),
        "architecture_required": bool(architecture),
        "mpi_required": full or bool(mpi_imports) or bool(set(changed) & mpi_paths)
                        or any(row["mpi"] for row in impact.values()),
        "openmp_required": full,
        "cpp_matrix": [i for i, shard in enumerate(cpp["target_shards"]) if shard] or [0],
        "python_matrix": [i for i, shard in enumerate(python_shards) if shard] or [0],
        "python_dimensions": sorted(dimensions) or [2],
    }
    plan = {
        "schema": 1, "changed_files": changed, "impact": impact,
        "full_reasons": full_reasons, "outputs": outputs,
        "cpp": cpp["selected"], "python": python.selected_tests,
        "architecture": architecture, "compile_cache": compile_cache,
        "mpi_import_consumers": mpi_imports,
        "python_shards": python_shards,
    }
    (output_dir / "python-plan.json").write_text(json.dumps({
        "selected": python.selected_tests, "mode": python.mode,
        "reasons": {key: sorted(value) for key, value in python.reasons.items()},
        "full_reasons": python.full_reasons,
    }, indent=2) + "\n")
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    changes = commands.add_parser("changes")
    changes.add_argument("--event-name", required=True)
    changes.add_argument("--base", default="")
    changes.add_argument("--head", default="")
    changes.add_argument("--output-file", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--event-name", required=True)
    plan.add_argument("--changed-files", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--github-output", type=Path, required=True)
    for flag in ("ci-full", "ci-kokkos", "force-full"):
        plan.add_argument("--" + flag, default="false")
    args = parser.parse_args()
    if args.command == "changes":
        try:
            paths = changed_files(args.base, args.head) if args.event_name == "pull_request" else []
        except (ValueError, subprocess.CalledProcessError) as exc:
            print(f"Cannot establish PR diff; selecting full fallback: {exc}", file=sys.stderr)
            paths = [UNKNOWN_DIFF]
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        write_lines(args.output_file, paths)
        return
    result = make_plan(select.read_changed_files(args.changed_files),
                       event_name=args.event_name, output_dir=args.output_dir,
                       **{name: ci_route_mode.parse_bool(getattr(args, name), name=name)
                          for name in ("ci_full", "ci_kokkos", "force_full")})
    outputs = {key: json.dumps(value, separators=(",", ":"))
               for key, value in result["outputs"].items()}
    select.write_github_outputs(str(args.github_output), outputs)
    print(f"Affected CI: {len(result['cpp'])} C++ targets, {len(result['python'])} Python files, "
          f"{len(result['architecture'])} architecture files; full={outputs['full']}")
    print("Fallback reasons: " + (", ".join(result["full_reasons"]) or "none"))


if __name__ == "__main__":
    main()
