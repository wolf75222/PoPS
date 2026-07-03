#!/usr/bin/env python3
"""Select affected tests for CI.

The policy is intentionally conservative. C++ selection is manifest-driven over the
GoogleTest targets. Python selection is manifest-driven too, with a static
import-closure for ``python/pops/**`` changes so pure Python edits can run only the
tests that import the changed module.

The module is stdlib-only and runs before any ``pip install`` in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_import_closure  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/test_manifest.toml"


CPP_BROAD_FILES = {
    "CMakeLists.txt",
    "CMakePresets.json",
    "tests/CMakeLists.txt",
    "tests/cpp/test_sources.cmake",
    "tests/test_manifest.toml",
}

CPP_BROAD_PREFIXES = (
    "cmake/",
    "include/pops/core/",
    "include/pops/parallel/",
    "tests/cpp/support/",
)

PYTHON_BROAD_FILES = {
    "pyproject.toml",
    "python/CMakeLists.txt",
    "python/pops/__init__.py",
    "tests/python/conftest.py",
    "tests/test_manifest.toml",
}

PYTHON_BROAD_PREFIXES = (
    "python/bindings/",
    "tests/python/support/",
)

META_PREFIXES = (
    ".github/",
    "docs/",
    "tutorials/",
    "tests/python/architecture/",
)

CPP_PATH_AREAS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("include/pops/mesh/",), ("mesh", "amr")),
    (("include/pops/amr/",), ("amr", "mesh")),
    (("include/pops/coupling/",), ("coupling", "elliptic", "runtime")),
    (("include/pops/numerics/elliptic/", "include/pops/numerics/linalg/"), ("elliptic",)),
    (("include/pops/numerics/",), ("numerics",)),
    (("include/pops/physics/",), ("physics", "numerics")),
    (("include/pops/runtime/amr/",), ("amr", "runtime")),
    (("include/pops/runtime/",), ("runtime",)),
    (("include/pops/validation/",), ("physics", "validation")),
    (("python/bindings/amr/",), ("amr", "runtime", "codegen")),
    (("python/bindings/system/",), ("runtime", "physics", "codegen")),
    (("scripts/gen_solver_kernel.py",), ("codegen", "elliptic")),
)

PYTHON_PATH_AREAS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("python/pops/mesh/",), ("mesh", "amr")),
    (("python/pops/runtime/amr/",), ("amr", "runtime")),
    (("python/pops/runtime/",), ("runtime",)),
    (("python/pops/solvers/", "python/pops/linalg/"), ("elliptic",)),
    (("python/pops/codegen/", "python/pops/ir/", "python/pops/lib/"), ("codegen",)),
    (("python/pops/model/",), ("runtime", "physics")),
    (("python/pops/physics/", "python/pops/moments/"), ("physics", "numerics")),
    (("python/pops/numerics/",), ("numerics",)),
    (("python/pops/problem/",), ("problem", "runtime")),
    (("python/pops/time/",), ("time", "numerics")),
    (("python/pops/diagnostics/", "python/pops/output/"), ("runtime",)),
    (("python/pops/params/",), ("runtime",)),
    (("scripts/gen_solver_kernel.py",), ("codegen", "elliptic")),
)

AREA_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "amr": ("amr", "mesh"),
    "bindings": ("bindings", "runtime", "native_loader"),
    "codegen": ("codegen", "native_loader", "compiler", "bindings"),
    "coupling": ("coupling", "runtime", "elliptic", "amr", "physics"),
    "elliptic": ("elliptic", "solvers"),
    "io": ("io", "runtime"),
    "mesh": ("mesh", "amr"),
    "native_loader": ("native_loader", "codegen", "compiler"),
    "numerics": ("numerics", "elliptic", "solvers", "time"),
    "physics": ("physics", "numerics"),
    "problem": ("problem", "runtime"),
    "runtime": ("runtime", "bindings", "native_loader"),
    "solvers": ("solvers", "elliptic"),
    "time": ("time", "numerics", "solvers"),
    "validation": ("validation", "physics", "runtime"),
}

CPP_SMOKE_TARGETS = (
    "test_box2d",
    "test_reduce",
    "test_system_abstraction",
)

PYTHON_SMOKE_TESTS = (
    "tests/python/integration/bindings/test_bindings.py",
    "tests/python/unit/runtime/test_capabilities.py",
)


def normalize(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("./"):
        return cleaned[2:]
    return cleaned


def read_changed_files(path: Path) -> list[str]:
    return [normalize(line) for line in path.read_text().splitlines() if normalize(line)]


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"missing test manifest: {MANIFEST.relative_to(ROOT)}")
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def write_github_outputs(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as out:
        for key, value in values.items():
            print(f"{key}={value}", file=out)


def startswith_any(path: str, prefixes: Iterable[str]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def areas_for(path: str, table: Iterable[tuple[tuple[str, ...], tuple[str, ...]]]) -> set[str]:
    matched: set[str] = set()
    for prefixes, areas in table:
        if startswith_any(path, prefixes):
            matched.update(areas)
    return matched


def expand_area_labels(areas: Iterable[str]) -> set[str]:
    labels: set[str] = set()
    for area in areas:
        labels.add(area)
        labels.update(AREA_LABEL_ALIASES.get(area, ()))
    return labels


def add_reason(reasons: dict[str, set[str]], item: str, reason: str) -> None:
    reasons.setdefault(item, set()).add(reason)


def manifest_cpp_suites(manifest: dict) -> list[dict]:
    suites: list[dict] = []
    for suite in manifest.get("cpp", {}).get("suite", []):
        name = str(suite.get("name", ""))
        labels = set(str(label) for label in suite.get("labels", []))
        if not name:
            raise SystemExit("invalid C++ suite without name in tests/test_manifest.toml")
        # MPI-only suites are built solely in the ci-mpi job; keep them out of the serial
        # selection or the gate hits `ninja: unknown target`. The manifest label/mpi_nproc
        # is the primary filter; the `mpi` NAME SEGMENT check is a belt-and-braces guard for
        # a suite that forgets the label (see #435, test_amr_regrid_mpi_parity).
        if "mpi" in labels or suite.get("mpi_nproc") or "mpi" in name.split("_"):
            continue
        sources = [normalize(str(source)) for source in suite.get("sources", [])]
        if not sources:
            raise SystemExit(f"C++ suite {name} has no sources in tests/test_manifest.toml")
        suites.append({"name": name, "labels": labels, "sources": sources})
    return sorted(suites, key=lambda item: item["name"])


def manifest_python_suites(manifest: dict) -> list[dict]:
    suites: list[dict] = []
    for suite in manifest.get("python", {}).get("suite", []):
        name = str(suite.get("name", ""))
        path = normalize(str(suite.get("path", "")))
        labels = set(str(label) for label in suite.get("labels", []))
        if not name or not path:
            raise SystemExit("invalid Python suite without name/path in tests/test_manifest.toml")
        if "architecture" in labels:
            continue
        suite_path = ROOT / path
        files = sorted(str(p.relative_to(ROOT)) for p in suite_path.glob("test_*.py"))
        if not files:
            raise SystemExit(f"Python suite {name} has no test_*.py files under {path}")
        suites.append({"name": name, "path": path, "labels": labels, "files": files})
    return sorted(suites, key=lambda item: item["name"])


def direct_cpp_targets(changed: Iterable[str], all_targets: set[str]) -> set[str]:
    targets: set[str] = set()
    for path in changed:
        if path.startswith("tests/cpp/") and path.endswith(".cpp"):
            target = Path(path).stem
            if target in all_targets:
                targets.add(target)
    return targets


def direct_python_tests(changed: Iterable[str], all_tests: set[str]) -> set[str]:
    return {path for path in changed if path in all_tests}


def select_cpp_by_labels(suites: Iterable[dict], areas: Iterable[str], reasons: dict[str, set[str]]) -> set[str]:
    wanted_labels = expand_area_labels(areas)
    selected: set[str] = set()
    if not wanted_labels:
        return selected
    for suite in suites:
        matched = suite["labels"] & wanted_labels
        if matched:
            selected.add(suite["name"])
            add_reason(reasons, suite["name"], "manifest-labels:" + ",".join(sorted(matched)))
    return selected


def select_python_by_labels(suites: Iterable[dict], areas: Iterable[str], reasons: dict[str, set[str]]) -> set[str]:
    wanted_labels = expand_area_labels(areas)
    selected: set[str] = set()
    if not wanted_labels:
        return selected
    for suite in suites:
        matched = suite["labels"] & wanted_labels
        if matched:
            for test in suite["files"]:
                selected.add(test)
                add_reason(reasons, test, f"manifest-suite:{suite['name']} labels=" + ",".join(sorted(matched)))
    return selected


def force_full_from_changed(changed: Iterable[str], files: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(path in files or startswith_any(path, prefixes) for path in changed)


def only_meta(changed: Iterable[str]) -> bool:
    return all(
        startswith_any(path, META_PREFIXES)
        or path in {"README.md", "CONTRIBUTING.md", "CHANGELOG.md", "SECURITY.md", ".gitignore"}
        for path in changed
    )


def shard(items: list[str], index: int | None, total: int | None) -> list[str]:
    if index is None or total is None:
        return items
    if total <= 0 or index < 0 or index >= total:
        raise SystemExit(f"invalid shard {index}/{total}")
    return [item for i, item in enumerate(items) if i % total == index]


def write_explain_file(path: str | None, payload: dict) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_cpp(args: argparse.Namespace) -> int:
    changed = read_changed_files(Path(args.changed_files))
    manifest = load_manifest()
    suites = manifest_cpp_suites(manifest)
    all_targets = sorted(suite["name"] for suite in suites)
    all_target_set = set(all_targets)

    if not all_targets:
        raise SystemExit("no non-MPI C++ suites found in tests/test_manifest.toml")

    full_reasons: list[str] = []
    if args.force_all:
        full_reasons.append("force-all")
    if force_full_from_changed(changed, CPP_BROAD_FILES, CPP_BROAD_PREFIXES):
        full_reasons.append("broad-build-or-support-change")
    full = bool(full_reasons)
    if not full and any(path.startswith("include/pops/") for path in changed):
        known_include = any(areas_for(path, CPP_PATH_AREAS) for path in changed if path.startswith("include/pops/"))
        if not known_include:
            full = True
            full_reasons.append("unknown-cpp-public-api-path")

    selected: set[str] = set()
    areas: set[str] = set()
    reasons: dict[str, set[str]] = {}
    if not full:
        direct = direct_cpp_targets(changed, all_target_set)
        selected.update(direct)
        for target in direct:
            add_reason(reasons, target, "direct-test-edit")
        for path in changed:
            areas.update(areas_for(path, CPP_PATH_AREAS))
        selected.update(select_cpp_by_labels(suites, areas, reasons))
        if selected:
            for target in CPP_SMOKE_TARGETS:
                if target in all_target_set:
                    selected.add(target)
                    add_reason(reasons, target, "smoke-backstop")
        elif not only_meta(changed):
            full = True
            full_reasons.append("no-manifest-match-for-non-meta-change")

    if full or len(selected) > len(all_targets) * 0.75:
        if not full:
            full_reasons.append("selected-more-than-75-percent")
        mode = "all"
        targets = all_targets
        regex = ""
    else:
        mode = "subset" if selected else "none"
        targets = sorted(selected)
        regex = "^(" + "|".join(re.escape(t) for t in targets) + r")(\.|$)" if targets else "$^"

    summary = f"{mode}: {len(targets)}/{len(all_targets)} C++ tests"
    print(summary)
    if areas:
        print("areas=" + ",".join(sorted(areas)))
    for target in targets:
        print(target)

    write_github_outputs(
        getattr(args, "github_output", None),
        {
            "cpp_mode": mode,
            "cpp_targets": " ".join(targets),
            "cpp_regex": regex,
            "cpp_count": str(len(targets)),
            "cpp_total": str(len(all_targets)),
            "cpp_areas": ",".join(sorted(areas)) if areas else "-",
            "cpp_summary": summary,
        },
    )
    write_explain_file(
        getattr(args, "explain_file", None),
        {
            "kind": "cpp",
            "mode": mode,
            "changed_files": changed,
            "areas": sorted(areas),
            "expanded_labels": sorted(expand_area_labels(areas)),
            "full_reasons": full_reasons,
            "selected_count": len(targets),
            "total_count": len(all_targets),
            "selected": targets,
            "selected_reasons": {key: sorted(value) for key, value in sorted(reasons.items()) if key in targets},
        },
    )
    return 0


def _apply_cross_test_closure(selected: set[str], reasons: dict[str, set[str]]) -> None:
    before = set(selected)
    ci_import_closure._close_cross_test(selected, _test_to_test())
    for test in selected - before:
        add_reason(reasons, test, "cross-test-closure")


def plan_python(args: argparse.Namespace) -> int:
    changed = read_changed_files(Path(args.changed_files))
    manifest = load_manifest()
    suites = manifest_python_suites(manifest)
    all_tests = sorted({test for suite in suites for test in suite["files"]})
    all_test_set = set(all_tests)

    if not all_tests:
        raise SystemExit("no Python suites found in tests/test_manifest.toml")

    full_reasons: list[str] = []
    why: set[str] = set()
    if args.force_all:
        full_reasons.append("force-all")
    if force_full_from_changed(changed, PYTHON_BROAD_FILES, PYTHON_BROAD_PREFIXES):
        full_reasons.append("broad-file")
    full = bool(full_reasons)
    why.update(full_reasons)

    selected: set[str] = set()
    areas: set[str] = set()
    reasons: dict[str, set[str]] = {}
    if not full:
        direct = direct_python_tests(changed, all_test_set)
        selected.update(direct)
        if direct:
            why.add("direct-test")
        for test in direct:
            add_reason(reasons, test, "direct-test-edit")

        pops_changed = [path for path in changed if path.startswith("python/pops/") and path.endswith(".py")]
        if pops_changed:
            try:
                closure = ci_import_closure.impacted_tests(pops_changed, repo_root=ROOT)
            except ci_import_closure.OffGraphChange:
                full = True
                full_reasons.append("off-graph-pops-file")
                why.add("off-graph-pops-file")
            else:
                closure_hits = {test for test in closure if test in all_test_set}
                selected.update(closure_hits)
                if closure_hits:
                    why.add("import-closure")
                for test in closure_hits:
                    add_reason(reasons, test, "import-closure")

        if not full and any(path.startswith("python/pops/") and not path.endswith(".py") for path in changed):
            full = True
            full_reasons.append("non-py-pops-file")
            why.add("non-py-pops-file")

        if not full:
            for path in changed:
                if path.startswith("python/pops/") and path.endswith(".py"):
                    continue
                areas.update(areas_for(path, PYTHON_PATH_AREAS))
                areas.update(areas_for(path, CPP_PATH_AREAS))
            label_hits = select_python_by_labels(suites, areas, reasons)
            if label_hits:
                selected.update(label_hits)
                why.add("manifest-labels")

        if not full and selected:
            _apply_cross_test_closure(selected, reasons)
            for test in PYTHON_SMOKE_TESTS:
                if test in all_test_set:
                    selected.add(test)
                    add_reason(reasons, test, "smoke-backstop")

        if not full and not selected and not only_meta(changed):
            full = True
            full_reasons.append("no-manifest-match-for-non-meta-change")
            why.add("no-manifest-match-for-non-meta-change")

    if full or len(selected) > len(all_tests) * 0.75:
        if not full:
            full_reasons.append("selected-more-than-75-percent")
            why.add("selected-more-than-75-percent")
        mode = "all"
        selected_tests = all_tests
    else:
        mode = "subset" if selected else "none"
        selected_tests = sorted(selected)

    sharded = shard(selected_tests, args.shard_index, args.shard_total)
    if args.tests_file:
        Path(args.tests_file).write_text("".join(f"{test}\n" for test in sharded), encoding="utf-8")

    why_text = ",".join(sorted(why)) if why else "meta-only"
    summary = f"{mode}: {len(selected_tests)}/{len(all_tests)} Python test files [why: {why_text}]"
    if args.shard_index is not None and args.shard_total is not None:
        summary += f" ({len(sharded)} in shard {args.shard_index}/{args.shard_total})"
    print(summary)
    if areas:
        print("areas=" + ",".join(sorted(areas)))
    for test in sharded:
        print(test)

    write_github_outputs(
        getattr(args, "github_output", None),
        {
            "python_mode": mode,
            "python_count": str(len(selected_tests)),
            "python_total": str(len(all_tests)),
            "python_shard_count": str(len(sharded)),
            "python_areas": ",".join(sorted(areas)) if areas else "-",
            "python_why": why_text,
            "python_summary": summary,
        },
    )
    write_explain_file(
        getattr(args, "explain_file", None),
        {
            "kind": "python",
            "mode": mode,
            "changed_files": changed,
            "areas": sorted(areas),
            "expanded_labels": sorted(expand_area_labels(areas)),
            "full_reasons": full_reasons,
            "selected_count": len(selected_tests),
            "total_count": len(all_tests),
            "shard_index": args.shard_index,
            "shard_total": args.shard_total,
            "sharded_count": len(sharded),
            "selected": selected_tests,
            "sharded": sharded,
            "selected_reasons": {key: sorted(value) for key, value in sorted(reasons.items()) if key in selected_tests},
        },
    )
    return 0


def _test_to_test() -> dict[str, set[str]]:
    """Return only the cross-test edge map from ``ci_import_closure``."""
    _, edges = ci_import_closure.test_imports(repo_root=ROOT)
    return edges


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    cpp = sub.add_parser("cpp")
    cpp.add_argument("--changed-files", required=True)
    cpp.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    cpp.add_argument("--explain-file")
    cpp.add_argument("--force-all", action="store_true")
    cpp.set_defaults(func=plan_cpp)

    py = sub.add_parser("python")
    py.add_argument("--changed-files", required=True)
    py.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    py.add_argument("--explain-file")
    py.add_argument("--tests-file")
    py.add_argument("--shard-index", type=int)
    py.add_argument("--shard-total", type=int)
    py.add_argument("--force-all", action="store_true")
    py.set_defaults(func=plan_python)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
