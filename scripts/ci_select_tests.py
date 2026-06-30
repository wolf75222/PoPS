#!/usr/bin/env python3
"""Select affected tests for CI.

The policy is intentionally conservative:

* shared build/runtime changes run the full relevant suite;
* direct test edits run the touched tests;
* clear domain changes run the matching domain tests plus a small smoke set;
* unknown paths run everything rather than silently dropping coverage.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


CPP_BROAD_FILES = {
    "CMakeLists.txt",
    "CMakePresets.json",
    "tests/CMakeLists.txt",
}

CPP_BROAD_PREFIXES = (
    "cmake/",
    "include/pops/core/",
    "include/pops/parallel/",
)

PYTHON_BROAD_FILES = {
    "pyproject.toml",
    "python/CMakeLists.txt",
    "python/pops/__init__.py",
}

PYTHON_BROAD_PREFIXES = (
    "python/bindings/",
)

META_PREFIXES = (
    ".github/",
    "docs/",
    "tutorials/",
    "tests/architecture/",
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
    (("python/pops/time/",), ("time", "numerics")),
    (("python/pops/diagnostics/", "python/pops/output/"), ("runtime",)),
    (("python/pops/params/",), ("runtime",)),
    (("scripts/gen_solver_kernel.py",), ("codegen", "elliptic")),
)

CPP_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "mesh": (
        "box",
        "fab",
        "mesh",
        "geometry",
        "boundary",
        "bc",
        "fill_boundary",
        "domain",
        "cut",
        "layout",
        "patch",
        "coverage",
        "cluster",
        "load_balance",
        "reduce",
    ),
    "amr": (
        "amr",
        "regrid",
        "refinement",
        "ref_ratio",
        "flux_register",
        "cf_interface",
        "multiblock",
        "substeps",
        "stride",
    ),
    "coupling": (
        "coupler",
        "coupled",
        "source",
        "fieldsolve",
        "solve_fields",
        "condensed",
        "schur",
    ),
    "elliptic": (
        "poisson",
        "elliptic",
        "mg",
        "schur",
        "krylov",
        "tensor",
        "epsilon",
        "fieldsolve",
        "field_solve",
        "solve",
        "newton",
        "potential",
    ),
    "runtime": (
        "runtime",
        "system",
        "program",
        "cache",
        "scheduler",
        "profil",
        "external",
        "module",
        "metadata",
        "facade",
        "config",
        "dynamic",
        "block_builder",
        "native",
        "capabilities",
    ),
    "numerics": (
        "riemann",
        "weno",
        "hll",
        "roe",
        "flux",
        "recon",
        "imex",
        "ssprk",
        "strang",
        "cfl",
        "diffusion",
        "splitting",
        "primitive",
        "projection",
        "rhs",
        "dt",
        "time",
        "multirate",
        "limiter",
        "wave_speed",
        "positivity",
    ),
    "physics": (
        "aux",
        "magnetic",
        "lorentz",
        "isothermal",
        "compressible",
        "fluid",
        "polar",
        "two_species",
        "board",
        "moments",
        "ap_limit",
        "vacuum",
        "bz",
        "te",
    ),
    "codegen": (
        "codegen",
        "dsl",
        "compile",
        "compiled",
        "aot",
        "jit",
        "operator",
        "ir",
        "generated",
        "loader",
    ),
    "validation": (
        "validation",
        "reference",
        "parity",
        "capabilities",
    ),
    "time": (
        "time",
        "imex",
        "ssprk",
        "strang",
        "euler",
        "bdf",
        "rk",
        "gmres",
        "multistage",
        "stride",
    ),
}

PYTHON_NAME_PATTERNS = CPP_NAME_PATTERNS | {
    "codegen": CPP_NAME_PATTERNS["codegen"]
    + (
        "dsl",
        "compile_cache",
        "module",
        "lib",
        "name_binding",
    ),
}

CPP_SMOKE_TARGETS = (
    "test_box2d",
    "test_reduce",
    "test_system_abstraction",
)

PYTHON_SMOKE_TESTS = (
    "python/tests/test_bindings.py",
    "python/tests/test_capabilities.py",
)


def normalize(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("./"):
        return cleaned[2:]
    return cleaned


def read_changed_files(path: Path) -> list[str]:
    return [normalize(line) for line in path.read_text().splitlines() if normalize(line)]


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


def parse_cpp_targets() -> list[str]:
    cmake = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    targets = set(re.findall(r"\bpops_add_test\(\s*([A-Za-z0-9_]+)", cmake))
    targets.update(re.findall(r"\badd_executable\(\s*([A-Za-z0-9_]+)", cmake))
    return sorted(t for t in targets if t.startswith("test_") and not t.startswith("test_mpi_"))


def list_python_tests() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "python/tests").glob("test_*.py"))


def select_by_name(names: Iterable[str], areas: Iterable[str], patterns: dict[str, tuple[str, ...]]) -> set[str]:
    wanted: set[str] = set()
    selected_areas = set(areas)
    for name in names:
        for area in selected_areas:
            if any(token in name for token in patterns.get(area, ())):
                wanted.add(name)
                break
    return wanted


def direct_cpp_targets(changed: Iterable[str], all_targets: set[str]) -> set[str]:
    targets: set[str] = set()
    for path in changed:
        if path.startswith("tests/test_") and path.endswith(".cpp"):
            target = Path(path).stem
            if target in all_targets:
                targets.add(target)
    return targets


def direct_python_tests(changed: Iterable[str], all_tests: set[str]) -> set[str]:
    return {path for path in changed if path in all_tests}


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


def plan_cpp(args: argparse.Namespace) -> int:
    changed = read_changed_files(Path(args.changed_files))
    all_targets = parse_cpp_targets()
    all_target_set = set(all_targets)

    full = args.force_all or force_full_from_changed(changed, CPP_BROAD_FILES, CPP_BROAD_PREFIXES)
    if not full and any(path.startswith("include/pops/") for path in changed):
        known_include = any(areas_for(path, CPP_PATH_AREAS) for path in changed if path.startswith("include/pops/"))
        full = not known_include

    selected: set[str] = set()
    areas: set[str] = set()
    if not full:
        selected.update(direct_cpp_targets(changed, all_target_set))
        for path in changed:
            areas.update(areas_for(path, CPP_PATH_AREAS))
        selected.update(select_by_name(all_targets, areas, CPP_NAME_PATTERNS))
        if selected:
            selected.update(t for t in CPP_SMOKE_TARGETS if t in all_target_set)
        elif not only_meta(changed):
            full = True

    if full or len(selected) > len(all_targets) * 0.75:
        mode = "all"
        targets = all_targets
        regex = ""
    else:
        mode = "subset" if selected else "none"
        targets = sorted(selected)
        regex = "^(" + "|".join(re.escape(t) for t in targets) + ")$" if targets else "$^"

    summary = f"{mode}: {len(targets)}/{len(all_targets)} C++ tests"
    print(summary)
    if areas:
        print("areas=" + ",".join(sorted(areas)))
    for target in targets:
        print(target)

    write_github_outputs(
        args.github_output,
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
    return 0


def plan_python(args: argparse.Namespace) -> int:
    changed = read_changed_files(Path(args.changed_files))
    all_tests = list_python_tests()
    all_test_set = set(all_tests)

    full = args.force_all or force_full_from_changed(changed, PYTHON_BROAD_FILES, PYTHON_BROAD_PREFIXES)
    if not full and any(path.startswith("python/pops/") for path in changed):
        known_python = any(areas_for(path, PYTHON_PATH_AREAS) for path in changed if path.startswith("python/pops/"))
        full = not known_python

    selected: set[str] = set()
    areas: set[str] = set()
    if not full:
        selected.update(direct_python_tests(changed, all_test_set))
        for path in changed:
            areas.update(areas_for(path, PYTHON_PATH_AREAS))
            areas.update(areas_for(path, CPP_PATH_AREAS))
        selected.update(select_by_name(all_tests, areas, PYTHON_NAME_PATTERNS))
        if selected:
            selected.update(t for t in PYTHON_SMOKE_TESTS if t in all_test_set)
        elif not only_meta(changed):
            full = True

    if full or len(selected) > len(all_tests) * 0.75:
        mode = "all"
        selected_tests = all_tests
    else:
        mode = "subset" if selected else "none"
        selected_tests = sorted(selected)

    sharded = shard(selected_tests, args.shard_index, args.shard_total)
    if args.tests_file:
        Path(args.tests_file).write_text("".join(f"{test}\n" for test in sharded), encoding="utf-8")

    summary = f"{mode}: {len(selected_tests)}/{len(all_tests)} Python test files"
    if args.shard_index is not None and args.shard_total is not None:
        summary += f" ({len(sharded)} in shard {args.shard_index}/{args.shard_total})"
    print(summary)
    if areas:
        print("areas=" + ",".join(sorted(areas)))
    for test in sharded:
        print(test)

    write_github_outputs(
        args.github_output,
        {
            "python_mode": mode,
            "python_count": str(len(selected_tests)),
            "python_total": str(len(all_tests)),
            "python_shard_count": str(len(sharded)),
            "python_areas": ",".join(sorted(areas)) if areas else "-",
            "python_summary": summary,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    cpp = sub.add_parser("cpp")
    cpp.add_argument("--changed-files", required=True)
    cpp.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    cpp.add_argument("--force-all", action="store_true")
    cpp.set_defaults(func=plan_cpp)

    py = sub.add_parser("python")
    py.add_argument("--changed-files", required=True)
    py.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    py.add_argument("--tests-file")
    py.add_argument("--shard-index", type=int)
    py.add_argument("--shard-total", type=int)
    py.add_argument("--force-all", action="store_true")
    py.set_defaults(func=plan_python)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
