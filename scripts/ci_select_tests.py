#!/usr/bin/env python3
"""Select affected tests for CI.

The policy is intentionally conservative. For the Python suite (``plan_python``) the
precedence, from most to least conservative, is:

(a) BROAD change -- any C++/bindings/CMake change (routed here as ``python`` too), a
    Python broad file (``pyproject.toml``, ``python/CMakeLists.txt``,
    ``python/pops/__init__.py``), or a ``python/bindings/`` file -> run ALL. Unchanged.
(b) direct test edit (a changed ``python/tests/test_*.py``) -> that test is selected
    directly, and its cross-test dependencies come along via the import closure
    (``ci_import_closure``); behaviour unchanged, now closure-aware.
(c) a changed ``python/pops/**`` file (not broad) -> the tests are chosen by the
    REVERSE import closure of the changed module (``ci_import_closure.impacted_tests``),
    UNION the existing smoke tests. This REPLACES the coarse name-token area heuristic
    for pops source changes.
(d) a changed ``python/pops`` file whose module is NOT on the import graph (a brand-new
    file the graph has never seen, or an unparseable one) -> fail-safe to ALL.
(e) the ``>75% selected -> all`` rule and the ``unknown non-meta path -> all`` rule are
    kept unchanged as the final safety nets.

``plan_cpp`` is untouched: it keeps the area heuristic (no C++ import-closure yet).

The whole module is stdlib-only so it runs on the bare runner interpreter before any
``pip install`` in the ``Select affected tests`` step.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_import_closure  # noqa: E402  (sibling stdlib-only module, same scripts/ dir)


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
    # The scraper is textual, so it also sees targets registered inside if(POPS_USE_MPI) blocks.
    # Those only exist in the ci-mpi build (the serial gate would hit `ninja: unknown target`);
    # by convention every MPI-only test carries an `mpi` NAME SEGMENT (prefix test_mpi_* or infix
    # like test_amr_regrid_mpi_parity), so drop any target with such a segment -- they are covered
    # by the MPI job in full mode.
    return sorted(t for t in targets
                  if t.startswith("test_") and "mpi" not in t.split("_"))


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

    # (a) BROAD -> all. Any C++/bindings/CMake change is routed to this job as `python`
    # too, so a broad Python file or a bindings change forces the whole suite.
    full = args.force_all or force_full_from_changed(changed, PYTHON_BROAD_FILES, PYTHON_BROAD_PREFIXES)
    reason = "force-all" if args.force_all else ("broad-file" if full else "")

    selected: set[str] = set()
    areas: set[str] = set()
    reasons: set[str] = set()
    if reason:
        reasons.add(reason)

    if not full:
        # (b) direct test edits -> the touched tests, closure-aware.
        direct = direct_python_tests(changed, all_test_set)
        if direct:
            selected.update(direct)
            reasons.add("direct-test")

        # (c) pops source changes -> the REVERSE import closure of the changed module,
        # union the smoke tests. (d) an off-graph pops file (new/unparseable) -> ALL.
        pops_changed = [p for p in changed if p.startswith("python/pops/") and p.endswith(".py")]
        if pops_changed:
            try:
                closure = ci_import_closure.impacted_tests(pops_changed, repo_root=ROOT)
            except ci_import_closure.OffGraphChange:
                full = True
                reasons.add("off-graph-pops-file")
            else:
                selected.update(t for t in closure if t in all_test_set)
                reasons.add("import-closure")

        # A non-.py pops change (e.g. a data/asset file under python/pops) has no module
        # to close over; keep the conservative old behaviour of running ALL for it.
        if not full and any(
            p.startswith("python/pops/") and not p.endswith(".py") for p in changed
        ):
            full = True
            reasons.add("non-py-pops-file")

        # Cross-test closure over ALL currently-selected tests (both directions): a
        # selected test pulls the helpers it imports, and any test importing a selected
        # helper is pulled in too. Applies to the direct edits and closure hits alike.
        if selected:
            ci_import_closure._close_cross_test(selected, _test_to_test())
            selected.update(t for t in PYTHON_SMOKE_TESTS if t in all_test_set)

        # (e) safety net: a non-meta change that resolved to nothing runs ALL rather than
        # silently dropping coverage (matches the historical area-heuristic fallback).
        if not full and not selected and not only_meta(changed):
            full = True
            reasons.add("unknown-path")

    # (e) safety net: >75% selected is not worth the bookkeeping -- run ALL.
    if full or len(selected) > len(all_tests) * 0.75:
        if not full:
            reasons.add(">75%-all")
        mode = "all"
        selected_tests = all_tests
    else:
        mode = "subset" if selected else "none"
        selected_tests = sorted(selected)

    sharded = shard(selected_tests, args.shard_index, args.shard_total)
    if args.tests_file:
        Path(args.tests_file).write_text("".join(f"{test}\n" for test in sharded), encoding="utf-8")

    why = ",".join(sorted(reasons)) if reasons else "meta-only"
    summary = f"{mode}: {len(selected_tests)}/{len(all_tests)} Python test files [why: {why}]"
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
            "python_why": why,
            "python_summary": summary,
        },
    )
    return 0


def _test_to_test() -> dict[str, set[str]]:
    """Return only the cross-test edge map (the second half of ``test_imports``)."""
    _, edges = ci_import_closure.test_imports(repo_root=ROOT)
    return edges


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
