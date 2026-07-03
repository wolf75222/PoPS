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
import ci_include_graph  # noqa: E402
import ci_shard_binpack  # noqa: E402


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

# The ADC-547 compliance matrix (label "compliance") is the cross-cutting regression net, so a
# change to any core route area (runtime / physics / elliptic / amr) pulls it in.
AREA_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "amr": ("amr", "mesh", "compliance"),
    "bindings": ("bindings", "runtime", "native_loader"),
    "codegen": ("codegen", "native_loader", "compiler", "bindings"),
    "coupling": ("coupling", "runtime", "elliptic", "amr", "physics"),
    "elliptic": ("elliptic", "solvers", "compliance"),
    "io": ("io", "runtime"),
    "mesh": ("mesh", "amr"),
    "native_loader": ("native_loader", "codegen", "compiler"),
    "numerics": ("numerics", "elliptic", "solvers", "time"),
    "physics": ("physics", "numerics", "compliance"),
    "problem": ("problem", "runtime"),
    "runtime": ("runtime", "bindings", "native_loader", "compliance"),
    "solvers": ("solvers", "elliptic"),
    "time": ("time", "numerics", "solvers"),
    "validation": ("validation", "physics", "runtime"),
}

# A changed file is HEADER-IMPACT ELIGIBLE only if it is a project header under this prefix.
# These are the only paths the include-graph impact selection reasons about; anything else in
# the changeset makes the whole change fall through to the coarser label logic or force-full.
CPP_INCLUDE_PREFIX = "include/pops/"
CPP_HEADER_SUFFIXES = (".hpp", ".h", ".hh", ".hxx", ".inc", ".ipp", ".tpp")

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


def is_cpp_header(path: str) -> bool:
    """True if ``path`` is a project header under ``include/pops/`` (an impact-graph node)."""
    return path.startswith(CPP_INCLUDE_PREFIX) and path.endswith(CPP_HEADER_SUFFIXES)


def changed_header_nodes(changed: Iterable[str]) -> set[str]:
    """Map the changed ``include/pops/`` headers to graph nodes (``pops/...`` identifiers)."""
    return {path[len("include/") :] for path in changed if is_cpp_header(path)}


def header_impact_eligible(changed: Iterable[str]) -> bool:
    """True iff EVERY changed file is a project header under ``include/pops/``.

    Header-impact selection is scoped to pure-header changesets: a single non-header C++
    concern (cmake, a ``.cpp``, tests support, a python binding) means the include graph does
    not capture the whole blast radius, so the caller keeps the coarser label logic instead.
    This predicate is only consulted after the broad-file force-full guards, so ``core`` /
    ``parallel`` header changes never reach it.
    """
    changed = list(changed)
    return bool(changed) and all(is_cpp_header(path) for path in changed)


def select_cpp_by_include_impact(
    suites: Iterable[dict],
    changed_headers: set[str],
    reasons: dict[str, set[str]],
) -> tuple[set[str], list[str]]:
    """Select the suites whose source include-closure intersects ``changed_headers``.

    Returns ``(selected, full_reasons)``. ``full_reasons`` is non-empty when the include graph
    forces a FULL selection instead of a subset -- either a changed header is a global includer
    (in the transitive closure of the heavy shared TUs / seams / emitter / cpp support, so it is
    compiled into or linked by every target) or an anomaly is hit (a changed header absent from
    the tree, or a suite source that cannot be read). Fail-open: any doubt escalates to FULL.
    """
    try:
        global_closure = ci_include_graph.global_includer_closure()
    except ci_include_graph.GraphError as exc:
        return set(), [f"include-graph-unreadable:{exc}"]

    # SOUNDNESS: a changed header reachable from the heavy shared TUs / seams / emitter / cpp
    # support is linked into effectively every test -> select ALL suites.
    global_hits = sorted(changed_headers & global_closure)
    if global_hits:
        return set(), ["header-in-global-includer-closure:" + ",".join(global_hits)]

    # FAIL-OPEN: a changed header that does not exist on disk cannot be reasoned about.
    missing = sorted(h for h in changed_headers if not ci_include_graph.header_exists(h))
    if missing:
        return set(), ["changed-header-not-in-tree:" + ",".join(missing)]

    selected: set[str] = set()
    for suite in suites:
        closure: set[str] = set()
        try:
            for source in suite["sources"]:
                closure |= ci_include_graph.source_closure(source)
        except ci_include_graph.GraphError as exc:
            return set(), [f"suite-source-missing:{suite['name']}:{exc}"]
        hits = sorted(closure & changed_headers)
        if hits:
            selected.add(suite["name"])
            add_reason(reasons, suite["name"], "include-impact:" + ",".join(hits))
    return selected, []


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
    """Duration-balanced binpacking of the selected files onto one shard (ADC-623).

    When ``index``/``total`` are omitted (the unsharded query used by the tests and by the
    "has this shard any work" check) the full selected list is returned unchanged, EXCEPT
    the compile-cache files that run in their own dedicated CI job -- excluding them here too
    keeps the unsharded view consistent with what the shards actually run.

    With a shard requested, ``ci_shard_binpack.shard_files`` removes the excluded files,
    greedily LPT-packs the remainder by measured duration, verifies the partition is an exact
    cover of ``items`` (fails loudly otherwise), and returns this shard's file list.
    """
    excluded = set(ci_shard_binpack.EXCLUDED_FROM_SHARDS)
    if index is None or total is None:
        return [item for item in items if item not in excluded]
    if total <= 0 or index < 0 or index >= total:
        raise SystemExit(f"invalid shard {index}/{total}")
    try:
        return ci_shard_binpack.shard_files(items, index, total)
    except ci_shard_binpack.PartitionError as exc:
        raise SystemExit(f"shard partition invariant violated: {exc}")


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

    selected: set[str] = set()
    areas: set[str] = set()
    reasons: dict[str, set[str]] = {}
    # ADC-629: a pure ``include/pops/`` header changeset (after the broad guards, so never
    # core/parallel) is narrowed by the include graph -- each suite is selected iff its source
    # include-closure reaches a changed header. A global-includer header or any graph anomaly
    # fails open to FULL; the coarser label logic below handles every non-pure-header change.
    if not full and header_impact_eligible(changed):
        changed_headers = changed_header_nodes(changed)
        impacted, impact_full = select_cpp_by_include_impact(suites, changed_headers, reasons)
        if impact_full:
            full = True
            full_reasons.extend(impact_full)
        else:
            # ``impacted`` may be empty: a pure-header change that reaches NO suite closure is a
            # header no test includes (and not a global includer). Selecting nothing would be
            # sound, but the smoke backstop keeps a conservative floor either way.
            selected.update(impacted)
            for target in CPP_SMOKE_TARGETS:
                if target in all_target_set:
                    selected.add(target)
                    add_reason(reasons, target, "smoke-backstop")
    elif not full and any(path.startswith("include/pops/") for path in changed):
        # Mixed change touching headers: the include graph does not capture the non-header
        # blast radius, so fall through to labels but keep the unknown-path force-full guard.
        known_include = any(areas_for(path, CPP_PATH_AREAS) for path in changed if path.startswith("include/pops/"))
        if not known_include:
            full = True
            full_reasons.append("unknown-cpp-public-api-path")

    if not full and not selected and not header_impact_eligible(changed):
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


class PythonSelection:
    """The result of the Python test selection, independent of any sharding."""

    def __init__(
        self,
        changed: list[str],
        all_tests: list[str],
        selected_tests: list[str],
        mode: str,
        areas: set[str],
        why: set[str],
        full_reasons: list[str],
        reasons: dict[str, set[str]],
    ) -> None:
        self.changed = changed
        self.all_tests = all_tests
        self.selected_tests = selected_tests
        self.mode = mode
        self.areas = areas
        self.why = why
        self.full_reasons = full_reasons
        self.reasons = reasons


def compute_python_selection(changed_files: str, force_all: bool) -> PythonSelection:
    """Compute the selected Python test files (the shard-independent selection).

    Shared by ``plan_python`` (which then shards + reports) and the ``verify`` subcommand
    (which reconstructs every shard and asserts an exact cover). Keeping selection in one
    place means the exactness check verifies the SAME set the shards run.
    """
    changed = read_changed_files(Path(changed_files))
    manifest = load_manifest()
    suites = manifest_python_suites(manifest)
    all_tests = sorted({test for suite in suites for test in suite["files"]})
    all_test_set = set(all_tests)

    if not all_tests:
        raise SystemExit("no Python suites found in tests/test_manifest.toml")

    full_reasons: list[str] = []
    why: set[str] = set()
    if force_all:
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

    return PythonSelection(
        changed=changed,
        all_tests=all_tests,
        selected_tests=selected_tests,
        mode=mode,
        areas=areas,
        why=why,
        full_reasons=full_reasons,
        reasons=reasons,
    )


def plan_python(args: argparse.Namespace) -> int:
    sel = compute_python_selection(args.changed_files, args.force_all)
    changed = sel.changed
    all_tests = sel.all_tests
    selected_tests = sel.selected_tests
    mode = sel.mode
    areas = sel.areas
    why = sel.why
    full_reasons = sel.full_reasons
    reasons = sel.reasons

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


def plan_verify(args: argparse.Namespace) -> int:
    """Reconstruct every shard and fail loudly unless they exactly cover the selection.

    The safety net for the duration binpacking (ADC-623): recompute the SAME selection the
    shard jobs use, pack it across ``--shard-total`` shards, and assert the union of all
    shards plus the excluded (dedicated-job) files equals the selection exactly -- no test
    silently dropped, none duplicated. Runs as an explicit CI step so a broken partition
    fails the gate rather than quietly under-testing.
    """
    sel = compute_python_selection(args.changed_files, args.force_all)
    selected = sel.selected_tests
    total = args.shard_total
    excluded = set(ci_shard_binpack.EXCLUDED_FROM_SHARDS)
    shardable = [t for t in selected if t not in excluded]
    shards = ci_shard_binpack.assign_shards(shardable, total)
    try:
        ci_shard_binpack.verify_partition(selected, shards, ci_shard_binpack.EXCLUDED_FROM_SHARDS)
    except ci_shard_binpack.PartitionError as exc:
        print(f"::error::shard partition is not an exact cover: {exc}")
        raise SystemExit(1)

    excluded_present = sorted(set(selected) & excluded)
    loads = [round(sum(1 for _ in shard), 0) for shard in shards]
    print(
        f"partition OK: {sel.mode} selection of {len(selected)} files "
        f"({len(shardable)} sharded across {total} + {len(excluded_present)} in the dedicated "
        f"compile-cache job) is an exact cover"
    )
    print("per-shard file counts: " + ", ".join(str(int(n)) for n in loads))
    if excluded_present:
        print("dedicated-job files (excluded from shards): " + ", ".join(excluded_present))
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

    verify = sub.add_parser("verify")
    verify.add_argument("--changed-files", required=True)
    verify.add_argument("--shard-total", type=int, required=True)
    verify.add_argument("--force-all", action="store_true")
    verify.set_defaults(func=plan_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
