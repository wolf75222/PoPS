#!/usr/bin/env python3
"""Select affected tests for CI.

The policy is intentionally conservative. C++ selection is COMPOSITIONAL per file (ADC-646):
each changed file contributes its own impact and the selection is their union -- an
``include/pops/`` header adds its include-closure suites, a ``src/runtime`` translation
unit adds the test targets that compile it, a pybind adapter adds the bindings suites, and a
``python/pops/codegen`` emitter adds the
codegen / native-loader group, a ``tests/cpp`` source adds its own target, and docs / non-codegen
``python/pops`` / ``tests/python`` add nothing. A global-includer or missing header, or any
unmapped build input (cmake / workflows / scripts / CMakeLists / the manifest), fails safe to ALL.
Python selection is manifest-driven with a static import-closure for ``python/pops/**`` changes so
pure Python edits can run only the tests that import the changed module.

The module is stdlib-only and runs before any ``pip install`` in CI.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_import_closure  # noqa: E402
import ci_include_graph  # noqa: E402
import ci_shard_binpack  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/test_manifest.toml"
CPP_DURATIONS_JSON = ROOT / "tests/cpp/test_durations.json"
CPP_BUILD_DURATIONS_JSON = ROOT / "tests/cpp/build_durations.json"
# CTest's catalog is the sum of per-target wall times. The preset can execute four
# independent tests concurrently, so convert that aggregate to critical-path seconds
# before combining it with the build catalog's already-normalized shard-wall estimate.
CPP_CTEST_PARALLEL_JOBS = 4.0


CPP_BROAD_FILES = {
    ".github/workflows/ci.yml",
    "CMakeLists.txt",
    "CMakePresets.json",
    "scripts/ci_include_graph.py",
    "scripts/ci_route_mode.py",
    "scripts/ci_select_tests.py",
    "scripts/ci_shard_binpack.py",
    "tests/CMakeLists.txt",
    "tests/cpp/build_durations.json",
    "tests/cpp/test_durations.json",
    "src/CMakeLists.txt",
    "tests/cpp/test_sources.cmake",
    "tests/test_manifest.toml",
}

CPP_BROAD_PREFIXES = (
    ".github/actions/setup-kokkos/",
    "cmake/",
    "include/pops/core/",
    "include/pops/parallel/",
    "tests/cpp/support/",
)

PYTHON_BROAD_FILES = {
    ".github/workflows/ci.yml",
    "pyproject.toml",
    "python/CMakeLists.txt",
    "scripts/ci_import_closure.py",
    "scripts/ci_route_mode.py",
    "scripts/ci_select_tests.py",
    "scripts/ci_shard_binpack.py",
    "src/CMakeLists.txt",
    "python/pops/__init__.py",
    "tests/python/conftest.py",
    "tests/python/test_durations.json",
    "tests/test_manifest.toml",
}

PYTHON_BROAD_PREFIXES = (
    ".github/actions/setup-kokkos/",
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
    (("src/runtime/amr/", "src/runtime/builders/amr/"), ("amr", "runtime", "codegen")),
    (("src/runtime/system/",), ("runtime", "physics", "codegen")),
    (("src/runtime/builders/",), ("runtime", "physics", "amr", "codegen")),
    (("python/bindings/core/",), ("runtime", "physics", "amr", "codegen")),
    (("scripts/gen_solver_kernel.py",), ("codegen", "elliptic")),
)

PYTHON_PATH_AREAS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("python/pops/_report.py", "python/pops/_inspect.py"), ("reporting",)),
    (("python/pops/boundary/",), ("boundary", "mesh", "numerics")),
    (("python/pops/domain/",), ("domain", "mesh", "problem")),
    (("python/pops/fields/",), ("fields", "physics", "elliptic", "runtime")),
    (("python/pops/initial/",), ("initial", "problem", "runtime")),
    (("python/pops/identity/",), ("identity", "codegen", "runtime")),
    (("python/pops/mesh/",), ("mesh", "amr")),
    (("python/pops/runtime/amr/",), ("amr", "runtime")),
    (("python/pops/runtime/",), ("runtime",)),
    (("python/pops/solvers/", "python/pops/linalg/"), ("elliptic",)),
    (("python/pops/codegen/", "python/pops/_ir/", "python/pops/lib/"), ("codegen",)),
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
    "boundary": ("boundary", "mesh", "numerics"),
    "codegen": ("codegen", "native_loader", "compiler", "bindings"),
    "coupling": ("coupling", "runtime", "elliptic", "amr", "physics"),
    "domain": ("domain", "mesh", "problem"),
    "elliptic": ("elliptic", "solvers", "compliance"),
    "examples": ("examples", "runtime", "io", "time", "amr", "physics"),
    "fields": ("fields", "physics", "elliptic", "runtime"),
    "io": ("io", "runtime"),
    "identity": ("identity", "codegen", "runtime"),
    "initial": ("initial", "problem", "runtime"),
    "mesh": ("mesh", "amr"),
    "moments": ("moments", "physics", "numerics"),
    "native_loader": ("native_loader", "codegen", "compiler"),
    "numerics": ("numerics", "elliptic", "solvers", "time"),
    "output": ("output", "io", "runtime"),
    "physics": ("physics", "numerics", "compliance"),
    "problem": ("problem", "runtime"),
    "reporting": ("reporting", "descriptors", "problem", "runtime"),
    "runtime": ("runtime", "bindings", "native_loader", "compliance"),
    "solvers": ("solvers", "elliptic"),
    "time": ("time", "numerics", "solvers"),
    "validation": ("validation", "physics", "runtime"),
}

# A changed file is a HEADER-IMPACT node only if it is a project header under this prefix.
# ADC-646: header impact is now assessed PER FILE (compositional union), so a header no longer
# needs the WHOLE changeset to be header-only -- its include-closure targets join the union.
CPP_INCLUDE_PREFIX = "include/pops/"
CPP_HEADER_SUFFIXES = (".hpp", ".h", ".hh", ".hxx", ".inc", ".ipp", ".tpp")

# ADC-646 per-file C++ impact classification.
# A ``src/runtime/**`` C++ translation unit is compiled into the shared runtime OBJECT libs
# (``pops_runtime_system`` / ``pops_runtime_amr``) that most test targets link. The precise
# per-target linkage is recovered from the central source manifest and test consumers. The
# ``.cpp``/``.hpp`` suffixes below are compiled units; any other runtime artifact (CMake fragment
# or ``.cpp.in`` seam template) is a build input and fails open to ALL.
CPP_RUNTIME_PREFIX = "src/runtime/"
CPP_RUNTIME_TU_SUFFIXES = (".cpp", ".hpp", ".h", ".hh", ".hxx")
# ``python/bindings`` now contains only module/init adapters. These feed only ``_pops`` and map to
# the bindings label group; they are never treated as runtime implementation ownership.
CPP_BINDING_PREFIX = "python/bindings/"
CPP_BINDING_TU_SUFFIXES = (".cpp", ".hpp", ".h", ".hh", ".hxx")
CPP_BINDING_AREAS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("python/bindings/core/",), ("runtime", "physics", "amr", "codegen")),
)
# ADC-646: a DSL codegen emitter under ``python/pops/codegen/**`` only changes the C++ that is
# EMITTED into generated translation units (the native_loader / compiled-model path), so it maps
# to the ``codegen`` label group ONLY -- never the whole physics/runtime blast radius.
CPP_CODEGEN_PREFIX = "python/pops/codegen/"
CPP_CODEGEN_AREAS = ("codegen",)

# ADC-646: paths with ZERO C++ test impact -- a change limited to these never selects a C++ suite.
CPP_ZERO_IMPACT_PREFIXES = (
    "docs/",
    "tutorials/",
    "tests/python/",
    "python/pops/",  # non-codegen pops python; codegen is handled before this prefix is tested
)
CPP_ZERO_IMPACT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "SECURITY.md",
    ".gitignore",
}

CPP_SMOKE_TARGETS = (
    "test_box2d",
    "test_reduce",
    "test_system_abstraction",
)

PYTHON_SMOKE_TESTS = (
    "tests/python/integration/bindings/test_m1_scalar_advection_pipeline.py",
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


def manifest_cpp_suites(manifest: dict, *, include_mpi: bool = False) -> list[dict]:
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
        if not include_mpi and (
            "mpi" in labels or suite.get("mpi_nproc") or "mpi" in name.split("_")
        ):
            continue
        sources = [normalize(str(source)) for source in suite.get("sources", [])]
        if not sources:
            raise SystemExit(f"C++ suite {name} has no sources in tests/test_manifest.toml")
        ranks_by_field: dict[str, tuple[int, ...]] = {}
        for field in ("mpi_nproc", "mpi_variants"):
            raw_ranks = suite.get(field, [])
            if not isinstance(raw_ranks, list):
                raise SystemExit(f"C++ suite {name} has invalid {field}; expected a TOML array")
            ranks = tuple(raw_ranks)
            if any(type(rank) is not int or rank <= 0 for rank in ranks):
                raise SystemExit(
                    f"C++ suite {name} has invalid {field}; ranks must be positive integers"
                )
            if len(ranks) != len(set(ranks)):
                raise SystemExit(f"C++ suite {name} has duplicate ranks in {field}")
            if ranks != tuple(sorted(ranks)):
                raise SystemExit(f"C++ suite {name} must sort {field} in ascending order")
            ranks_by_field[field] = ranks
        mpi_nproc = ranks_by_field["mpi_nproc"]
        mpi_variants = ranks_by_field["mpi_variants"]
        if ("mpi" in labels) != bool(mpi_nproc):
            raise SystemExit(
                f"C++ suite {name} must pair its mpi label with an exact mpi_nproc rank set"
            )
        if mpi_variants and ("mpi" in labels or mpi_nproc):
            raise SystemExit(
                f"C++ suite {name} cannot mix mpi_variants with an MPI-only label/mpi_nproc"
            )
        suites.append(
            {
                "name": name,
                "labels": labels,
                "sources": sources,
                "mpi_nproc": mpi_nproc,
                "mpi_variants": mpi_variants,
            }
        )
    return sorted(suites, key=lambda item: item["name"])


def cpp_targets_with_label(manifest: dict, label: str) -> list[str]:
    """Return the exact manifest-owned C++ targets required by ``label``.

    Dedicated backend jobs use this instead of duplicating target names in workflow YAML. MPI
    targets are intentionally excluded from the ordinary serial selector but remain first-class
    manifest suites for this label projection.  The MPI projection additionally includes ordinary
    serial executables declaring ``mpi_variants``: CTest launches those same binaries under MPI,
    so the dedicated job must build them even though their primary suite is not MPI-only.
    """
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise SystemExit("C++ suite label must contain only letters, digits, '.', '_' or '-'")
    targets = sorted(
        suite["name"]
        for suite in manifest_cpp_suites(manifest, include_mpi=True)
        if label in suite["labels"] or (label == "mpi" and suite["mpi_variants"])
    )
    if not targets:
        raise SystemExit(
            f"no C++ suites carry label {label!r} in tests/test_manifest.toml"
        )
    return targets


def cpp_mpi_ctest_count(manifest: dict) -> int:
    """Return the exact number of manifest-owned ``ctest -L mpi`` launches."""
    suites = manifest_cpp_suites(manifest, include_mpi=True)
    return sum(
        len(suite["mpi_nproc"]) + len(suite["mpi_variants"])
        for suite in suites
    )


def manifest_python_mpi_entrypoints(manifest: dict) -> list[dict]:
    """Return the exact Python scripts that must run under a real MPI launcher.

    Ordinary ``[[python.suite]]`` rows remain pytest collection units.  A suite may additionally
    declare ``mpi_entrypoints`` for script-style contracts whose ``__main__`` path turns internal
    check failures into a non-zero process status.  Keeping ranks and paths beside the owning suite
    makes the manifest -- not workflow YAML -- the single authority for distributed Python tests.
    """
    entries: list[dict] = []
    seen_paths: set[str] = set()
    for suite in manifest.get("python", {}).get("suite", []):
        name = str(suite.get("name", ""))
        suite_path = normalize(str(suite.get("path", ""))).rstrip("/")
        labels = {str(label) for label in suite.get("labels", [])}
        raw_entries = suite.get("mpi_entrypoints", [])
        if not isinstance(raw_entries, list):
            raise SystemExit(
                f"Python suite {name or '<unnamed>'} has invalid mpi_entrypoints; "
                "expected a TOML array"
            )
        if raw_entries and "mpi" not in labels:
            raise SystemExit(
                f"Python suite {name or '<unnamed>'} declares mpi_entrypoints without an mpi label"
            )
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise SystemExit(
                    f"Python suite {name or '<unnamed>'} has a non-table mpi_entrypoint"
                )
            path = normalize(str(raw.get("path", "")))
            nproc = raw.get("nproc")
            candidate = Path(path)
            if (
                not path
                or candidate.is_absolute()
                or ".." in candidate.parts
                or "\t" in path
                or "\n" in path
                or candidate.suffix != ".py"
            ):
                raise SystemExit(
                    f"Python suite {name or '<unnamed>'} has invalid MPI entrypoint path {path!r}"
                )
            if not suite_path or not path.startswith(suite_path + "/"):
                raise SystemExit(
                    f"Python MPI entrypoint {path!r} is outside owning suite {suite_path!r}"
                )
            if type(nproc) is not int or nproc < 2:
                raise SystemExit(
                    f"Python MPI entrypoint {path!r} needs an exact nproc integer >= 2"
                )
            if not (ROOT / path).is_file():
                raise SystemExit(f"Python MPI entrypoint does not exist: {path}")
            if path in seen_paths:
                raise SystemExit(f"duplicate Python MPI entrypoint: {path}")
            seen_paths.add(path)
            entries.append({"suite": name, "path": path, "nproc": nproc})
    return sorted(entries, key=lambda item: (item["path"], item["nproc"]))


def manifest_python_suites(manifest: dict) -> list[dict]:
    # CI selects the repository snapshot, not arbitrary untracked scratch files in a developer
    # checkout.  This also keeps local plans stable when editors create suffixed copies beside a
    # test.  A source archive without Git falls back to its complete filesystem tree.
    try:
        tracked_output = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "tests/python"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        tracked = {
            item.decode("utf-8") for item in tracked_output.split(b"\0") if item
        }
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        tracked = None
    mpi_entrypoint_paths = {
        entry["path"] for entry in manifest_python_mpi_entrypoints(manifest)
    }
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
        # A suite path owns its full subtree, matching the manifest coverage fence.  Using
        # ``glob`` here silently omitted nested families such as unit/mesh/amr even though the
        # manifest correctly declared their parent suite.
        files = sorted(
            str(p.relative_to(ROOT))
            for p in suite_path.rglob("test_*.py")
            if (tracked is None or str(p.relative_to(ROOT)) in tracked)
            and str(p.relative_to(ROOT)) not in mpi_entrypoint_paths
        )
        if not files:
            raise SystemExit(f"Python suite {name} has no test_*.py files under {path}")
        suites.append({"name": name, "path": path, "labels": labels, "files": files})
    return sorted(suites, key=lambda item: item["name"])


def direct_python_tests(changed: Iterable[str], all_tests: set[str]) -> set[str]:
    return {path for path in changed if path in all_tests}


def is_cpp_header(path: str) -> bool:
    """True if ``path`` is a project header under ``include/pops/`` (an impact-graph node)."""
    return path.startswith(CPP_INCLUDE_PREFIX) and path.endswith(CPP_HEADER_SUFFIXES)


def is_cpp_binding_tu(path: str) -> bool:
    """True if ``path`` is a compiled pybind module/init adapter."""
    return path.startswith(CPP_BINDING_PREFIX) and path.endswith(CPP_BINDING_TU_SUFFIXES)


def is_cpp_runtime_tu(path: str) -> bool:
    """True if ``path`` is a compiled ``src/runtime/**`` source or private header."""
    return path.startswith(CPP_RUNTIME_PREFIX) and path.endswith(CPP_RUNTIME_TU_SUFFIXES)


# src/CMakeLists.txt is the target-source map; tests/CMakeLists.txt owns only consumers.
TESTS_CMAKE = ROOT / "tests" / "CMakeLists.txt"
RUNTIME_CMAKE = ROOT / "src" / "CMakeLists.txt"
# The heavy runtime TUs are compiled ONCE into these OBJECT libs (ADC-336 / ADC-632 / ADC-335)
# and spliced into every consuming test target. A change to a TU in one of them impacts exactly
# that lib's consumers, so we read the central source list and the test consumer list together.
_RUNTIME_OBJECT_LIBS = ("pops_runtime_system", "pops_runtime_amr")


def _cmake_object_lib_sources(text: str, libname: str) -> set[str]:
    """Repo-relative native sources in the central object-library source manifest."""
    sources: set[str] = set()
    source_var = {
        "pops_runtime_system": "POPS_RUNTIME_SYSTEM_SOURCES",
        "pops_runtime_amr": "POPS_RUNTIME_AMR_SOURCES",
    }[libname]
    match = re.search(r"set\(\s*" + source_var + r"\b(.*?)\)", text, re.DOTALL)
    if match:
        for hit in re.finditer(r"\b(runtime/[^\s)]+\.(?:cpp|hpp|h|hh|hxx))", match.group(1)):
            sources.add("src/" + hit.group(1))
    return sources


def _cmake_object_lib_consumers(text: str, libname: str) -> set[str]:
    """Test-target NAMEs whose ``pops_add_gtest_suite(...)`` call links ``libname``.

    The consumers reference the OBJECT lib through ``EXTRA_LIBS``; the serial-target filter is
    applied by the caller (an MPI-only consumer never reaches the serial selection).
    """
    consumers: set[str] = set()
    for body in re.findall(r"pops_add_gtest_suite\((.*?)\)", text, re.DOTALL):
        if libname not in body:
            continue
        name = re.search(r"\bNAME\s+(\S+)", body)
        if name:
            consumers.add(name.group(1))
    return consumers


_runtime_map_cache: dict[str, tuple[dict[str, set[str]], dict[str, set[str]]]] = {}


def _runtime_object_lib_map() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return central ``(lib_sources, lib_consumers)`` maps (memoized).

    ``lib_sources[lib]`` is the set of ``src/runtime/**`` ``.cpp``/``.hpp`` compiled into the
    OBJECT lib; ``lib_consumers[lib]`` is the set of test-target names that link it. A change to
    ``tests/CMakeLists.txt`` is a broad-file force-all (``CPP_BROAD_FILES``), so a stale parse can
    never under-select on a changeset that edited the map.
    """
    key = str(RUNTIME_CMAKE) + "\0" + str(TESTS_CMAKE)
    if key not in _runtime_map_cache:
        runtime_text = (
            RUNTIME_CMAKE.read_text(encoding="utf-8", errors="ignore")
            if RUNTIME_CMAKE.is_file()
            else ""
        )
        tests_text = (
            TESTS_CMAKE.read_text(encoding="utf-8", errors="ignore")
            if TESTS_CMAKE.is_file()
            else ""
        )
        sources = {lib: _cmake_object_lib_sources(runtime_text, lib) for lib in _RUNTIME_OBJECT_LIBS}
        consumers = {lib: _cmake_object_lib_consumers(tests_text, lib) for lib in _RUNTIME_OBJECT_LIBS}
        _runtime_map_cache[key] = (sources, consumers)
    return _runtime_map_cache[key]


def _runtime_header_included_by(path: str, lib_sources: set[str]) -> bool:
    """True if a private runtime header is ``#include``d by a source in ``lib_sources``.

    Runtime TUs include their private headers with quoted RELATIVE paths (e.g. ``system_impl.hpp``
    next to ``system_fields.cpp``), so a change to that header impacts the same OBJECT lib as the
    TUs. Matched by basename against each TU's quoted includes -- best-effort source parse, safe:
    a miss is treated as an unregistered runtime input and fails open to all tests.
    """
    target_base = Path(path).name
    for source in lib_sources:
        if not source.endswith(".cpp"):
            continue
        src_path = ROOT / source
        if not src_path.is_file():
            continue
        text = src_path.read_text(encoding="utf-8", errors="ignore")
        for quoted in re.finditer(r'#\s*include\s*"([^"]+)"', text):
            if Path(quoted.group(1)).name == target_base:
                return True
    return False


def runtime_tu_targets(path: str, all_target_set: set[str]) -> tuple[set[str], list[str]]:
    """Map a ``src/runtime/**`` C++ file to serial tests consuming its object library.

    Resolves ``path`` to the runtime OBJECT lib(s) it belongs to -- either it IS one of the lib's
    listed ``.cpp``/``.hpp`` sources, or (for a private header) it is ``#include``d by one of the
    lib's ``.cpp`` TUs -- and returns that lib's serial consumer targets. Returns
    ``(targets, matched_libs)``. An empty ``matched_libs`` means the source is missing from the
    central manifest and must fail open rather than silently under-select.
    """
    lib_sources, lib_consumers = _runtime_object_lib_map()
    matched: list[str] = []
    targets: set[str] = set()
    for lib in _RUNTIME_OBJECT_LIBS:
        sources = lib_sources[lib]
        belongs = path in sources or (
            path.endswith((".hpp", ".h", ".hh", ".hxx")) and _runtime_header_included_by(path, sources)
        )
        if belongs:
            matched.append(lib)
            targets |= {t for t in lib_consumers[lib] if t in all_target_set}
    return targets, matched


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
        raise SystemExit(f"shard partition invariant violated: {exc}") from exc


def cpp_target_weights(targets: list[str]) -> dict[str, float]:
    """Return modeled shard-wall seconds for build then CTest of each target.

    ``build_durations.json`` is deliberately separate from CTest timings.  Its heavy rows are
    cold-CI measurements of the Ninja ``pops_heavy_test_tu=1`` serial pool; its light rows are the
    measured parallel-share floor.  Treating every target as the old constant 100 CPU-seconds hid
    five-minute template TUs and placed two of them on the same shard.  Both catalogs are exact:
    adding a manifest target without seeding build and test costs fails before any build starts.
    """
    measured_build_seconds = ci_shard_binpack.load_durations(CPP_BUILD_DURATIONS_JSON)
    measured_test_seconds = ci_shard_binpack.load_durations(CPP_DURATIONS_JSON)
    missing_weights = sorted(
        set(targets) - (measured_build_seconds.keys() & measured_test_seconds.keys())
    )
    if missing_weights:
        raise SystemExit(
            "C++ build/test duration catalogs are missing selected targets: "
            + ", ".join(missing_weights)
        )
    return {
        target: measured_build_seconds[target]
        + measured_test_seconds[target] / CPP_CTEST_PARALLEL_JOBS
        for target in targets
    }


def cpp_target_shards(targets: list[str], total: int) -> list[list[str]]:
    """Return a deterministic, build-and-test-balanced exact partition of C++ targets."""
    if total <= 0:
        raise SystemExit(f"invalid C++ shard total: {total}")
    if len(targets) != len(set(targets)):
        raise SystemExit("C++ shard partition input contains duplicate targets")
    weights = cpp_target_weights(targets)
    try:
        shards = ci_shard_binpack.assign_shards(targets, total, weights)
        ci_shard_binpack.verify_partition(targets, shards, excluded=())
    except ci_shard_binpack.PartitionError as exc:
        raise SystemExit(f"C++ shard partition invariant violated: {exc}") from exc
    return shards


def cpp_test_regex(names: Iterable[str]) -> str:
    """Match CTest names belonging to the supplied manifest targets / standalone tests."""
    escaped = [re.escape(name) for name in names]
    return "^(" + "|".join(escaped) + r")(\.|$)" if escaped else "$^"


def cpp_target_label_regex(names: Iterable[str]) -> str:
    """Match the per-build-target CTest labels installed by pops_add_gtest_suite."""
    escaped = [re.escape(f"cpp-target:{name}") for name in names]
    return "^(" + "|".join(escaped) + ")$" if escaped else "$^"


def verify_cpp_target_labels(args: argparse.Namespace) -> int:
    """Fail unless every selected build target owns discovered CTest cases.

    ``gtest_discover_tests`` names cases after their GTest suite rather than their
    CMake executable, so the C++ shards execute through exact ``cpp-target:*``
    labels.  Verifying the labels from CTest's JSON model before execution keeps
    a malformed CMake property list from turning a shard into a false-green
    ``No tests were found`` run.
    """
    targets = list(dict.fromkeys(args.targets))
    if not targets:
        raise SystemExit("C++ target-label verification requires at least one target")

    try:
        payload = json.loads(Path(args.ctest_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read CTest JSON inventory {args.ctest_json}: {exc}") from exc

    tests = payload.get("tests")
    if not isinstance(tests, list):
        raise SystemExit("CTest JSON inventory has no tests array")

    expected = {f"cpp-target:{target}": target for target in targets}
    hits: dict[str, list[str]] = {target: [] for target in targets}
    for test in tests:
        if not isinstance(test, dict):
            continue
        test_name = test.get("name")
        if not isinstance(test_name, str):
            continue
        labels: set[str] = set()
        properties = test.get("properties", [])
        if isinstance(properties, list):
            for prop in properties:
                if not isinstance(prop, dict) or prop.get("name") != "LABELS":
                    continue
                value = prop.get("value", [])
                if isinstance(value, str):
                    encoded_labels = (value,)
                elif isinstance(value, list):
                    encoded_labels = tuple(
                        item for item in value if isinstance(item, str)
                    )
                else:
                    encoded_labels = ()
                # CTest's JSON inventory is the selection authority: each
                # LABELS entry is one atomic label.  A semicolon here is not a
                # second serialization layer; it proves CMake overescaped the
                # LABELS property and CTest will treat the complete string as
                # one label.  Fail closed instead of inventing labels which an
                # exact ``ctest -L`` expression cannot select.
                malformed = [label for label in encoded_labels if ";" in label]
                if malformed:
                    raise SystemExit(
                        "CTest target-label contract failed; test "
                        f"{test_name!r} has non-atomic LABELS entries: "
                        + ", ".join(repr(label) for label in malformed)
                    )
                labels.update(label for label in encoded_labels if label)
        for label in labels & expected.keys():
            hits[expected[label]].append(test_name)

    missing = [target for target in targets if not hits[target]]
    if missing:
        details = ", ".join(
            f"{target} (expected cpp-target:{target})" for target in missing
        )
        raise SystemExit(
            "CTest target-label contract failed; selected targets without discovered cases: "
            + details
        )

    print(
        f"verified {len(targets)} C++ target labels across "
        f"{sum(len(names) for names in hits.values())} discovered cases"
    )
    return 0


def write_explain_file(path: str | None, payload: dict) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_cpp_impact(
    changed: list[str],
    suites: list[dict],
    all_target_set: set[str],
    reasons: dict[str, set[str]],
) -> tuple[set[str], list[str], set[str], dict[str, dict]]:
    """Union the per-file C++ impact of every changed file (ADC-646).

    Each file is classified independently and its impact joins the union, instead of the old
    all-or-nothing rule where a single non-header file collapsed the whole change to coarse
    labels. Returns ``(selected, full_reasons, areas, impact)``:

    * ``selected`` -- the union of every file's include-closure / runtime-target / binding-label /
      codegen-label /
      direct-test targets;
    * ``full_reasons`` -- non-empty iff some file forces a FULL selection (a global-includer or
      missing header, or an unmapped build-input path); the caller then escalates to ALL;
    * ``areas`` -- the label areas contributed by runtime/binding/codegen files;
    * ``impact`` -- ``{file: {"kind": ..., "targets": [...]/"labels": [...]/...}}`` for the
      ``--explain-file`` plan, so every file's reason is auditable.

    The per-file kinds:

    * ``include/pops/**`` header -> ``include-impact`` (its source-closure suites) or, when the
      header is a global includer / absent, ``all`` (soundness / fail-open);
    * ``src/runtime/**`` ``.cpp``/``.hpp`` -> ``runtime-tu-targets`` (the serial consumers of its
      central OBJECT lib); an unregistered source or other build input fails open to ``all``;
    * ``python/bindings/**`` adapters -> ``binding-labels``; non-source adapter build inputs fail
      open to ``all``;
    * ``tests/cpp/**`` ``.cpp`` -> ``test-target`` (that one suite, when it is a serial target);
    * ``python/pops/codegen/**`` -> ``codegen-labels`` (the native_loader / compiled-model group);
    * other ``python/pops/**``, ``docs/**``, ``tutorials/**``, ``tests/python/**``, top-level
      docs/CHANGELOG -> ``none`` (zero C++ impact);
    * everything else (cmake, workflows, scripts, CMakeLists, ``*.cmake``, the manifest) ->
      ``all`` (unmapped build input, fail-safe).
    """
    selected: set[str] = set()
    areas: set[str] = set()
    impact: dict[str, dict] = {}
    full_reasons: list[str] = []

    # The include-graph global-includer closure is read once (fail-open on any graph error).
    try:
        global_closure = ci_include_graph.global_includer_closure()
        graph_error: str | None = None
    except ci_include_graph.GraphError as exc:
        global_closure = set()
        graph_error = f"include-graph-unreadable:{exc}"

    for path in changed:
        if is_cpp_header(path):
            node = path[len("include/") :]
            if graph_error is not None:
                impact[path] = {"kind": "all", "reason": graph_error}
                full_reasons.append(f"{path}:{graph_error}")
                continue
            if node in global_closure:
                impact[path] = {"kind": "all", "reason": "header-in-global-includer-closure"}
                full_reasons.append(f"{path}:header-in-global-includer-closure")
                continue
            if not ci_include_graph.header_exists(node):
                impact[path] = {"kind": "all", "reason": "changed-header-not-in-tree"}
                full_reasons.append(f"{path}:changed-header-not-in-tree")
                continue
            hit_targets: set[str] = set()
            try:
                for suite in suites:
                    closure: set[str] = set()
                    for source in suite["sources"]:
                        closure |= ci_include_graph.source_closure(source)
                    if node in closure:
                        hit_targets.add(suite["name"])
                        add_reason(reasons, suite["name"], f"include-impact:{node}")
            except ci_include_graph.GraphError as exc:
                impact[path] = {"kind": "all", "reason": f"suite-source-missing:{exc}"}
                full_reasons.append(f"{path}:suite-source-missing:{exc}")
                continue
            selected.update(hit_targets)
            impact[path] = {"kind": "include-impact", "targets": sorted(hit_targets)}
            continue

        if path.startswith(CPP_RUNTIME_PREFIX):
            if is_cpp_runtime_tu(path):
                tu_targets, matched_libs = runtime_tu_targets(path, all_target_set)
                if not matched_libs:
                    impact[path] = {
                        "kind": "all",
                        "reason": "runtime-source-not-in-central-manifest",
                    }
                    full_reasons.append(f"{path}:runtime-source-not-in-central-manifest")
                    continue
                selected.update(tu_targets)
                for target in tu_targets:
                    add_reason(reasons, target, "runtime-tu:" + ",".join(sorted(matched_libs)))
                impact[path] = {
                    "kind": "runtime-tu-targets",
                    "object_libs": sorted(matched_libs),
                    "targets": sorted(tu_targets),
                }
            else:
                impact[path] = {"kind": "all", "reason": "runtime-build-input"}
                full_reasons.append(f"{path}:runtime-build-input")
            continue

        if path.startswith(CPP_BINDING_PREFIX):
            if not is_cpp_binding_tu(path):
                impact[path] = {"kind": "all", "reason": "binding-build-input"}
                full_reasons.append(f"{path}:binding-build-input")
                continue
            file_areas = areas_for(path, CPP_BINDING_AREAS)
            areas.update(file_areas)
            hit = select_cpp_by_labels(suites, file_areas, reasons)
            selected.update(hit)
            impact[path] = {
                "kind": "binding-labels",
                "labels": sorted(expand_area_labels(file_areas)),
                "targets": sorted(hit),
            }
            continue

        if path.startswith("tests/cpp/") and path.endswith(".cpp"):
            target = Path(path).stem
            if target in all_target_set:
                selected.add(target)
                add_reason(reasons, target, "direct-test-edit")
                impact[path] = {"kind": "test-target", "targets": [target]}
            else:
                # A test source not in the serial manifest (MPI-only, or a support file that
                # slipped the broad guard) has no serial target to build.
                impact[path] = {"kind": "none", "reason": "non-serial-test-source"}
            continue

        if path.startswith(CPP_CODEGEN_PREFIX):
            areas.update(CPP_CODEGEN_AREAS)
            hit = select_cpp_by_labels(suites, CPP_CODEGEN_AREAS, reasons)
            selected.update(hit)
            impact[path] = {
                "kind": "codegen-labels",
                "labels": sorted(expand_area_labels(CPP_CODEGEN_AREAS)),
                "targets": sorted(hit),
            }
            continue

        if path in CPP_ZERO_IMPACT_FILES or startswith_any(path, CPP_ZERO_IMPACT_PREFIXES):
            impact[path] = {"kind": "none", "reason": "zero-cpp-impact"}
            continue

        # Unmapped: cmake, workflows, scripts, CMakeLists, ``*.cmake``, the manifest, or any path
        # this classifier does not recognise -> fail-safe ALL.
        impact[path] = {"kind": "all", "reason": "unmapped-path"}
        full_reasons.append(f"{path}:unmapped-path")

    return selected, full_reasons, areas, impact


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
    # Broad build / support / core / parallel changes still short-circuit to ALL before the
    # compositional pass: a change to those touches the shared build or every target's headers,
    # so per-file impact cannot bound it (kept from the pre-ADC-646 guards).
    if force_full_from_changed(changed, CPP_BROAD_FILES, CPP_BROAD_PREFIXES):
        full_reasons.append("broad-build-or-support-change")
    full = bool(full_reasons)

    selected: set[str] = set()
    areas: set[str] = set()
    reasons: dict[str, set[str]] = {}
    impact: dict[str, dict] = {}
    # ADC-646: compositional per-file impact. Each changed file contributes its own C++ impact
    # (include-closure / binding-label group / direct test / codegen-label group / nothing), and
    # the selection is their UNION. A global-includer or missing header, or an unmapped build
    # input, escalates the whole change to ALL (soundness / fail-safe); everything else prunes.
    if not full:
        selected, per_file_full, areas, impact = classify_cpp_impact(
            changed, suites, all_target_set, reasons
        )
        if per_file_full:
            full = True
            full_reasons.extend(per_file_full)
        elif selected:
            for target in CPP_SMOKE_TARGETS:
                if target in all_target_set:
                    selected.add(target)
                    add_reason(reasons, target, "smoke-backstop")
        elif not only_meta(changed) and not all(
            entry.get("kind") == "none" for entry in impact.values()
        ):
            # No file forced ALL, yet nothing was selected and the change is not pure-meta and not
            # purely zero-impact: an area we recognise contributed no suite. Fail safe to ALL.
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
        regex = cpp_test_regex(targets)

    shard_index_arg = getattr(args, "shard_index", None)
    shard_total_arg = getattr(args, "shard_total", None)
    if shard_index_arg is None and shard_total_arg is None:
        shard_index = 0
        shard_total = 1
    elif shard_index_arg is None or shard_total_arg is None:
        raise SystemExit("C++ sharding requires both --shard-index and --shard-total")
    else:
        shard_index = shard_index_arg
        shard_total = shard_total_arg
    if shard_index < 0 or shard_index >= shard_total:
        raise SystemExit(f"invalid C++ shard {shard_index}/{shard_total}")

    target_shards = cpp_target_shards(targets, shard_total)
    shard_targets = target_shards[shard_index]
    shard_regex = cpp_test_regex(shard_targets)
    shard_label_regex = cpp_target_label_regex(shard_targets)

    summary = f"{mode}: {len(targets)}/{len(all_targets)} C++ tests"
    shard_summary = (
        f"{len(shard_targets)} targets in shard {shard_index}/{shard_total}"
    )
    print(summary)
    print(shard_summary)
    if areas:
        print("areas=" + ",".join(sorted(areas)))
    for target in shard_targets:
        print(target)

    write_github_outputs(
        getattr(args, "github_output", None),
        {
            "cpp_mode": mode,
            "cpp_targets": " ".join(targets),
            "cpp_regex": regex,
            "cpp_count": str(len(targets)),
            "cpp_total": str(len(all_targets)),
            "cpp_shard_index": str(shard_index),
            "cpp_shard_total": str(shard_total),
            "cpp_shard_targets": " ".join(shard_targets),
            "cpp_shard_regex": shard_regex,
            "cpp_shard_label_regex": shard_label_regex,
            "cpp_shard_count": str(len(shard_targets)),
            "cpp_shard_counts": ",".join(str(len(shard)) for shard in target_shards),
            "cpp_areas": ",".join(sorted(areas)) if areas else "-",
            "cpp_summary": summary,
            "cpp_shard_summary": shard_summary,
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
            "shard_index": shard_index,
            "shard_total": shard_total,
            "shard_count": len(shard_targets),
            "shard_counts": [len(shard) for shard in target_shards],
            "shard_targets": shard_targets,
            "shard_regex": shard_regex,
            "shard_label_regex": shard_label_regex,
            "target_shards": target_shards,
            "selected": targets,
            "selected_reasons": {key: sorted(value) for key, value in sorted(reasons.items()) if key in targets},
            # ADC-646: per-file impact -- {file: {kind, targets/labels/reason}} -- so the plan
            # spells out why each changed file did (or did not) pull suites into the selection.
            "impact": {key: impact[key] for key in sorted(impact)},
        },
    )
    return 0


def plan_cpp_label(args: argparse.Namespace) -> int:
    """Project one exact C++ manifest label into build targets for a dedicated CI job."""
    manifest = load_manifest()
    targets = cpp_targets_with_label(manifest, args.label)
    mpi_ctest_count = cpp_mpi_ctest_count(manifest) if args.label == "mpi" else 0
    summary = f"label {args.label}: {len(targets)} C++ targets"
    if args.label == "mpi":
        summary += f", {mpi_ctest_count} CTest launches"
    print(summary)
    for target in targets:
        print(target)
    write_github_outputs(
        getattr(args, "github_output", None),
        {
            "cpp_label": args.label,
            "cpp_label_targets": " ".join(targets),
            "cpp_label_count": str(len(targets)),
            "cpp_label_ctest_count": str(mpi_ctest_count),
            "cpp_label_summary": summary,
        },
    )
    write_explain_file(
        getattr(args, "explain_file", None),
        {
            "kind": "cpp-label",
            "label": args.label,
            "selected_count": len(targets),
            "ctest_count": mpi_ctest_count,
            "selected": targets,
        },
    )
    return 0


def plan_python_mpi(args: argparse.Namespace) -> int:
    """Write the manifest-owned real-MPI Python launch plan for the dedicated CI job."""
    entries = manifest_python_mpi_entrypoints(load_manifest())
    if not entries:
        raise SystemExit("no Python MPI entrypoints declared in tests/test_manifest.toml")
    plan_path = Path(args.plan_file)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        "".join(f"{entry['nproc']}\t{entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )
    summary = f"{len(entries)} manifest-owned Python MPI entrypoints"
    print(summary)
    for entry in entries:
        print(f"np={entry['nproc']} {entry['path']}")
    write_github_outputs(
        getattr(args, "github_output", None),
        {
            "python_mpi_count": str(len(entries)),
            "python_mpi_plan_file": str(plan_path),
            "python_mpi_summary": summary,
        },
    )
    write_explain_file(
        getattr(args, "explain_file", None),
        {
            "kind": "python-mpi",
            "selected_count": len(entries),
            "selected": entries,
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
        raise SystemExit(1) from exc

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
    cpp.add_argument("--shard-index", type=int)
    cpp.add_argument("--shard-total", type=int)
    cpp.add_argument("--force-all", action="store_true")
    cpp.set_defaults(func=plan_cpp)

    cpp_label = sub.add_parser("cpp-label")
    cpp_label.add_argument("--label", required=True)
    cpp_label.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    cpp_label.add_argument("--explain-file")
    cpp_label.set_defaults(func=plan_cpp_label)

    cpp_target_labels = sub.add_parser("verify-cpp-target-labels")
    cpp_target_labels.add_argument("--ctest-json", required=True)
    cpp_target_labels.add_argument("--targets", nargs="+", required=True)
    cpp_target_labels.set_defaults(func=verify_cpp_target_labels)

    py_mpi = sub.add_parser("python-mpi")
    py_mpi.add_argument("--plan-file", required=True)
    py_mpi.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    py_mpi.add_argument("--explain-file")
    py_mpi.set_defaults(func=plan_python_mpi)

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
