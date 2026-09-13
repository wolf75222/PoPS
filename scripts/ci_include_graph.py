#!/usr/bin/env python3
"""Shared ``#include <pops/...>`` graph over ``include/pops/**`` (ADC-629).

The C++ test selection (``ci_select_tests.py``) narrows a header-only change to the suites
whose sources transitively include the changed header. That needs a reusable, source-only
model of the include graph: which ``pops/...`` headers each file pulls in, and the transitive
closure of any starting set. The ADC-608 quarantine fence
(``tests/python/architecture/test_no_quarantined_header_leak.py``) already walked this graph
for the production roots; this module factors that walk out so both callers share ONE
implementation instead of two divergent copies.

Scope and conventions
----------------------
* Public graph nodes use ``#include <pops/...>``. Source closures also follow quoted
  local includes, so private runtime and test-support dependencies reach their actual
  consuming sources. System headers remain outside the project graph.
* Header identifiers are paths RELATIVE to ``include/`` (i.e. ``pops/...``), matching the text
  of the ``#include`` directive, so a changed file ``include/pops/x/y.hpp`` maps to the graph
  node ``pops/x/y.hpp`` by stripping the ``include/`` prefix.
* Everything is stdlib-only and pure source-parse (no ``pops`` / ``_pops`` import), so the
  architecture gate runs it before any build.

Determinism
-----------
The graph is a plain dict; closures are computed by BFS and returned as sorted lists/sets by
the callers. No clock, no randomness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIR = ROOT / "include"
POPS_CODEGEN = ROOT / "python" / "pops"
BINDINGS_DIR = ROOT / "python" / "bindings"
RUNTIME_DIR = ROOT / "src" / "runtime"
CPP_TESTS_DIR = ROOT / "tests" / "cpp"

# Matches ``#include <pops/...>`` both as a real directive and inside a codegen string literal.
# Identical shape to the ADC-608 fence (kept in sync on purpose).
_INCLUDE_RE = re.compile(r"#\s*include\s*<\s*(pops/[^>]+?)\s*>")


class GraphError(RuntimeError):
    """Raised when the include graph cannot be built or read (fail-open signal to callers)."""


def pops_includes(text: str) -> set[str]:
    """Return the set of ``pops/...`` headers referenced by ``text`` (directive or literal)."""
    return {m.group(1) for m in _INCLUDE_RE.finditer(text)}


def _read(path: Path) -> str:
    """Read ``path`` as text, tolerating encoding noise (source-parse, best effort)."""
    return path.read_text(encoding="utf-8", errors="ignore")


def header_includes(rel: str) -> set[str]:
    """Return the ``pops/...`` headers included by ``include/<rel>``.

    An absent header returns the empty set: the caller decides whether a missing node is an
    anomaly (fail-open) or an expected leaf that simply includes nothing.
    """
    path = INCLUDE_DIR / rel
    if not path.is_file():
        return set()
    return pops_includes(_read(path))


def header_exists(rel: str) -> bool:
    """True if ``include/<rel>`` is a real file on disk."""
    return (INCLUDE_DIR / rel).is_file()


def transitive_closure(roots: Iterable[str]) -> set[str]:
    """BFS the transitive ``#include`` closure of ``roots`` over ``include/pops``.

    The returned set INCLUDES the roots themselves (a source is in its own closure) and every
    ``pops/...`` header reachable from them. Absent headers are still recorded as nodes (so the
    caller can detect a dangling include) but contribute no out-edges.
    """
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(nxt for nxt in header_includes(cur) if nxt not in seen)
    return seen


@lru_cache(maxsize=None)
def _source_dependencies(root: str, include_dir: str, source_rel: str) -> frozenset[str]:
    """Source-only dependency snapshot, including local quoted private headers.

    CI plans one immutable checkout per process. The root is part of the cache key so
    synthetic repositories and separate checkouts cannot share graph entries.
    """
    repo = Path(root)
    include = Path(include_dir)
    start = repo / source_rel
    if not start.is_file():
        raise GraphError(f"source file not found: {source_rel}")
    seen: set[Path] = set()
    pending = [start.resolve()]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        try:
            text = _read(path)
        except OSError as exc:
            raise GraphError(f"cannot read {path}: {exc}") from exc
        pending.extend((include / node).resolve() for node in pops_includes(text))
        for quoted in re.findall(r'#\s*include\s*"([^"\n]+)"', text):
            # Production private headers use paths relative to their translation unit;
            # quoted project includes may also resolve against the public include root.
            candidates = [path.parent / quoted, include / quoted]
            if source_rel.startswith("tests/cpp/"):
                # pops_add_gtest_suite declares this include directory for every
                # C++ suite; bare test_harness.hpp and support helpers resolve here.
                candidates.append(repo / "tests/cpp/support" / quoted)
            for candidate in candidates:
                candidate = candidate.resolve()
                if candidate.is_relative_to(repo) and candidate.is_file():
                    pending.append(candidate)
                    break
    return frozenset(str(path.relative_to(repo)) for path in seen if path.is_relative_to(repo))


def source_dependencies(source_rel: str) -> set[str]:
    """Repo-relative source and public/private header dependencies of a source file."""
    return set(_source_dependencies(str(ROOT), str(INCLUDE_DIR), source_rel))


def source_closure(source_rel: str) -> set[str]:
    """Public ``pops/...`` headers reached through public or local quoted includes."""
    return {
        path[len("include/"):]
        for path in source_dependencies(source_rel)
        if path.startswith("include/pops/")
    }


def runtime_and_binding_includes() -> set[str]:
    """Collect direct ``pops/...`` includes of native runtime and pybind adapter sources.

    Covers ``src/runtime/**`` production sources and seam templates plus adapter TUs.
    This aggregate supports architecture fences. Test selection uses the actual linked
    consumers of each runtime source instead of treating the aggregate as globally shared.
    """
    roots: set[str] = set()
    for pattern in ("*.cpp", "*.cpp.in", "*.hpp", "*.h"):
        for source_root in (RUNTIME_DIR, BINDINGS_DIR):
            for src in source_root.rglob(pattern):
                roots |= pops_includes(_read(src))
    return roots


def emitter_includes() -> set[str]:
    """Collect the ``pops/...`` includes the DSL codegen emits into generated ``.cpp``.

    These live as ``#include <pops/...>`` string literals in ``python/pops/**``. The aggregate
    is useful for production-header architecture fences; individual emitter dependencies
    also provide the native-to-Python test-selection bridge.
    """
    roots: set[str] = set()
    for py in POPS_CODEGEN.rglob("*.py"):
        roots |= pops_includes(_read(py))
    return roots


def cpp_support_includes() -> set[str]:
    """Collect the ``pops/...`` includes of the shared ``tests/cpp/support/**`` headers.

    This aggregate is retained for architecture fences. Individual source closures follow
    the support headers actually included by that source.
    """
    support = CPP_TESTS_DIR / "support"
    roots: set[str] = set()
    if not support.is_dir():
        return roots
    for src in support.rglob("*"):
        if src.is_file():
            roots |= pops_includes(_read(src))
    return roots


def global_includer_roots() -> set[str]:
    """Direct ``pops/...`` includes of the heavy shared TUs, seams, emitter and cpp support.

    Historical name retained for architecture callers. This is a production-root union,
    not proof that every test consumes every header. Selection follows source ownership.
    """
    return (
        runtime_and_binding_includes()
        | emitter_includes()
        | cpp_support_includes()
    )


def global_includer_closure() -> set[str]:
    """Transitive closure of :func:`global_includer_roots` over ``include/pops``."""
    return transitive_closure(global_includer_roots())
