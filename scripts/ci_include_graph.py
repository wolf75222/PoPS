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
* Only ``#include <pops/...>`` edges are modelled (angle-bracket, project headers). System
  headers and the test-local ``"test_harness.hpp"`` / ``"gtest_compat.hpp"`` relative includes
  are deliberately ignored: they are not part of the ``include/pops`` header graph and are
  handled by the caller's broad-file rules.
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


def source_closure(source_rel: str) -> set[str]:
    """Return the ``pops/...`` header closure reachable from a single source file.

    ``source_rel`` is a repo-relative path (e.g. a suite's ``tests/cpp/...cpp``). Its direct
    ``pops/...`` includes seed a transitive closure over ``include/pops``. Raises
    ``GraphError`` if the source file is missing (fail-open: the caller escalates to FULL).
    """
    path = ROOT / source_rel
    if not path.is_file():
        raise GraphError(f"source file not found: {source_rel}")
    return transitive_closure(pops_includes(_read(path)))


def runtime_and_binding_includes() -> set[str]:
    """Collect direct ``pops/...`` includes of native runtime and pybind adapter sources.

    Covers ``src/runtime/**`` production sources and seam templates plus the actual adapter TUs
    under ``python/bindings/**``. These translation units are compiled into or linked by
    effectively every test target, so their transitive closure is the GLOBAL-INCLUDERS set below.
    """
    roots: set[str] = set()
    for pattern in ("*.cpp", "*.cpp.in", "*.hpp", "*.h"):
        for source_root in (RUNTIME_DIR, BINDINGS_DIR):
            for src in source_root.rglob(pattern):
                roots |= pops_includes(_read(src))
    return roots


def emitter_includes() -> set[str]:
    """Collect the ``pops/...`` includes the DSL codegen emits into generated ``.cpp``.

    These live as ``#include <pops/...>`` string literals in ``python/pops/**``; they land in
    every generated translation unit, so they are part of the global-includers roots.
    """
    roots: set[str] = set()
    for py in POPS_CODEGEN.rglob("*.py"):
        roots |= pops_includes(_read(py))
    return roots


def cpp_support_includes() -> set[str]:
    """Collect the ``pops/...`` includes of the shared ``tests/cpp/support/**`` headers.

    The support headers (``test_harness.hpp`` etc.) are pulled into nearly every test source,
    so any production header they reach is a global includer too.
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

    A header in the TRANSITIVE CLOSURE of this set is compiled into or linked by effectively
    every test target; a change to it must select ALL suites (the soundness rule). The union is
    the seed; ``ci_select_tests`` closes it transitively.
    """
    return (
        runtime_and_binding_includes()
        | emitter_includes()
        | cpp_support_includes()
    )


def global_includer_closure() -> set[str]:
    """Transitive closure of :func:`global_includer_roots` over ``include/pops``."""
    return transitive_closure(global_includer_roots())
