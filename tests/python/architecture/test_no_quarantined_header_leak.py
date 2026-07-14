"""ADC-608: production headers must never reach a quarantined non-production header.

Of the headers under ``include/pops``, most are production-reachable from the three real
roots the compiled/bound artifacts pull in:

  1. EMITTER includes -- the ``#include <pops/...>`` lines the DSL codegen writes verbatim
     into every generated ``.cpp`` (parsed from the ``python/pops`` string literals).
  2. BINDINGS includes -- every ``#include <pops/...>`` in ``python/bindings`` (the pybind
     translation units and their headers).
  3. SEAM includes -- the ``python/bindings/templates/*.cpp.in`` transport/flux seams.

A residual set of headers is validation/reference/legacy test scaffolding that must stay OUT
of that production include closure. This file pins the quarantine as a pure SOURCE-PARSE check
(no ``pops`` / ``_pops`` import) so the source-only architecture gate always runs it. It makes
two assertions:

  (a) NO-LEAK: the production closure -- recomputed live here by a BFS from the three root sets
      -- never contains a quarantined header. The closure is DERIVED from source every run, so a
      future production edit that starts pulling a quarantined header fails here, not silently.
  (b) JUSTIFIED-OR-GONE: every quarantined header is either referenced by at least one file under
      ``tests/cpp`` (a real test justifies keeping it) or does not exist (it was deleted). A
      quarantined header that goes orphan -- still present, no test -- fails loud, forcing a
      delete-or-justify decision instead of letting dead scaffolding rot in the tree.
"""
import pathlib
import re

# tests/python/architecture/<this file> -> repo root is parents[3].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INCLUDE_DIR = REPO_ROOT / "include"
POPS_CODEGEN = REPO_ROOT / "python" / "pops"
BINDINGS_DIR = REPO_ROOT / "python" / "bindings"
CPP_TESTS_DIR = REPO_ROOT / "tests" / "cpp"
HEADER_MANIFEST = INCLUDE_DIR / "pops_public_headers.manifest"

# Matches ``#include <pops/...>`` both as a real directive and inside a codegen string literal.
_INCLUDE_RE = re.compile(r"#\s*include\s*<\s*(pops/[^>]+?)\s*>")

# The quarantined, non-production headers (paths relative to include/, i.e. ``pops/...``).
# The two AMR reference oracles and the two zero-reference validation bricks were DELETED under
# ADC-608 (git history preserves them); the assertions below tolerate their absence via (b). The
# rest are legitimate TEST-ONLY headers classified by the packaging manifest and fenced from
# production by (a). This keeps quarantine and installation classification in one source of truth.
_DELETED_QUARANTINED = (
    # Deleted (dead AMR reference oracles).
    "pops/numerics/time/reference/amr_reflux.hpp",
    "pops/numerics/time/reference/amr_level.hpp",
    # Deleted (zero-reference validation bricks).
    "pops/validation/physics/langmuir.hpp",
    "pops/validation/physics/two_fluid_isothermal.hpp",
)


def _manifest_test_only():
    rows = []
    for raw in HEADER_MANIFEST.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if row.startswith("test-only "):
            rows.append(row.removeprefix("test-only "))
    return tuple(rows)


_QUARANTINED = _DELETED_QUARANTINED + _manifest_test_only()


def _pops_includes(text):
    """Return the set of ``pops/...`` headers referenced by ``text`` (directive or literal)."""
    return {m.group(1) for m in _INCLUDE_RE.finditer(text)}


def _root_includes():
    """Collect the production root include set from the emitter, bindings and seam sources."""
    roots = set()
    for py in POPS_CODEGEN.rglob("*.py"):
        roots |= _pops_includes(py.read_text(encoding="utf-8", errors="ignore"))
    for pattern in ("*.cpp", "*.cpp.in", "*.hpp", "*.h"):
        for src in BINDINGS_DIR.rglob(pattern):
            roots |= _pops_includes(src.read_text(encoding="utf-8", errors="ignore"))
    return roots


def _header_includes(rel):
    """Return the ``pops/...`` headers included by ``include/<rel>`` (empty if it is absent)."""
    path = INCLUDE_DIR / rel
    if not path.is_file():
        return set()
    return _pops_includes(path.read_text(encoding="utf-8", errors="ignore"))


def _production_closure():
    """BFS the transitive ``#include`` closure of the production roots over ``include/pops``."""
    seen = set()
    stack = list(_root_includes())
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(nxt for nxt in _header_includes(cur) if nxt not in seen)
    return seen


def _cpp_test_referenced(rel):
    """True if any file under tests/cpp references ``rel`` (justifies keeping a fenced header)."""
    for src in CPP_TESTS_DIR.rglob("*"):
        if not src.is_file():
            continue
        if rel in src.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def test_production_never_reaches_a_quarantined_header():
    """(a) The production include closure contains no quarantined header (ADC-608)."""
    closure = _production_closure()
    leaked = sorted(closure.intersection(_QUARANTINED))
    assert not leaked, (
        "production headers (the codegen-emitter, bindings and seam include set) must never reach "
        "a quarantined validation/reference/test-only header, but the include closure now pulls "
        "%s -- remove the offending #include so the quarantined header stays out of production" % leaked
    )


def test_every_quarantined_header_is_test_justified_or_deleted():
    """(b) Each quarantined header is referenced by a test or does not exist (ADC-608)."""
    orphans = []
    for rel in _QUARANTINED:
        if not (INCLUDE_DIR / rel).is_file():
            continue  # deleted -- git history preserves it, nothing to justify.
        if not _cpp_test_referenced(rel):
            orphans.append(rel)
    assert not orphans, (
        "every quarantined header must be justified by at least one tests/cpp reference or deleted, "
        "but %s is present with no test using it -- delete it (git preserves the history) or add a "
        "test that exercises it" % orphans
    )


if __name__ == "__main__":
    # Runnable directly (the source-only architecture gate also collects it).
    test_production_never_reaches_a_quarantined_header()
    test_every_quarantined_header_is_test_justified_or_deleted()
    print("OK test_no_quarantined_header_leak")
