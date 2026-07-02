#!/usr/bin/env python3
"""Import-closure test selection for the PoPS CI gate.

This module answers one question for the ``gate-python`` selector: given a set of
changed ``python/pops/**`` files, which ``tests/python/**/test_*.py`` files can be
affected by that change? It replaces the coarse "area heuristic" (name-token match)
for pops source changes with the ACTUAL static import graph.

The whole module is stdlib-only (``ast`` + ``pathlib``) and SOURCE-ONLY: it never
imports ``pops`` or ``_pops``, so it runs on the bare runner interpreter before any
``pip install``.

Two graphs are built by walking the source tree:

* the pops MODULE graph -- ``python/pops/**/*.py``, mapping each pops module to the
  set of pops modules it imports;
* the TEST import map -- ``tests/python/**/test_*.py``, mapping each test file to the
  pops modules it imports AND the sibling ``test_*`` modules it imports (cross-test
  edges resolved by the ``python3 <file>`` sys.path[0] convention).

Both walks collect ``Import`` / ``ImportFrom`` nodes at EVERY AST depth (``ast.walk``),
because pops modules and the tests import pervasively at function scope (lazy imports
that keep the import-graph architecture gate green). A module-scope-only walker would
silently under-select.

The impact query is a REVERSE transitive closure: a changed pops file affects a test
iff the test (transitively) imports the changed file's module. See :func:`impacted_tests`.

The algorithm is deterministic (sorted iteration), cycle-safe (visited set), and
runs in O(nodes + edges).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


# Repo layout anchors. ``parents[1]`` is the repo root (this file is scripts/…).
ROOT = Path(__file__).resolve().parents[1]
POPS_DIR = ROOT / "python" / "pops"
TESTS_DIR = ROOT / "tests" / "python"


# --------------------------------------------------------------------------- #
# Module-name <-> path helpers                                                 #
# --------------------------------------------------------------------------- #
def module_name_for_pops_path(path: Path) -> str | None:
    """Return the dotted ``pops...`` module name for a file under ``python/pops``.

    ``python/pops/runtime/_bound_sim.py`` -> ``pops.runtime._bound_sim`` and a package
    ``__init__.py`` -> the package name (``python/pops/runtime/__init__.py`` ->
    ``pops.runtime``). Returns None for a path outside ``python/pops``.
    """
    try:
        rel = path.resolve().relative_to(POPS_DIR.parent)
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or parts[0] != "pops":
        return None
    return ".".join(parts)


def _pops_module_files() -> dict[str, Path]:
    """Map every existing pops module name to its source file.

    Both a package (its ``__init__.py``) and a plain module resolve to one name; the
    package name maps to the ``__init__.py`` so a ``pops.x`` edge is a real node.
    """
    files: dict[str, Path] = {}
    for path in sorted(POPS_DIR.rglob("*.py")):
        name = module_name_for_pops_path(path)
        if name is not None:
            files[name] = path
    return files


def _resolve_relative(
    importer_module: str, is_package: bool, level: int, tail: str | None
) -> str | None:
    """Resolve a relative ``ImportFrom`` (``level>0``) to an absolute dotted name.

    ``importer_module`` is the dotted name of the file doing the import (a package's
    ``__init__`` is collapsed to the package name). Python resolves ``from ... import``
    against the importer's CONTAINING package: for a plain module ``pkg.mod`` a
    ``level==1`` import (``from . import x``) resolves in ``pkg``; for a package
    ``__init__`` (``is_package``) ``level==1`` resolves in the package ITSELF. So the
    anchor is the module name minus ``level`` trailing parts for a module, minus
    ``level-1`` for a package. Returns the absolute ``pops...`` anchor, or None if the
    import walks above ``pops``.
    """
    parts = importer_module.split(".")
    drop = level if not is_package else level - 1
    if drop > len(parts):
        return None
    base = parts[: len(parts) - drop]
    if not base:
        return None
    if tail:
        base = base + tail.split(".")
    anchor = ".".join(base)
    if anchor != "pops" and not anchor.startswith("pops."):
        return None
    return anchor


def _target_with_ancestors(target: str, valid: set[str]) -> set[str]:
    """Expand a dotted import target to itself plus every ancestor PACKAGE that is real.

    Importing ``pops.numerics.riemann.waves`` executes each parent ``__init__.py``
    (``pops``, ``pops.numerics``, ``pops.numerics.riemann``) as a side effect, so a
    change to ANY of those packages can affect the importer. We therefore register an
    edge to every ancestor that exists as a real pops module, not just the leaf. Only
    names present in ``valid`` are returned (a name that resolves to a class/function,
    or a package with no ``__init__.py`` node, is dropped).
    """
    out: set[str] = set()
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in valid:
            out.add(cand)
        if cand == "pops":
            break
    return out


def _iter_import_targets(
    tree: ast.AST, importer_module: str | None, is_package: bool = False
):
    """Yield absolute dotted import targets that name a pops module or subpackage.

    Walks the FULL tree (every depth), so function-scope / lazy imports are captured.
    ``importer_module`` supplies the anchor for relative imports (``is_package`` says
    whether that module is a package ``__init__``, which shifts the relative anchor by
    one level); pass ``importer_module=None`` for a test file (test files never use
    pops-relative imports -- they are top-level scripts).

    For ``from pops.x.y import z`` we yield BOTH ``pops.x.y`` (the module the names come
    from) and ``pops.x.y.z`` (in case ``z`` is itself a submodule). The caller filters
    against the set of real module names, so a non-module name (a class/function) is
    dropped harmlessly.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pops" or alias.name.startswith("pops."):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and importer_module is not None:
                base = _resolve_relative(importer_module, is_package, node.level, node.module)
                if base is None:
                    continue
                yield base
                for alias in node.names:
                    if alias.name != "*":
                        yield f"{base}.{alias.name}"
            elif node.level == 0 and node.module:
                if node.module == "pops" or node.module.startswith("pops."):
                    yield node.module
                    for alias in node.names:
                        if alias.name != "*":
                            yield f"{node.module}.{alias.name}"


# --------------------------------------------------------------------------- #
# Graph builders                                                               #
# --------------------------------------------------------------------------- #
def build_module_graph(repo_root: Path | None = None) -> dict[str, set[str]]:
    """Return ``{pops_module: {pops_module it imports, ...}}`` over ``python/pops/**``.

    An edge is kept only when the target resolves to a REAL pops module file (so
    imported class/function names are dropped). Absolute and relative imports at any AST
    depth are covered. Self-edges are dropped.
    """
    global POPS_DIR, TESTS_DIR
    if repo_root is not None:
        POPS_DIR = repo_root / "python" / "pops"
        TESTS_DIR = repo_root / "tests" / "python"

    modules = _pops_module_files()
    valid = set(modules)
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, ValueError):
            # A pops file we cannot parse is handled fail-safe by the caller
            # (plan_python selects ALL when a changed pops module is off-graph).
            continue
        is_pkg = path.name == "__init__.py"
        for target in _iter_import_targets(tree, name, is_pkg):
            for real in _target_with_ancestors(target, valid):
                if real != name:
                    graph[name].add(real)
    return graph


def _test_files() -> list[Path]:
    return sorted(
        path
        for path in TESTS_DIR.rglob("test_*.py")
        if "architecture" not in path.relative_to(TESTS_DIR).parts
    )


def test_imports(
    repo_root: Path | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return two maps over ``tests/python/**/test_*.py``.

    * ``test_to_pops``: ``{"tests/python/unit/x/test_x.py": {pops modules it imports}}``.
    * ``test_to_test``: ``{"tests/python/unit/x/test_x.py": {"tests/python/unit/x/test_y.py", ...}}``
      -- the cross-test edges (a bare ``from test_y import ...`` resolves to the sibling
      file via the ``python3 <file>`` sys.path[0] convention).

    Both are collected at every AST depth. Test paths are repo-root-relative POSIX
    strings, matching the selector's file identifiers.
    """
    global POPS_DIR, TESTS_DIR
    if repo_root is not None:
        POPS_DIR = repo_root / "python" / "pops"
        TESTS_DIR = repo_root / "tests" / "python"

    pops_modules = set(_pops_module_files())
    files = _test_files()
    test_stems_by_dir: dict[Path, dict[str, Path]] = defaultdict(dict)
    for file in files:
        test_stems_by_dir[file.parent][file.stem] = file

    test_to_pops: dict[str, set[str]] = {}
    test_to_test: dict[str, set[str]] = {}
    for path in files:
        rel = path.resolve().relative_to(ROOT).as_posix()
        test_stems = test_stems_by_dir[path.parent]
        pops_hit: set[str] = set()
        sibling_hit: set[str] = set()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, ValueError):
            test_to_pops[rel] = pops_hit
            test_to_test[rel] = sibling_hit
            continue
        # pops imports (tests use absolute pops.* only -> importer_module=None). Each
        # target expands to its real ancestor packages too (their __init__ runs on import).
        for target in _iter_import_targets(tree, None):
            pops_hit.update(_target_with_ancestors(target, pops_modules))
        # sibling test imports (bare module name, any depth).
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    head = alias.name.split(".")[0]
                    if head in test_stems and head != path.stem:
                        sibling_hit.add(test_stems[head].resolve().relative_to(ROOT).as_posix())
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    head = node.module.split(".")[0]
                    if head in test_stems and head != path.stem:
                        sibling_hit.add(test_stems[head].resolve().relative_to(ROOT).as_posix())
        test_to_pops[rel] = pops_hit
        test_to_test[rel] = sibling_hit
    return test_to_pops, test_to_test


# --------------------------------------------------------------------------- #
# Impact query                                                                 #
# --------------------------------------------------------------------------- #
def _reverse_module_closure(
    seed_modules: set[str], graph: dict[str, set[str]]
) -> set[str]:
    """Return every module that (transitively) imports any module in ``seed_modules``.

    Builds the reverse adjacency once, then BFS from the seeds. Cycle-safe via the
    visited set; O(edges).
    """
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, dsts in graph.items():
        for dst in dsts:
            reverse[dst].add(src)
    closed: set[str] = set()
    frontier = sorted(m for m in seed_modules if m in graph)
    stack = list(frontier)
    closed.update(frontier)
    while stack:
        current = stack.pop()
        for importer in sorted(reverse.get(current, ())):
            if importer not in closed:
                closed.add(importer)
                stack.append(importer)
    return closed


def impacted_tests(
    changed_py_files: list[str], repo_root: Path | None = None
) -> set[str]:
    """Return the set of test files impacted by the changed ``python/pops`` files.

    ``changed_py_files`` are repo-root-relative POSIX paths. Steps:

    1. Map each changed ``python/pops/**`` file to its module name; a file whose module
       is NOT on the graph (a brand-new file not yet imported, or an unparseable one)
       raises ``OffGraphChange`` so the caller can fall back to ALL (fail-safe).
    2. Reverse-transitive-close those seed modules over the pops module graph -> every
       pops module that can be affected.
    3. Select every test that imports (directly) any module in that closure.
    4. Close over the cross-test edges BOTH ways: if a selected test is imported by
       another test, that importer is pulled in too (its behaviour depends on the
       selected helper); and if a selected test imports a sibling helper, the helper is
       pulled in (it must run for the importer to make sense as a unit).

    Returns repo-root-relative POSIX test paths. Deterministic and cycle-safe.
    """
    graph = build_module_graph(repo_root)
    valid_modules = set(graph)

    seed_modules: set[str] = set()
    for changed in changed_py_files:
        norm = changed.strip().replace("\\", "/")
        if not norm.startswith("python/pops/") or not norm.endswith(".py"):
            continue
        name = module_name_for_pops_path(ROOT / norm)
        if name is None or name not in valid_modules:
            raise OffGraphChange(norm)
        seed_modules.add(name)

    if not seed_modules:
        return set()

    affected_modules = _reverse_module_closure(seed_modules, graph)

    test_to_pops, test_to_test = test_imports(repo_root)
    selected: set[str] = {
        test
        for test, mods in test_to_pops.items()
        if mods & affected_modules
    }

    _close_cross_test(selected, test_to_test)
    return selected


def _close_cross_test(selected: set[str], test_to_test: dict[str, set[str]]) -> None:
    """In-place close ``selected`` over cross-test edges both directions.

    Forward: a selected test's sibling helpers are added (it imports them).
    Reverse: any test importing a selected test is added (it depends on it).
    """
    reverse: dict[str, set[str]] = defaultdict(set)
    for importer, imported in test_to_test.items():
        for dep in imported:
            reverse[dep].add(importer)
    stack = sorted(selected)
    while stack:
        current = stack.pop()
        for dep in sorted(test_to_test.get(current, ())):  # forward: imported helpers
            if dep not in selected:
                selected.add(dep)
                stack.append(dep)
        for importer in sorted(reverse.get(current, ())):  # reverse: dependents
            if importer not in selected:
                selected.add(importer)
                stack.append(importer)


class OffGraphChange(Exception):
    """A changed ``python/pops`` file has no known module (new/unparseable file).

    The selector treats this as a fail-safe: run the full suite rather than risk
    under-selecting a file the static graph has never seen.
    """


if __name__ == "__main__":  # tiny CLI for manual dry-runs
    import sys

    files = sys.argv[1:]
    try:
        tests = impacted_tests(files)
    except OffGraphChange as exc:
        print(f"off-graph change ({exc}); fail-safe -> ALL tests")
        raise SystemExit(0)
    for test in sorted(tests):
        print(test)
    print(f"# {len(tests)} impacted test files", file=sys.stderr)
