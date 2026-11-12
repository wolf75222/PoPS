#!/usr/bin/env python3
"""MG9 (spec 0.3, section 18.2): the registry of the bridges of ``pops.lib``.

A word of ``pops.lib`` that returns a native route without a Python body is a bridge. The registry
``tests/gates/mg9_bridges.toml`` lists each one (``[[bridge]]``), and each module of ``pops.lib``
still written with a b6 signature (``[[earlier]]``); the lists only shrink (MG6) and are empty at
the exit of phase 1 (OD29, OD31 condition S6). This module reads the source tree only: it imports
neither ``pops`` nor a case.

``classify`` sorts every module of ``pops.lib`` into a module of the surface (it imports
``pops.surface``, or no module of ``pops`` and ``[[earlier]]`` does not list it), a module of
``[[earlier]]``, a package that only re-exports its submodules, or an unlisted module.
``found_bridges`` finds the bridges of the modules of the surface by their source: a subclass of
``SchemeValue``, or a call that builds a native route of face, reconstruction, diffusion or solver
(``riemann.X()`` and ``reconstruction.X()`` other than ``User``, a name imported from
``pops.numerics`` or ``pops.solvers`` other than ``StencilFlux``), under the public word that
returns it.

``python3 scripts/mg9_bridges.py check [--check-base REVISION]`` checks the registry against the
tree and, with ``--check-base``, against the registry of the base revision: no bridge, no earlier
module and no word may be added (MG6). A base without the registry records nothing to compare.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "tests/gates/mg9_bridges.toml"
LIBRARY = "python/pops/lib"
GATE = "mg9-bridges"
# The native routes that carry a Python body: a face or a reconstruction written in Python
# (``riemann.User``, ``reconstruction.User``) and the stencil of a face flux (``StencilFlux``).
WITH_BODY = {"User", "StencilFlux"}
NATIVE_MODULES = ("pops.numerics", "pops.solvers")
ROUTE_MODULES = {"riemann", "reconstruction"}


# A [[bridge]] entry names its word, module, route and owner. It may also state the conditions
# of a ruling: ``refusal``, the test (path::name) of the refusal that the bridge carries until its
# owner removes it, ``exit_test``, what the test that removes it runs, and ``arrives``, the lot
# that brings the module when the registry records the bridge before it lands (the arriving lot
# removes the key: the list does not grow, MG6).
BRIDGE_KEYS = {"word", "module", "route", "owner"}
BRIDGE_CONDITIONS = {"arrives", "refusal", "exit_test"}


class RegistryError(ValueError):
    """The registry or the tree breaks a rule of MG9."""


def load(text: str) -> dict:
    data = tomllib.loads(text)
    if data.get("schema_version") != 1 or data.get("gate") != GATE:
        raise RegistryError("%s: schema_version 1 and gate %r are required" % (REGISTRY, GATE))
    bridges = data.get("bridge", [])
    earlier = data.get("earlier", [])
    for entry in bridges:
        if not BRIDGE_KEYS <= set(entry) <= BRIDGE_KEYS | BRIDGE_CONDITIONS:
            raise RegistryError("a [[bridge]] entry has word, module, route and owner, and may "
                                "have arrives, refusal and exit_test: %r" % (entry,))
    for entry in earlier:
        if set(entry) != {"modules", "words", "owner"} or not entry["modules"]:
            raise RegistryError("an [[earlier]] entry has modules, words and owner: %r"
                                % (entry,))
    modules = [module for entry in earlier for module in entry["modules"]]
    if len(modules) != len(set(modules)):
        raise RegistryError("a module is listed by two [[earlier]] entries")
    return {"bridge": bridges, "earlier": earlier}


def read(root: Path = ROOT) -> dict:
    return load((root / REGISTRY).read_text(encoding="utf-8"))


def library_modules(root: Path = ROOT) -> dict:
    """{dotted name: path} of every module of ``pops.lib``."""
    base = root / "python"
    modules = {}
    for path in sorted((root / LIBRARY).rglob("*.py")):
        parts = list(path.relative_to(base).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def is_surface(source: str) -> bool:
    return "import pops.surface" in source or "from pops.surface" in source


def imports_no_pops(source: str) -> bool:
    """Whether a module imports no module of ``pops`` (a writer of ``pops.lib.output``, in plain
    Python and NumPy). Such a module is of the surface unless ``[[earlier]]`` lists it: a body
    over b6's continuation, like ``pops.lib.time.when``, holds a b6 signature without an import."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0]
                                                 == "pops"):
            return False
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "pops"
                                                for alias in node.names):
            return False
    return True


def is_package_index(tree: ast.Module) -> bool:
    """Whether a module holds only a docstring, relative imports of its submodules and
    ``__all__``."""
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
                "__all__"]:
            continue
        return False
    return True


def classify(root: Path = ROOT, registry: dict | None = None) -> dict:
    """{dotted name: "surface" | "earlier" | "package" | "unlisted"}."""
    registry = read(root) if registry is None else registry
    earlier = {module for entry in registry["earlier"] for module in entry["modules"]}
    kinds = {}
    for name, path in library_modules(root).items():
        source = path.read_text(encoding="utf-8")
        if is_surface(source):
            kinds[name] = "surface"
        elif name in earlier:
            kinds[name] = "earlier"
        elif is_package_index(ast.parse(source)):
            kinds[name] = "package"
        elif imports_no_pops(source):
            kinds[name] = "surface"
        else:
            kinds[name] = "unlisted"
    return kinds


def bridges_of(source: str, module: str) -> set:
    """{(word, module, route)}: the bridges that a module of the surface builds. ``word`` is the
    function or class of the module that builds the route."""
    tree = ast.parse(source)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                and node.module.startswith(NATIVE_MODULES):
            for alias in node.names:
                imported[alias.asname or alias.name] = "%s.%s" % (node.module, alias.name)
    found = set()

    def visit(node: ast.AST, word: str) -> None:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if name == "SchemeValue":
                    found.add((node.name, module, "SchemeValue"))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and word == "<module>":
            word = node.name
        if isinstance(node, ast.Call):
            route = None
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id in ROUTE_MODULES and func.attr not in WITH_BODY:
                route = "%s.%s" % (func.value.id, func.attr)
            elif isinstance(func, ast.Name) and func.id in imported \
                    and func.id not in WITH_BODY and func.id not in ROUTE_MODULES:
                route = imported[func.id]
            if route is not None:
                found.add((word, module, route))
        for child in ast.iter_child_nodes(node):
            visit(child, word)

    visit(tree, "<module>")
    # a public function that returns a private builder of a route is the word of the bridge
    # (cg() returns _CG(...), whose method calls pops.solvers.CG)
    private = {word for word, _, _ in found if word.startswith("_")}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            called = {call.func.id for call in ast.walk(node) if isinstance(call, ast.Call)
                      and isinstance(call.func, ast.Name) and call.func.id in private}
            for word, where, route in list(found):
                if word in called:
                    found.add((node.name, where, route))
                    found.discard((word, where, route))
    return found


def found_bridges(root: Path = ROOT, kinds: dict | None = None) -> set:
    kinds = classify(root) if kinds is None else kinds
    modules = library_modules(root)
    found = set()
    for name, kind in kinds.items():
        if kind == "surface":
            found |= bridges_of(modules[name].read_text(encoding="utf-8"), name)
    return found


def check(root: Path = ROOT) -> list:
    """The breaches of rules 1 and 2 of MG9 in the tree: an unlisted module, a listed module that
    is gone or written with the surface, and a bridge found or listed but not both."""
    registry = read(root)
    kinds = classify(root, registry)
    problems = ["%s is neither a module of the surface, nor [[earlier]], nor a package that "
                "re-exports its submodules" % name
                for name, kind in sorted(kinds.items()) if kind == "unlisted"]
    for entry in registry["earlier"]:
        for module in entry["modules"]:
            if module not in kinds:
                problems.append("[[earlier]] lists %s, which pops.lib no longer has: remove it"
                                % module)
            elif kinds[module] == "surface":
                problems.append("[[earlier]] lists %s, which is written with the surface: remove "
                                "it" % module)
    found = {(word, module) for word, module, _ in found_bridges(root, kinds)}
    listed = set()
    for entry in registry["bridge"]:
        item = (entry["word"], entry["module"])
        if entry["module"] not in kinds:
            if "arrives" not in entry:
                problems.append("[[bridge]] lists %s of %s, which pops.lib does not have: remove "
                                "it, or state the lot that brings it with arrives=" % item)
            continue
        if "arrives" in entry:
            problems.append("[[bridge]] %s of %s has arrived: remove its arrives=" % item)
        if "refusal" in entry:
            path, _, name = entry["refusal"].partition("::")
            test = root / path
            if not name or not test.is_file() or \
                    ("def %s(" % name) not in test.read_text(encoding="utf-8"):
                problems.append("[[bridge]] %s of %s names the refusal %s, which the tree does "
                                "not have" % (item + (entry["refusal"],)))
        listed.add(item)
    problems.extend("%s of %s returns a native route and is not a [[bridge]]" % item
                    for item in sorted(found - listed))
    problems.extend("[[bridge]] lists %s of %s, which no longer returns a native route: remove it"
                    % item for item in sorted(listed - found))
    return problems


def _shown(root: Path, revision: str) -> str | None:
    shown = subprocess.run(["git", "-C", str(root), "show", "%s:%s" % (revision, REGISTRY)],
                           capture_output=True, text=True)
    if shown.returncode != 0:
        exists = subprocess.run(["git", "-C", str(root), "cat-file", "-e",
                                 revision + "^{commit}"], capture_output=True)
        if exists.returncode != 0:
            raise RegistryError("the base revision %s is not in the repository" % revision)
        return None
    return shown.stdout


def grown(current: dict, base: dict) -> list:
    """What the registry records beyond its base (MG6): a bridge, an earlier module or a word."""
    problems = []
    base_bridges = {(entry["word"], entry["module"]) for entry in base["bridge"]}
    for entry in current["bridge"]:
        if (entry["word"], entry["module"]) not in base_bridges:
            problems.append("[[bridge]] %s of %s is not in the base" % (entry["word"],
                                                                         entry["module"]))
    base_words = {module: set(entry["words"]) for entry in base["earlier"]
                  for module in entry["modules"]}
    for entry in current["earlier"]:
        for module in entry["modules"]:
            if module not in base_words:
                problems.append("[[earlier]] %s is not in the base" % module)
            elif set(entry["words"]) - base_words[module]:
                problems.append("[[earlier]] %s gains the words %s" % (
                    module, ", ".join(sorted(set(entry["words"]) - base_words[module]))))
    return problems


def check_base(revision: str, root: Path = ROOT) -> list:
    text = _shown(root, revision)
    if text is None:
        return []
    return grown(read(root), load(text))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("check", help="check the registry against the tree")
    command.add_argument("--check-base", metavar="REVISION",
                         help="also fail when the registry records more than at REVISION (MG6)")
    command.add_argument("--source", default=str(ROOT), help="the root of the tree")
    arguments = parser.parse_args(argv)
    root = Path(arguments.source).resolve()
    try:
        problems = check(root)
        if arguments.check_base:
            problems += check_base(arguments.check_base, root)
    except RegistryError as error:
        problems = [str(error)]
    for problem in problems:
        print("MG9: %s" % problem, file=sys.stderr)
    if not problems:
        print("MG9: %s holds every bridge and every earlier module of pops.lib" % REGISTRY)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
