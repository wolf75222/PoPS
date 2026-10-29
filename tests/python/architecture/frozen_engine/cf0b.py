"""CF0b (spec 0.3, section 16.1; A4, A7): the library is outside the core, and stays honest.

``python/pops/lib/`` (the one path that CF0 excludes from the core, ``cf0_paths.toml``) imports only
the public ``pops`` surface: no module or name whose name starts with ``_``, nothing from
``pops.codegen``, ``pops.runtime`` or the native bindings ``pops._pops``. It reaches none of the C++
entry routes of A7 (``pops.external``, ``NativeFunction`` and ``NativeDerivative``,
``PreparedNativeComponent``, ``load_cpp_library``, a brick id passed to ``User``), holds no C++ file
and no C++ source string, imports nothing from the native component tree, and holds no private
module of its own.

The same intention covers the objects that the library receives from the core (A4: a library
object can be written in a case, with the same identity, and a case reads no ``model._m``): the
library neither reads nor writes a private attribute of a core object, as ``obj._x`` or as
``getattr(obj, "_x")``. The private attributes that the library defines itself (``self._x``, a
class attribute or a method of a library class) stay the library's own business.

The scan reads source only. Imports inside ``pops.lib`` itself are the library's own business (a
private module of the library is one site, at its path), so ``from ._factory import x`` adds no site
beyond ``lib/time/_factory.py``.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib

from . import pysource, ratchet, scan

EXCEPTIONS = scan.HERE / "cf0b_exceptions.txt"

# The C++ entry routes of A7 (gap C37) and the modules that define them at b6e64c1d.
ROUTE_NAMES = frozenset({"NativeFunction", "NativeDerivative", "PreparedNativeComponent",
                         "load_cpp_library"})
ROUTE_MODULES = frozenset({"pops.native_calls", "pops.native_components"})
EXTERNAL = "external"  # refused only where it resolves into pops: the word itself is generic
BINDINGS = "pops._pops"
# Names that start with one underscore but are the documented public API of a Python type: the
# methods and fields of a namedtuple and the members of an Enum.
PUBLIC_UNDERSCORE_NAMES = frozenset({
    "_asdict", "_field_defaults", "_fields", "_make", "_replace",
    "_generate_next_value_", "_ignore_", "_missing_", "_name_", "_order_", "_value_"})
_ATTRIBUTE_CALLS = frozenset({"getattr", "setattr", "hasattr", "delattr"})

TERMS = {
    "private-import": "imports a module or a name of the core whose name starts with _",
    "private-access": "reaches a private attribute of a core module",
    "private-attribute": "reads or writes a private attribute of an object of the core (A4)",
    "codegen-import": "imports pops.codegen",
    "runtime-import": "imports pops.runtime",
    "bindings-import": "imports the native bindings pops._pops",
    "private-module": "a private module of the library itself",
    "native-route-module": "imports a module of the C++ entry routes (A7, C37)",
    "external": "reaches pops.external (A7)",
    "brick-id": "selects an external C++ brick by id through User(\"id\") (A7, C37)",
    "cpp-file": "a C++ file in the library",
    "cpp-source": "a C++ source string",
    "cpp-file-reference": "names a C++ file",
    "native-tree-import": "imports the native component tree",
    "native-tree-path": "names a path of the native component tree",
}
TERMS.update({name: "reaches %s, a C++ entry route (A7, C37)" % name for name in ROUTE_NAMES})


@dataclasses.dataclass(frozen=True, order=True)
class Site:
    path: str
    line: int
    term: str
    excerpt: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.term


def library_root(engine: scan.FrozenEngine | None = None) -> str:
    """The library path: the one directory that CF0 excludes from the core paths."""
    engine = engine or scan.load_cf0()
    (root,) = engine.core_exclude  # CF0 excludes the library and nothing else
    return root


def _in_library(module: str, library_module: str) -> bool:
    return module == library_module or module.startswith(library_module + ".")


def _import_terms(record: pysource.Import, library_module: str, native_top: str):
    module, names = record.module, record.names
    if module.split(".")[0] == native_top:
        yield "native-tree-import"
        return
    targets = [module] if not names else ["%s.%s" % (module, name) for name in names]
    for target in targets:
        target_parts = target.split(".")
        if target_parts[0] != "pops":
            continue
        # A C++ entry route is refused wherever it is spelled, pops.lib.load_cpp_library included.
        if target_parts[-1] in ROUTE_NAMES:
            yield target_parts[-1]
        elif target_parts[-1] == EXTERNAL:
            yield "external"
        elif target in ROUTE_MODULES:
            yield "native-route-module"
        elif _in_library(target, library_module):
            continue  # the library's own modules: a private one is a site at its path
        elif target == BINDINGS or target.startswith(BINDINGS + "."):
            yield "bindings-import"
        elif target_parts[:2] == ["pops", "codegen"]:
            yield "codegen-import"
        elif target_parts[:2] == ["pops", "runtime"]:
            yield "runtime-import"
        elif any(pysource.is_private_part(part) for part in target_parts):
            yield "private-import"


def route_terms(node: ast.AST, names: pysource.Names, library_module: str):
    """Terms of one reference (a ``Name`` or ``Attribute`` load) to a C++ entry route or to a
    private attribute of a core module."""
    dotted = names.dotted(node)
    if dotted is None:
        return
    parts = dotted.split(".")
    if isinstance(node, ast.Name):
        if parts[-1] in ROUTE_NAMES and (len(parts) == 1 or parts[0] == "pops"):
            yield parts[-1]  # bound to the route, or a bare name distinctive enough alone
        elif parts[0] == "pops" and parts[-1] == EXTERNAL:
            yield "external"
        return
    if parts[0] != "pops":
        return
    if parts[-1] in ROUTE_NAMES:
        yield parts[-1]
    elif parts[-1] == EXTERNAL:
        yield "external"
    elif dotted in ROUTE_MODULES:
        yield "native-route-module"
    elif pysource.is_private_part(parts[-1]) and not _in_library(dotted, library_module):
        yield "private-access"


def library_attributes(trees) -> frozenset[str]:
    """The private attribute names that the library defines itself: stored on ``self`` or ``cls``,
    bound in a class body (methods, class attributes, ``__slots__``)."""
    defined = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) \
                    and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                defined.add(node.attr)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        defined.add(item.name)
                    targets = item.targets if isinstance(item, ast.Assign) else \
                        [item.target] if isinstance(item, ast.AnnAssign) else []
                    for target in targets:
                        if isinstance(target, ast.Name):
                            defined.add(target.id)
                            if target.id == "__slots__" and isinstance(item.value, (
                                    ast.Tuple, ast.List)):
                                defined.update(element.value for element in item.value.elts
                                               if isinstance(element, ast.Constant)
                                               and isinstance(element.value, str))
    return frozenset(name for name in defined if pysource.is_private_part(name))


def _private_name(name: str) -> bool:
    return pysource.is_private_part(name) and name not in PUBLIC_UNDERSCORE_NAMES


def _object_term(base: ast.AST, attribute: str, names: pysource.Names, local: set[str],
                 defined: frozenset[str], library_module: str, *, module_access: bool):
    """The term of a private @p attribute reached on @p base, or None. A base that resolves into
    ``pops`` is a core module or a name imported from one: it is reported as private-access when
    @p module_access (a store, or ``getattr(module, "_x")``), since route_terms reports the
    dotted loads."""
    if isinstance(base, ast.Name) and base.id in ("self", "cls"):
        return None
    dotted = names.dotted(base)
    if dotted is not None and dotted.split(".")[0] == "pops":
        if module_access and not _in_library(dotted, library_module):
            return "private-access"
        return None
    if isinstance(base, ast.Name) and (base.id in names.bound or base.id in local):
        return None  # a module or name of another package, or a class of this module
    return None if attribute in defined else "private-attribute"


def private_attribute_terms(tree: ast.AST, names: pysource.Names, defined: frozenset[str],
                            library_module: str):
    """(line, term) of every read or write of a private attribute of a core object."""
    local = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private_name(node.attr):
            term = _object_term(node.value, node.attr, names, local, defined, library_module,
                                module_access=not isinstance(node.ctx, ast.Load))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _ATTRIBUTE_CALLS and len(node.args) >= 2 \
                and isinstance(node.args[1], ast.Constant) \
                and isinstance(node.args[1].value, str) and _private_name(node.args[1].value):
            term = _object_term(node.args[0], node.args[1].value, names, local, defined,
                                library_module, module_access=True)
        else:
            continue
        if term:
            yield node.lineno, term


def brick_id_call(node: ast.Call, names: pysource.Names) -> bool:
    """``User("id")``: a pops ``User`` selector called with a brick id instead of a body."""
    dotted = names.dotted(node.func)
    if dotted is None or not dotted.startswith("pops.") or dotted.rsplit(".", 1)[-1] != "User":
        return False
    return bool(node.args) and isinstance(node.args[0], ast.Constant) \
        and isinstance(node.args[0].value, str)


def python_sites(rel: str, text: str, *, library_module: str, native_tree: str,
                 library_attributes: frozenset[str] | None = None):
    """(line, term) of every CF0b-kind site of one Python file (CF8 reuses it for case files).

    With @p library_attributes (the private attribute names that the library defines), the reads
    and writes of private attributes of core objects are sites too.
    """
    tree = ast.parse(text)
    native_top = native_tree.strip("/").split("/")[0]
    names = pysource.Names(tree, rel)
    if library_attributes is not None:
        yield from private_attribute_terms(tree, names, library_attributes, library_module)
    for record in pysource.imports(tree, rel):
        for term in _import_terms(record, library_module, native_top):
            yield record.line, term
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            for term in route_terms(node, names, library_module):
                yield getattr(node, "end_lineno", None) or node.lineno, term
        if isinstance(node, ast.Call) and brick_id_call(node, names):
            yield node.lineno, "brick-id"
    for line, value in pysource.code_strings(tree):
        if pysource.looks_like_cpp(value):
            yield line, "cpp-source"
        elif pysource.names_cpp_file(value):
            yield line, "cpp-file-reference"
        elif value.startswith(native_tree) or ("/" + native_tree) in value:
            yield line, "native-tree-path"


def _excerpt(lines, line):
    if line == 0:
        return "(path)"
    return " ".join(lines[line - 1].split())[:110] if 1 <= line <= len(lines) else ""


def scan_library(root: pathlib.Path = scan.REPO_ROOT, engine: scan.FrozenEngine | None = None):
    """Every CF0b site of the checkout."""
    engine = engine or scan.load_cf0()
    library = library_root(engine)
    library_module = pysource.module_name(library.rstrip("/"))
    files = [rel for rel in scan.tracked_files(root) if rel.startswith(library)]
    texts = {rel: (root / rel).read_text(encoding="utf-8") for rel in files if rel.endswith(".py")}
    defined = library_attributes(ast.parse(text) for text in texts.values())
    sites: set[Site] = set()
    for rel in files:
        inner = pathlib.PurePosixPath(rel[len(library):])
        if pysource.is_cpp_file(rel):
            sites.add(Site(rel, 0, "cpp-file", "(path)"))
        if any(pysource.is_private_part(part) for part in inner.with_suffix("").parts) \
                and rel.endswith(".py"):
            sites.add(Site(rel, 0, "private-module", "(path)"))
        if rel not in texts:
            continue
        lines = texts[rel].splitlines()
        for line, term in python_sites(rel, texts[rel], library_module=library_module,
                                       native_tree=engine.native_tree,
                                       library_attributes=defined):
            sites.add(Site(rel, line, term, _excerpt(lines, line)))
    return sorted(sites)


# (issue, library sub-paths or None, reason): the first matching rule names the issue that removes a
# site. The spec empties this list at the Phase 1 exit (section 18) without naming an issue; these
# owners are the Phase 1 issues that rewrite each part of the library.
OWNER_RULES = (
    ("N12", ("python/pops/lib/amr/",),
     "the tagging-program ABI of the tag-graph VM, removed with it (C36), Phase 1"),
    ("N8", ("python/pops/lib/time/", "python/pops/lib/initial/"),
     "library time programs and initial data rebuilt over the public Pi surface (section 5), "
     "Phase 1"),
    ("N3", None, "the library catalogue (section 2.6.9), Phase 1"),
)


def owner_of(path: str) -> str:
    for issue, paths, _ in OWNER_RULES:
        if paths is None or scan.covered(path, paths):
            return issue
    raise AssertionError("OWNER_RULES has no default")


def current_entries(root: pathlib.Path = scan.REPO_ROOT, engine=None):
    return [ratchet.Entry(site.path, site.line, site.term, owner_of(site.path), site.excerpt)
            for site in scan_library(root, engine)]
