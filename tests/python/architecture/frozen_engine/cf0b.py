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
``getattr(obj, "_x")``. The private attributes that a library module defines itself (``self._x``, a
class attribute or a method of one of its classes) stay that module's business. The set is taken
per module, so a name that one module defines never hides the same name on a core object in
another.

The scan reads source only. Imports inside ``pops.lib`` itself are the library's own business (a
private module of the library is one site, at its path), so ``from ._factory import x`` adds no site
beyond ``lib/time/_factory.py``.

A site is one occurrence of a term, and its excerpt is its spelling: the dotted name of an import
or a reference (the module when the module alone is a site), the attribute (``._frame``) or the
string literal. The ratchet pairs sites with list entries by spelling within each (path, term),
counted with multiplicity, so an edit elsewhere on a listed line or a re-wrap keeps the site, while
a new spelling, or one more occurrence of a listed one, is a new site.
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


def _target_term(target: str, library_module: str) -> str | None:
    """The term of an imported dotted name, or None."""
    parts = target.split(".")
    if parts[0] != "pops":
        return None
    # A C++ entry route is refused wherever it is spelled, pops.lib.load_cpp_library included.
    if parts[-1] in ROUTE_NAMES:
        return parts[-1]
    if parts[-1] == EXTERNAL:
        return "external"
    if target in ROUTE_MODULES:
        return "native-route-module"
    if _in_library(target, library_module):
        return None  # the library's own modules: a private one is a site at its path
    if target == BINDINGS or target.startswith(BINDINGS + "."):
        return "bindings-import"
    if parts[:2] == ["pops", "codegen"]:
        return "codegen-import"
    if parts[:2] == ["pops", "runtime"]:
        return "runtime-import"
    if any(pysource.is_private_part(part) for part in parts):
        return "private-import"
    return None


def _import_terms(record: pysource.Import, library_module: str, native_top: str):
    """(term, spelling) of each site of one import. The spelling is the dotted name that holds the
    term: the module when the module alone is a site (once, however many names the statement
    takes from it), the module and the name otherwise."""
    module, names = record.module, record.names
    if module.split(".")[0] == native_top:
        yield "native-tree-import", module
        return
    module_term = _target_term(module, library_module) if names else None
    spelled = set()
    for target in [module] if not names else ["%s.%s" % (module, name) for name in names]:
        term = _target_term(target, library_module)
        spelling = module if term == module_term else target
        if term and (term, spelling) not in spelled:
            spelled.add((term, spelling))
            yield term, spelling


def route_terms(node: ast.AST, names: pysource.Names, library_module: str):
    """(term, spelling) of one reference (a ``Name`` or ``Attribute`` load) to a C++ entry route or
    to a private attribute of a core module. The spelling is the dotted name of the reference."""
    dotted = names.dotted(node)
    if dotted is None:
        return
    parts = dotted.split(".")
    if isinstance(node, ast.Name):
        if parts[-1] in ROUTE_NAMES and (len(parts) == 1 or parts[0] == "pops"):
            yield parts[-1], dotted  # bound to the route, or a bare name distinctive enough alone
        elif parts[0] == "pops" and parts[-1] == EXTERNAL:
            yield "external", dotted
        return
    if parts[0] != "pops":
        return
    if parts[-1] in ROUTE_NAMES:
        yield parts[-1], dotted
    elif parts[-1] == EXTERNAL:
        yield "external", dotted
    elif dotted in ROUTE_MODULES:
        yield "native-route-module", dotted
    elif pysource.is_private_part(parts[-1]) and not _in_library(dotted, library_module):
        yield "private-access", dotted


def module_attributes(tree: ast.AST) -> frozenset[str]:
    """The private attribute names that one library module defines itself: stored on ``self`` or
    ``cls``, bound in a class body (methods, class attributes, ``__slots__``)."""
    defined = set()
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
    """(line, term, spelling) of every read or write of a private attribute of a core object. The
    spelling is the attribute (``._frame``, whether read as ``model._frame`` or through
    ``getattr``), or the dotted attribute of a core module."""
    local = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private_name(node.attr):
            base, attribute = node.value, node.attr
            term = _object_term(base, attribute, names, local, defined, library_module,
                                module_access=not isinstance(node.ctx, ast.Load))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _ATTRIBUTE_CALLS and len(node.args) >= 2 \
                and isinstance(node.args[1], ast.Constant) \
                and isinstance(node.args[1].value, str) and _private_name(node.args[1].value):
            base, attribute = node.args[0], node.args[1].value
            term = _object_term(base, attribute, names, local, defined, library_module,
                                module_access=True)
        else:
            continue
        if term:
            owner = names.dotted(base) if term == "private-access" else ""
            yield node.lineno, term, "%s.%s" % (owner, attribute)


def brick_id_call(node: ast.Call, names: pysource.Names) -> bool:
    """``User("id")``: a pops ``User`` selector called with a brick id instead of a body."""
    dotted = names.dotted(node.func)
    if dotted is None or not dotted.startswith("pops.") or dotted.rsplit(".", 1)[-1] != "User":
        return False
    return bool(node.args) and isinstance(node.args[0], ast.Constant) \
        and isinstance(node.args[0].value, str)


def python_sites(rel: str, text: str, *, library_module: str, native_tree: str,
                 private_attributes: bool = False):
    """(line, term, spelling) of every CF0b-kind site of one Python file (CF8 reuses it for case
    files). The spelling is what holds the term: a dotted name, an attribute or a string literal.

    With @p private_attributes, the reads and writes of private attributes of core objects are
    sites too, except for the names that this module defines itself.
    """
    tree = ast.parse(text)
    native_top = native_tree.strip("/").split("/")[0]
    names = pysource.Names(tree, rel)
    if private_attributes:
        yield from private_attribute_terms(tree, names, module_attributes(tree), library_module)
    for record in pysource.imports(tree, rel):
        for term, spelling in _import_terms(record, library_module, native_top):
            yield record.line, term, spelling
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            for term, spelling in route_terms(node, names, library_module):
                yield getattr(node, "end_lineno", None) or node.lineno, term, spelling
        if isinstance(node, ast.Call) and brick_id_call(node, names):
            yield node.lineno, "brick-id", 'User("%s")' % node.args[0].value
    for line, value in pysource.code_strings(tree):
        if pysource.looks_like_cpp(value):
            yield line, "cpp-source", pysource.string_spelling(value)
        elif pysource.names_cpp_file(value):
            yield line, "cpp-file-reference", pysource.string_spelling(value)
        elif value.startswith(native_tree) or ("/" + native_tree) in value:
            yield line, "native-tree-path", pysource.string_spelling(value)


def scan_library(root: pathlib.Path = scan.REPO_ROOT, engine: scan.FrozenEngine | None = None):
    """Every CF0b site of the checkout: one per occurrence of a term."""
    engine = engine or scan.load_cf0()
    library = library_root(engine)
    library_module = pysource.module_name(library.rstrip("/"))
    files = [rel for rel in scan.tracked_files(root) if rel.startswith(library)]
    texts = {rel: (root / rel).read_text(encoding="utf-8") for rel in files if rel.endswith(".py")}
    sites: list[Site] = []
    for rel in files:
        inner = pathlib.PurePosixPath(rel[len(library):])
        if pysource.is_cpp_file(rel):
            sites.append(Site(rel, 0, "cpp-file", "(path)"))
        if any(pysource.is_private_part(part) for part in inner.with_suffix("").parts) \
                and rel.endswith(".py"):
            sites.append(Site(rel, 0, "private-module", "(path)"))
        if rel not in texts:
            continue
        for line, term, spelling in python_sites(rel, texts[rel], library_module=library_module,
                                                 native_tree=engine.native_tree,
                                                 private_attributes=True):
            sites.append(Site(rel, line, term, pysource.site_excerpt(spelling)))
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
