"""Python source helpers shared by the CF0b and CF8 scans (spec 0.3, sections 16.1 and 16.2).

Both scans read source only: they never import ``pops``, ``_pops`` or a case. They need to know what
a name in a module refers to (``riemann.User`` after ``from pops.numerics import riemann`` is
``pops.numerics.riemann.User``), which imports a module makes (top level or inside a function,
relative or absolute, through ``importlib.import_module``), which string literals are code rather
than docstrings or error text, and whether a string holds C++ source. ``site_excerpt`` writes the
excerpt of a site of CF0b, CF7 or CF8 from its spellings.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import re

from . import scan

CPP_SUFFIXES = frozenset(scan.CPP_SUFFIXES) | {".cuh", ".hh", ".hxx", ".ipp", ".tpp"}


def is_cpp_file(rel: str) -> bool:
    """A C++ source or header, including a template such as ``kernel.cpp.in``."""
    name = pathlib.PurePosixPath(rel).name
    stem = name[:-3] if name.endswith(".in") else name
    return pathlib.PurePosixPath(stem).suffix in CPP_SUFFIXES


def module_name(rel: str) -> str:
    """The dotted module of a repository-relative ``.py`` path (``python/`` is the package root)."""
    path = pathlib.PurePosixPath(rel)
    parts = list(path.with_suffix("").parts)
    if parts and parts[0] == "python":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_of(rel: str) -> str:
    """The package that a relative import in @p rel is resolved against."""
    module = module_name(rel)
    if pathlib.PurePosixPath(rel).name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def is_private_part(part: str) -> bool:
    """``_x`` is private; dunder names (``__future__``, ``__init__``) are not."""
    return part.startswith("_") and not (part.startswith("__") and part.endswith("__"))


@dataclasses.dataclass(frozen=True)
class Import:
    line: int
    module: str  # the module that the statement loads
    names: tuple[str, ...]  # names taken from it by ``from module import ...``; () for ``import``


def _resolve_relative(package: str, level: int, module: str | None) -> str:
    base = package.split(".") if package else []
    if level > 1:
        base = base[:len(base) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def imports(tree: ast.AST, rel: str):
    """Every import of a module, wherever it is written, with relative imports resolved."""
    package = package_of(rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Import(node.lineno, alias.name, ())
        elif isinstance(node, ast.ImportFrom):
            module = node.module if not node.level else _resolve_relative(
                package, node.level, node.module)
            yield Import(node.lineno, module or "", tuple(alias.name for alias in node.names))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            callee = call_name(node.func)
            if callee in ("import_module", "__import__"):
                yield Import(node.lineno, node.args[0].value, ())


class Names:
    """What each name bound by an import refers to, as a dotted path.

    The table is flat (one scope): a name rebound in a function shadows nothing here. It is used
    only to spell the dotted path of an attribute chain, so a stale binding errs towards
    reporting.
    """

    def __init__(self, tree: ast.AST, rel: str):
        self.bound: dict[str, str] = {}
        package = package_of(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.bound[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        self.bound[head] = head
            elif isinstance(node, ast.ImportFrom):
                module = node.module if not node.level else _resolve_relative(
                    package, node.level, node.module)
                for alias in node.names:
                    if alias.name != "*":
                        self.bound[alias.asname or alias.name] = \
                            "%s.%s" % (module, alias.name) if module else alias.name

    def dotted(self, node: ast.AST) -> str | None:
        """The dotted path of a ``Name`` or of an attribute chain on a name, or None."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        head = self.bound.get(node.id, node.id)
        return ".".join([head] + parts[::-1])


def call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def site_excerpt(spellings) -> str:
    """The excerpt of a site of an exception list: the distinct spellings of its term on its line,
    sorted, as CF2 writes them (``scan.cf2_sites_in_text``). The ratchet pairs a site with an entry
    by it, so an edit elsewhere on the line keeps the site, and a new spelling is a new site."""
    return " ".join(sorted(set(spellings)))[:scan.EXCERPT_WIDTH].rstrip()


def string_spelling(value: str) -> str:
    return '"%s"' % " ".join(value.split())


def code_strings(tree: ast.AST):
    """(line, value) of every string literal that is code: not a docstring, a bare string statement
    or the text of an error (CF2's reading of "outside comments and user-facing error text")."""
    excluded = scan._python_excluded_strings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in excluded:
            yield node.lineno, node.value


# C++ source in a string: a preprocessor directive, a C++-only keyword or construct, or a function
# body with statements. Any one strong marker is enough; a qualified name alone ("pops::Rusanov",
# a native target id) is not C++ source.
_CPP_STRONG = re.compile(
    r"^\s*#\s*(include|pragma|define|if|ifdef|ifndef)\b"
    r"|\bextern\s+\"C\""
    r"|\btemplate\s*<"
    r"|\bnamespace\s+[A-Za-z_]\w*\s*\{"
    r"|\bstd::\w+"
    r"|\b(static|reinterpret|const|dynamic)_cast\s*<"
    r"|\bconstexpr\b"
    r"|\bKOKKOS_(INLINE_FUNCTION|FUNCTION|LAMBDA|CLASS_LAMBDA)\b"
    r"|\bPOPS_(HD|INLINE|DEVICE)\b"
    r"|__(global|device|host)__",
    re.MULTILINE)
_CPP_BODY = re.compile(r"\{[^{}]*\breturn\b[^;{}]*;[^{}]*\}", re.DOTALL)


def looks_like_cpp(text: str) -> bool:
    return bool(_CPP_STRONG.search(text) or _CPP_BODY.search(text))


def names_cpp_file(text: str) -> bool:
    """A string that is the path of a C++ file (``"kernel.cpp"``, ``"flux.hpp"``)."""
    return bool(re.fullmatch(r"[\w./\-]+", text)) and is_cpp_file(text)
