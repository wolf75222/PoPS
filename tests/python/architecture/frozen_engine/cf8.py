"""CF8 (spec 0.3, section 16.2; B2, B3, B5, A7): no infrastructure and no native code in case files.

A source scan of the conformance cases, litmus cases, probes and the final witness (the paths of
``cases.toml``) refuses:

* the sizes that the plan derives (B2, section 12.1): halo widths, stencil radii, component counts
  (``components=``, ``field_components=``), dimensions given as an integer to a call, history
  depths and the slots of a history ring, written by the author;
* the runtime resources (B3, B5): memory and metadata budgets, byte capacities, buffer sizes,
  storage sizes and every ``capacity=``;
* C++ files, C++ source strings and the names of C++ files;
* the C++ entry routes of A7 (gap C37): ``pops.external``, ``NativeFunction`` and
  ``NativeDerivative``, ``PreparedNativeComponent``, ``load_cpp_library`` and a brick id passed to
  ``User`` (``riemann.User("id")``), and imports from the native component tree.

A method budget (a Newton or Krylov iteration limit, a retry count, ``budget=`` of a solve) is part
of the method and stays in the case (B1, section 12.1), and so do a physical quantity
(``dimension=pops.Dimension(...)``, a heat capacity) and the parameter of a builder that declares
its support (``def build(dimension=2)``). A size is reported where the author writes
it: a keyword argument, a parameter default, an attribute assignment or a key of a ``**{...}``
argument. Library objects reach no C++ entry route either (CF0b), so a case cannot reach one through
``pops.lib``. The exception list of b6 (``cf8_exceptions.txt``) only shrinks (MG6).

A site is one occurrence of a term, and its excerpt is its spelling: where the size is written,
without its value (``StencilFlux(radius=)``, ``def build(ghost_depth=)``, ``.capacity =``), or the
spelling of CF0b for native code. The ratchet pairs sites with list entries by spelling within each
(path, term), counted with multiplicity, so a case may edit or re-wrap a listed line, and change a
listed value, until the issue that owns the site removes it; the same size written in another call,
or once more in the same call, is a new site.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import tomllib

from . import cf0b, pysource, ratchet, scan

CASES = scan.HERE / "cases.toml"
EXCEPTIONS = scan.HERE / "cf8_exceptions.txt"

# The CF0b kinds of site that CF8 also refuses in case files (A7, CF8); private imports and imports
# of pops.codegen or pops.runtime are the library's rule (CF0b, A4), not a case's.
NATIVE_TERMS = frozenset(cf0b.ROUTE_NAMES) | {
    "external", "brick-id", "native-route-module", "cpp-source", "cpp-file-reference",
    "native-tree-import", "native-tree-path"}

_SIZE_WORDS = frozenset({"depth", "width", "cells", "layers", "size", "count", "extent", "extents"})
_STENCIL_CALLEES = frozenset({"stencil", "flux", "face", "reconstruction", "footprint"})
_COUNT_NAMES = frozenset({"n_components", "num_components", "component_count", "ncomp", "n_comp",
                          "ncomponents", "species_count", "n_species", "num_species"})
_HISTORY_DEPTH_NAMES = frozenset({"history_depth", "history_size", "history_length"})
_RESOURCE_BUDGET_WORDS = frozenset({"memory", "metadata", "byte", "bytes", "storage", "buffer",
                                    "workspace", "expression"})
# A capacity with one of these words is a physical quantity (a heat capacity), not the capacity of
# a set or of a storage (B5).
_PHYSICAL_CAPACITY_WORDS = frozenset({"heat", "thermal", "specific", "molar", "isobaric",
                                      "isochoric", "volumetric"})
_HISTORY_SLOT_NAMES = frozenset({"slots", "source_slots", "target_slots"})


def load_case_paths(path: pathlib.Path = CASES) -> tuple[str, ...]:
    return tuple(tomllib.loads(path.read_text(encoding="utf-8"))["case_paths"]["include"])


def load_cases(path: pathlib.Path = CASES) -> list[dict]:
    return tomllib.loads(path.read_text(encoding="utf-8")).get("case", [])


def _is_int_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def infrastructure_kind(name: str, value: ast.AST | None, callee: str) -> str | None:
    """The kind of infrastructure that @p name = @p value written in a call of @p callee is, or
    None.

    @p value is None for a parameter written without a default; @p callee is "" outside a call.
    """
    words = set(scan.split_words(name))
    callee_words = set(scan.split_words(callee))
    if words & {"capacity", "capacities"}:
        if words & _PHYSICAL_CAPACITY_WORDS:
            return None
        return "byte capacity" if words & {"byte", "bytes"} else "capacity"
    if words & {"byte", "bytes"}:
        return "byte capacity"
    if "budget" in words and words & _RESOURCE_BUDGET_WORDS:
        return "memory or metadata budget"
    if "buffer" in words:
        return "buffer size"
    if name.startswith("storage_"):
        return "storage size"
    if words & {"ghost", "ghosts", "halo", "nghost"} and (words & _SIZE_WORDS or words <= {
            "ghost", "ghosts", "halo", "nghost", "n", "num"}):
        return "halo width"
    if words & {"radius", "radii", "reach"} and (_is_int_literal(value)
                                                 or callee_words & _STENCIL_CALLEES
                                                 or "stencil" in words):
        return "stencil radius"
    if name in _COUNT_NAMES or (name == "components" and _is_int_literal(value)) \
            or (name == "field_components" and callee):
        return "component count"  # components=("u", "v") or {axis: ...} declares, never counts
    if name in _HISTORY_DEPTH_NAMES or (name == "depth" and "history" in callee_words) \
            or (name in _HISTORY_SLOT_NAMES and "history" in callee_words):
        return "history depth"  # HistoryTransfer(slots=3): the slots of the ring are its depth
    if name == "dimension" and callee and _is_int_literal(value):
        return "dimension"  # the rank of the support that a call reads; Dimension(...) is physical
    return None


@dataclasses.dataclass(frozen=True, order=True)
class Site:
    path: str
    line: int
    term: str
    excerpt: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.term


def infrastructure_sites(tree: ast.AST):
    """(line, name, spelling) of every infrastructure size or resource written in a module. The
    spelling is where the size is written, without its value."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = pysource.call_name(node.func)
            for keyword in node.keywords:
                if keyword.arg is not None:
                    if infrastructure_kind(keyword.arg, keyword.value, callee):
                        yield keyword.value.lineno, keyword.arg, "%s(%s=)" % (callee, keyword.arg)
                    continue
                for sub in ast.walk(keyword.value):  # **{"byte_capacity": c}
                    if isinstance(sub, ast.Dict):
                        for key, value in zip(sub.keys, sub.values, strict=True):
                            if isinstance(key, ast.Constant) and isinstance(key.value, str) \
                                    and infrastructure_kind(key.value, value, callee):
                                yield key.lineno, key.value, '%s(**{"%s"})' % (callee, key.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            defaults = [None] * (len(positional) - len(arguments.defaults)) \
                + list(arguments.defaults)
            pairs = list(zip(positional, defaults, strict=True)) \
                + list(zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True))
            function = "def %s" % node.name if not isinstance(node, ast.Lambda) else "lambda"
            for arg, default in pairs:
                if infrastructure_kind(arg.arg, default, ""):
                    yield arg.lineno, arg.arg, "%s(%s%s)" % (function, arg.arg,
                                                             "" if default is None else "=")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) \
                        and infrastructure_kind(target.attr, node.value, ""):
                    yield target.lineno, target.attr, ".%s =" % target.attr


def case_files(root: pathlib.Path = scan.REPO_ROOT, paths=None) -> list[str]:
    paths = load_case_paths() if paths is None else paths
    return [rel for rel in scan.tracked_files(root) if scan.covered(rel, paths)]


def scan_cases(root: pathlib.Path = scan.REPO_ROOT, engine: scan.FrozenEngine | None = None,
               paths=None):
    """Every CF8 site of the case files of the checkout: one per occurrence of a term."""
    engine = engine or scan.load_cf0()
    library_module = pysource.module_name(cf0b.library_root(engine).rstrip("/"))
    sites: list[Site] = []
    for rel in case_files(root, paths):
        if pysource.is_cpp_file(rel):
            sites.append(Site(rel, 0, "cpp-file", "(path)"))
        if not rel.endswith(".py"):
            continue
        text = (root / rel).read_text(encoding="utf-8")
        for line, name, spelling in infrastructure_sites(ast.parse(text)):
            sites.append(Site(rel, line, name, pysource.site_excerpt(spelling)))
        for line, term, spelling in cf0b.python_sites(rel, text, library_module=library_module,
                                                      native_tree=engine.native_tree):
            if term in NATIVE_TERMS:
                sites.append(Site(rel, line, term, pysource.site_excerpt(spelling)))
    return sorted(sites)


def kind_of(term: str) -> str:
    """The CF8 bullet that a term of the exception list falls under."""
    if term in NATIVE_TERMS or term == "cpp-file":
        return "native code (A7, C37)"
    if term == "depth" or term in _HISTORY_SLOT_NAMES:
        return "history depth"  # depth= and slots= are sites only in a history call
    return infrastructure_kind(term, ast.Constant(value=2), "stencil") or "infrastructure"


# (issue, kinds, terms or None, reason): the first rule whose kind and terms match names the issue
# whose mechanism makes the written size or resource unnecessary; the case file loses it then. The
# spec fixes no exit for CF8 before the corpus (Phase 3); these are proposals.
OWNER_RULES = (
    ("N7", ("memory or metadata budget",), frozenset({"flux_expression_budget"}),
     "the retained-expression budget of the stencil analysis (C33, IR19), Phase 1"),
    ("G07", ("memory or metadata budget",), None,
     "history persistence is a runtime resource with a default (B3, CN5), Phase 2"),
    ("N7", ("halo width", "stencil radius", "component count"), None,
     "footprints and halo widths derived from the reads (IR14, IR17; N7a pilot), Phase 1"),
    ("N7", ("dimension",), None,
     "the rank of the support that a call reads (B2): N7's typed index maps derive it for "
     "StencilFlux and TraceFlux; the gradient calls go with the field operators of G10 and the "
     "regrid_topology and continuation_forest calls with G11, Phase 1 and 2"),
    ("N8", ("history depth",), None,
     "lag(..., startup=f) replaces keep_history and its depth (section 5.12, T27), Phase 1"),
    ("G13", ("byte capacity", "buffer size"), None,
     "device-resident arrays and halo buffers sized by the plan (DM7-DM10, B3), Phase 1"),
    ("G02", ("capacity", "storage size"), None,
     "capacities by instance path, runtime resources with a plan default (B5, B7), Phase 2"),
    ("N4", ("native code (A7, C37)",), None,
     "the C++ entry routes folded into pops.external (A7, C37), Phase 2"),
    ("G02", None, None, "a runtime resource with a plan default (B3, B5), Phase 2"),
)


def owner_of(term: str) -> str:
    kind = kind_of(term)
    for issue, kinds, terms, _ in OWNER_RULES:
        if kinds is not None and kind not in kinds:
            continue
        if terms is not None and term not in terms:
            continue
        return issue
    raise AssertionError("OWNER_RULES has no default")


def current_entries(root: pathlib.Path = scan.REPO_ROOT, engine=None):
    return [ratchet.Entry(site.path, site.line, site.term, owner_of(site.term), site.excerpt)
            for site in scan_cases(root, engine)]
