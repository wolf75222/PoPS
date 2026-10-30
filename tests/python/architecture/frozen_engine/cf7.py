"""CF7 (spec 0.3, section 16.2; B4, B6): no constant limit on analysis work in the frozen engine.

An analysis of the IR (coverage, delivery, footprint, dependency, linearity) is symbolic and scales
with the IR; it never truncates and never depends on a count of unrolled iterations or enumerated
alternatives (B4). The interpreters of b6 (the analytic-expression VM and the coupled-source
bytecode, C44; the tag-graph VM, C36) cap the programs they accept in the same way.

The scan reads the core paths and the native component tree (FC6b, ``cf0_paths.toml``) and reports
every named integer limit on analysis work: a constant, a default argument, a keyword argument, a
member initializer or a ``#define`` whose name holds a limit word (``max``, ``maximum``, ``limit``,
``cap``, ``ceiling``, ``budget``) and a noun of analysis work (alternatives, iterations unrolled,
entries visited, nodes, depth, stack, registers, terms, program length, instructions), bound to an
integer literal of at least 2 that is not the range of an integer type. ``METHOD_BUDGETS`` names the
limits whose words match but which are method budgets of section 12.1 (the iteration limit of a
Newton, Krylov, multigrid or eigenvalue solver), with their reason; ``RUNTIME_BACKSTOPS`` names the
limits of a loop of the run that are not analysis caps but that no author writes either (a B1 and
B7 gap, with the issue that closes it); ``ANCHORED_SITES`` names the b6 caps that the vocabulary
does not catch. The exception list of b6 (``cf7_exceptions.txt``) only shrinks (MG6); at the exit
of Phase 1 it holds only the caps of the two interpreters of C44 (section 18.3), and at the exit of
Phase 2 it is empty.

A site is one limit on one line. Its excerpt is its spelling, the name and the value of the limit
(``kAnalyticMaxNodes = 64``; the source line of an anchored site). The ratchet pairs sites with
list entries by it, so an edit elsewhere on the line keeps the site, while a cap whose value
changes is a new site: a listed cap leaves the list only by being removed.
"""
from __future__ import annotations

import ast
import collections
import dataclasses
import pathlib
import re

from . import pysource, ratchet, scan

EXCEPTIONS = scan.HERE / "cf7_exceptions.txt"

LIMIT_WORDS = frozenset({"max", "maximum", "limit", "cap", "ceiling", "budget"})
# CF7: "iterations unrolled, alternatives enumerated, map entries visited", and the program sizes
# that the interpreters of C36 and C44 accept. Solver words (cycles, steps, evaluations, backtracks,
# attempts) are absent: those limits are method budgets (section 12.1, B1).
ANALYSIS_NOUNS = frozenset({
    "alternative", "alternatives", "iteration", "iterations", "iter", "iters", "unroll",
    "unrolled", "unrolling", "entry", "entries", "visit", "visits", "visited", "node", "nodes",
    "depth", "stack", "reg", "regs", "register", "registers", "term", "terms", "prog", "program",
    "programs", "instruction", "instructions", "expansion", "expansions", "frontier", "frontiers",
    "choices", "combinations", "candidates",
})


def is_limit_name(name: str) -> bool:
    words = {scan._strip_order(word) for word in scan.split_words(name)}
    return bool(words & LIMIT_WORDS) and bool(words & ANALYSIS_NOUNS)


def is_type_range(value: int) -> bool:
    """2**k or 2**k - 1 for k >= 31: the range of an integer type (INT32_MAX, 2**53, UINT64_MAX)."""
    return value >= (1 << 31) - 1 and ((value & (value + 1)) == 0 or (value & (value - 1)) == 0)


def counts_as_limit(value) -> bool:
    return type(value) is int and value >= 2 and not is_type_range(value)


# ---------------------------------------------------------------------------------------------
# Python.
# ---------------------------------------------------------------------------------------------
def _int_value(node: ast.AST):
    """The value of an integer literal expression (``4096``, ``64 * 1024``, ``1 << 20``),
    or None."""
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is int else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _int_value(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.LShift,
                                                           ast.Pow)):
        left, right = _int_value(node.left), _int_value(node.right)
        if left is None or right is None or (isinstance(node.op, (ast.LShift, ast.Pow))
                                             and not 0 <= right <= 128):
            return None
        return {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right,
                ast.LShift: left << right if isinstance(node.op, ast.LShift) else 0,
                ast.Pow: left ** right if isinstance(node.op, ast.Pow) else 0}[type(node.op)]
    return None


def python_limits(text: str):
    """(line, name, value) of every named integer limit of a Python file."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        bound = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    bound.append((target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None \
                and isinstance(node.target, (ast.Name, ast.Attribute)):
            bound.append((node.target, node.value))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            bound.extend(zip(positional[len(positional) - len(arguments.defaults):],
                             arguments.defaults, strict=True))
            bound.extend((arg, default) for arg, default in
                         zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
                         if default is not None)
        elif isinstance(node, ast.keyword) and node.arg:
            bound.append((node, node.value))
        elif isinstance(node, ast.Dict):
            bound.extend((key, value) for key, value in zip(node.keys, node.values, strict=True)
                         if isinstance(key, ast.Constant) and isinstance(key.value, str))
        for target, value in bound:
            name = (target.id if isinstance(target, ast.Name) else target.attr
                    if isinstance(target, ast.Attribute) else target.value
                    if isinstance(target, ast.Constant) else target.arg)
            number = _int_value(value)
            if is_limit_name(name) and counts_as_limit(number):
                yield value.lineno, name, number


_JSON_ENTRY = re.compile(r'"((?:\\.|[^"\\\n])*)"\s*:\s*(-?[0-9]+)(?![0-9.eE])')


def json_limits(text: str):
    """(line, key, value) of every integer-valued key of a JSON file that names a limit."""
    lines = scan._Lines(text)
    for match in _JSON_ENTRY.finditer(text):
        name, value = match.group(1), int(match.group(2))
        if is_limit_name(name) and counts_as_limit(value):
            yield lines(match.start()), name, value


# ---------------------------------------------------------------------------------------------
# C++.
# ---------------------------------------------------------------------------------------------
_CPP_COMMENT_OR_TEXT = re.compile(
    r'//[^\n]*|/\*.*?\*/|R"(?P<d>[^()\\\s]{0,16})\(.*?\)(?P=d)"|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])+'", re.DOTALL)
_EXPRESSION = r"[0-9(][0-9A-Fa-fxX'uUlLzZ()\s*+<-]*"
_CPP_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+(" + _EXPRESSION + ")$",
                         re.MULTILINE)
_CPP_INIT = re.compile(r"\b([A-Za-z_]\w*)\s*(?:(?<![=!<>+\-*/%&|^])=(?!=)|\{)\s*(" + _EXPRESSION
                       + r")\s*(?=[;,)}])")
_CPP_INTEGER = re.compile(r"(0[xX][0-9A-Fa-f']+|0[bB][01']+|[0-9][0-9']*)[uUlLzZ]*")


def _cpp_value(expression: str):
    """The value of a C++ integer literal expression, or None (a float, a name, a call)."""
    expression = expression.strip()
    while expression.endswith(")") and expression.count(")") > expression.count("("):
        expression = expression[:-1].rstrip()  # the ")" that closes a parameter list
    python = _CPP_INTEGER.sub(lambda m: str(int(m.group(1).replace("'", ""), 0)
                                            if not re.fullmatch(r"0[0-7']+", m.group(1))
                                            else int(m.group(1).replace("'", ""), 8)),
                              expression.strip())
    if not re.fullmatch(r"[0-9()\s*+<-]+", python) or not python.strip():
        return None
    try:
        return _int_value(ast.parse(python, mode="eval").body)
    except (SyntaxError, ValueError):
        return None


_CPP_ENUM = re.compile(r"\benum\b(\s+(class|struct))?[^;{}()]*\{")


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def _blank_enum_bodies(text: str) -> str:
    """An enumerator is a name for a value (a status code), never a limit: blank enum bodies."""
    position = 0
    while (match := _CPP_ENUM.search(text, position)) is not None:
        depth, index = 1, match.end()
        while index < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        text = _blank(text, match.end(), index - 1)
        position = index
    return text


def cpp_limits(text: str):
    """(line, name, value) of every named integer limit of a C++ file, outside comments, string
    literals and enum bodies."""
    blanked = _CPP_COMMENT_OR_TEXT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    blanked = _blank_enum_bodies(blanked)
    lines = scan._Lines(blanked)
    for pattern in (_CPP_DEFINE, _CPP_INIT):
        for match in pattern.finditer(blanked):
            name, value = match.group(1), _cpp_value(match.group(2))
            if is_limit_name(name) and counts_as_limit(value):
                yield lines(match.start(1)), name, value


# ---------------------------------------------------------------------------------------------
# What is not a cap on analysis work, and the b6 caps that are not named constants.
# ---------------------------------------------------------------------------------------------
# (path, names, reason). A method budget of section 12.1 is the iteration limit of a numerical
# method, written by the method author and reported when exceeded (B1); it bounds no analysis of the
# IR. The test fails on an entry that no longer names a limit of its file.
METHOD_BUDGETS = (
    ("include/pops/numerics/elliptic/interface/field_nonlinear.hpp",
     ("max_iterations", "linear_max_iterations"),
     "the Newton and inner linear iteration limits of a nonlinear field solve"),
    ("include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp", ("maximum_iterations",),
     "the Krylov iteration limit of the polar tensor solve"),
    ("include/pops/numerics/linalg/dense_eig.hpp", ("max_iter", "max_iter_per_eig"),
     "the iteration limits of the native dense eigenvalue algorithms"),
    ("include/pops/numerics/nonlinear/prepared_local_nonlinear.hpp", ("max_iterations",),
     "the iteration limit of a local Newton solve"),
    ("include/pops/runtime/amr/tensor_composite_fac.hpp", ("maximum_iterations",),
     "the FAC iteration limit of the AMR tensor solve"),
    ("include/pops/runtime/numerical_defaults.hpp",
     ("kNewtonDefaultMaxIters", "kPolarTensorKrylovDefaultMaxIters", "kSchurKrylovPolarMaxIters",
      "kCartesianCGDefaultMaxIterations", "kFACDefaultMaxIters"),
     "the default iteration limits of the native Newton, Krylov, CG and FAC solvers"),
    ("python/pops/fields/providers.py", ("max_iterations",),
     "the iteration limit of an external field-solver provider"),
    ("python/pops/moments/model_builder.py", ("eig_max_iter",),
     "the eigenvalue iteration limit of the wave-speed bound of a moment model"),
    ("python/pops/runtime/defaults.py",
     ("max_iters", "polar_tensor_max_iters", "schur_polar_max_iters", "max_iterations"),
     "the Python mirror of numerical_defaults.hpp"),
    ("python/pops/solvers/elliptic/_descriptor.py", ("_CARTESIAN_CG_DEFAULT_MAX_ITERATIONS",),
     "the default CG iteration limit"),
    ("python/pops/solvers/elliptic/_prepared_field_providers.py", ("max_iters",),
     "the FAC iteration limit of a prepared field provider"),
    ("python/pops/solvers/nonlinear/__init__.py", ("max_iterations", "linear_max_iterations"),
     "the Newton and inner linear iteration limits of the nonlinear solver descriptors"),
    ("python/pops/solvers/providers.py", ("_DEFAULT_MAX_ITER",),
     "the default iteration limit of a hierarchy solver provider"),
)

# (path, names, reason). A limit of a loop of the run that no author writes: not a cap on analysis
# work, so CF7 leaves it out, but not a method budget either, since B1 and B7 want a budget written
# in the Program and reported in the manifest. Each entry names the gap and the issue that closes
# it. The test fails on an entry that no longer names a limit of its file.
RUNTIME_BACKSTOPS = (
    ("python/pops/codegen/solvers/dsl.py", ("_SOLVER_MAX_ITERS",),
     "the hard cap that solver_cpp.py emits into the loop of a generated custom solver whose IR "
     "omits one (_emit_while): it bounds a loop of the run, not an analysis, but the Program does "
     "not write it and the manifest does not report it, a B1 and B7 gap that G10 closes when "
     "pops.lib.solve references with iterate(..., max_steps) replace the custom solvers"),
)

# (path, term, regular expression of the line, reason). A b6 cap that the vocabulary does not catch,
# because no named constant carries it or because its name holds no noun of analysis work; the scan
# reports it while its line is there. Pinned at b6: entries are removed, never added.
ANCHORED_SITES = (
    ("python/pops/codegen/program_stencil_budget.py", "runtime-bound-repetition",
     r'\s*if bound\.op == "int64_constant":',
     "the retained-expression analysis multiplies by the count of a fold or iterate whose bound "
     "is a constant and counts a runtime-valued bound as one repetition, \"the default finite "
     "ceiling\" of lines 26-28, which an author lifts with flux_expression_budget (CF7, IR19)"),
    ("include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp",
     "kMaxAnalyticBoundaryComponents",
     r"\s*inline constexpr int kMaxAnalyticBoundaryComponents = \d+;",
     "the number of analytic-VM programs that a boundary state holds, one per component "
     "(programs[kMaxAnalyticBoundaryComponents]), refused beyond it: a program-size cap of the "
     "analytic-expression VM of C44 on its boundary route (found by the lot 2 audit)"),
)


@dataclasses.dataclass(frozen=True, order=True)
class Site:
    path: str
    line: int
    term: str
    excerpt: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.term


def _method_budget(rel: str, name: str):
    """The METHOD_BUDGETS or RUNTIME_BACKSTOPS entry that exempts @p name in @p rel, or None."""
    for entry in METHOD_BUDGETS + RUNTIME_BACKSTOPS:
        path, names, _ = entry
        if path == rel and name in names:
            return entry
    return None


def scan_engine(root: pathlib.Path = scan.REPO_ROOT, engine: scan.FrozenEngine | None = None):
    """(CF7 sites of the frozen engine, METHOD_BUDGETS and RUNTIME_BACKSTOPS entries used,
    ANCHORED_SITES entries used). A site is one per (path, line, term), its excerpt the spellings
    of the limit on that line."""
    spellings = collections.defaultdict(set)  # (path, line, term) -> spellings
    budgets_used, anchors_used = set(), set()
    anchored = {}
    for entry in ANCHORED_SITES:
        anchored.setdefault(entry[0], []).append(entry)
    readers = {"python": python_limits, "cpp": cpp_limits, "json": json_limits}
    for rel in scan.frozen_engine_files(engine, root):
        kind = scan.scanned_kind(rel)
        if kind not in readers:
            continue
        text = (root / rel).read_text(encoding="utf-8")
        lines = text.splitlines()
        found = readers[kind](text)
        for line, name, value in found:
            budget = _method_budget(rel, name)
            if budget is not None:
                budgets_used.add(budget)
                continue
            spellings[(rel, line, name)].add("%s = %d" % (name, value))
        for entry in anchored.get(rel, ()):
            _, term, pattern, _ = entry
            for number, source in enumerate(lines, start=1):
                if re.fullmatch(pattern, source):
                    anchors_used.add(entry)
                    spellings[(rel, number, term)].add(" ".join(source.split()))
    sites = sorted(Site(rel, line, term, pysource.site_excerpt(spelled))
                   for (rel, line, term), spelled in spellings.items())
    return sites, budgets_used, anchors_used


_TAGGING_CAPS = frozenset({"maximum_stencil_terms", "maximum_instruction_count",
                           "POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1",
                           "POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1"})

# (issue, path entries or None, terms or None, reason): the first rule that matches both its paths
# and its terms names the issue (section 18.3 for the exit of Phase 1).
OWNER_RULES = (
    ("N7", ("python/pops/codegen/realization_delivery.py",
            "python/pops/codegen/program_stencil_budget.py"), None,
     "C33: typed delivery and retained-expression analyses that scale with the IR, Phase 1"),
    ("N12", None, _TAGGING_CAPS,
     "C36: the program caps of the tag-graph VM (schema, Python and C++ ABI), removed with it, "
     "Phase 1"),
    ("G04+G07", ("python/pops/analytic/", "python/pops/codegen/_analytic_expression_lowering.py",
                 "include/pops/runtime/analytic/"), None,
     "C44: the analytic-expression VM, removed once its boundary (G04) and initial-data (G07) "
     "routes are replaced, Phase 2"),
    ("G06", ("python/pops/physics/_coupled_abi.py", "include/pops/coupling/source/"), None,
     "C44: the coupled-source bytecode, Phase 2"),
    ("G04", ("include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp",),
     frozenset({"kMaxAnalyticBoundaryComponents"}),
     "C44: the analytic-expression VM on the boundary-state route, removed when G04 replaces that "
     "route, Phase 2"),
    ("N7", None, None, "B4: the analyses of the IR scale with it (IR8-IR21), Phase 1"),
)


def owner_of(path: str, term: str) -> str:
    for issue, paths, terms, _ in OWNER_RULES:
        if paths is not None and not scan.covered(path, paths):
            continue
        if terms is not None and term not in terms:
            continue
        return issue
    raise AssertionError("OWNER_RULES has no default")


def current_entries(root: pathlib.Path = scan.REPO_ROOT, engine=None):
    """(entries of every CF7 site, METHOD_BUDGETS entries used, ANCHORED_SITES entries used)."""
    sites, budgets_used, anchors_used = scan_engine(root, engine)
    entries = [ratchet.Entry(site.path, site.line, site.term, owner_of(site.path, site.term),
                             site.excerpt) for site in sites]
    return entries, budgets_used, anchors_used
