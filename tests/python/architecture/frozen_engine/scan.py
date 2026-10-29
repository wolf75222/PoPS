"""Source-only scanner shared by the engine guards of spec 0.3 (Linear N2).

It reads the CF0 path list (``cf0_paths.toml``), enumerates the files of the frozen engine (the core
paths and the native component tree, FC6b) and yields, for CF2, every occurrence of the knowledge
that section 15.3 forbids there. Comments, docstrings and user-facing error text are not scanned
(CF2: "outside comments and user-facing error text"); identifiers, file paths and every other string
literal are, because a route table keyed on ``"rusanov"`` is exactly what CF2 refuses.

Matching is word-based. An identifier, a string or a path is split into words at ``_``, at
non-alphanumeric characters and at camel-case boundaries (``HLLCFlux`` -> ``hllc``, ``flux``;
``kRusanov`` -> ``k``, ``rusanov``); a term is a sequence of whole words, optionally followed by an
order or version number (``weno5``, ``ssprk2``). A term never matches a fragment of a longer word,
so no pattern is wider than the name it refuses.

The vocabulary refuses proper names (Rusanov, Poisson, Dirichlet), physical quantities and the
machinery that selects them (the role enumeration, route tables and their switches: FC3, FC4), not
category nouns (flux, limiter, reconstruction, Riemann problem, closure, moment). Words that also
carry a generic mathematical sense are refused only in their unambiguous spellings: ``density``
alone is a flux density or a stored density (spec D6, R8), so only the role token ``"density"``,
``VariableRole`` and the member ``Density`` are refused; short names such as ``mc`` or ``n0`` are
refused only as a whole identifier or string; a term inside a longer term is the longer term
(``divergence_preserving``). The few sites where a refused word carries a generic sense (Higham's
rounding bound gamma_n, the slope of a level-set fit) are listed, with their reason, in
``GENERIC_SENSES`` below.

The module imports neither ``pops`` nor ``_pops``. ``python -m
tests.python.architecture.frozen_engine.scan`` prints every occurrence in the format of
``cf2_exceptions.txt``.
"""
from __future__ import annotations

import ast
import bisect
import dataclasses
import json
import pathlib
import re
import subprocess
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
CF0_PATHS = HERE / "cf0_paths.toml"

PYTHON_SUFFIXES = {".py", ".pyi"}
CPP_SUFFIXES = {".hpp", ".h", ".cpp", ".cc", ".cxx", ".inc", ".cu"}
HASH_COMMENT_SUFFIXES = {".cmake", ".toml"}
JSON_SUFFIXES = {".json"}


# ---------------------------------------------------------------------------------------------
# CF0: the core paths and the native component tree.
# ---------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FrozenEngine:
    core_include: tuple[str, ...]
    core_exclude: tuple[str, ...]
    native_tree: str

    def is_core(self, rel: str) -> bool:
        return covered(rel, self.core_include) and not covered(rel, self.core_exclude)

    def is_native(self, rel: str) -> bool:
        return covered(rel, (self.native_tree,))

    def is_frozen(self, rel: str) -> bool:
        return self.is_core(rel) or self.is_native(rel)


def covered(rel: str, entries) -> bool:
    """True when @p rel is one of the file entries or lies under one of the directory entries."""
    for entry in entries:
        if entry.endswith("/"):
            if rel.startswith(entry):
                return True
        elif rel == entry:
            return True
    return False


def load_cf0(path: pathlib.Path = CF0_PATHS) -> FrozenEngine:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return FrozenEngine(
        core_include=tuple(data["core"]["include"]),
        core_exclude=tuple(data["core"].get("exclude", ())),
        native_tree=data["native_tree"]["path"],
    )


def tracked_files(root: pathlib.Path = REPO_ROOT) -> list[str]:
    """Every tracked file of the checkout, repository-relative and POSIX-spelled."""
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, stdout=subprocess.PIPE)
    return sorted(rel.decode() for rel in completed.stdout.split(b"\0") if rel)


def frozen_engine_files(engine: FrozenEngine | None = None,
                        root: pathlib.Path = REPO_ROOT) -> list[str]:
    engine = engine or load_cf0()
    return [rel for rel in tracked_files(root) if engine.is_frozen(rel)]


# Frozen-engine files that hold no code: documentation, the typing marker and the installed-header
# manifest (a list of the header paths, whose words the path scan already reads).
UNSCANNED_SUFFIXES = frozenset({".md", ".typed", ".manifest"})


def scanned_kind(rel: str) -> str | None:
    """The lexer of a frozen-engine file, or None for a file that holds no code (docs, markers)."""
    path = pathlib.PurePosixPath(rel)
    if path.name == "CMakeLists.txt" or path.name.endswith(".cmake.in"):
        return "hash"
    suffix = path.suffix
    if suffix in PYTHON_SUFFIXES:
        return "python"
    if suffix in CPP_SUFFIXES:
        return "cpp"
    if suffix in HASH_COMMENT_SUFFIXES:
        return "hash"
    if suffix in JSON_SUFFIXES:
        return "json"
    return None


# ---------------------------------------------------------------------------------------------
# Lexers: each yields (line, text, kind) with kind "ident", "string" or (Python only) "link".
# ---------------------------------------------------------------------------------------------
class _Lines:
    """Line numbers of character offsets, in O(log n) per query."""

    def __init__(self, text: str):
        self._starts = [0] + [match.end() for match in re.finditer("\n", text)]

    def __call__(self, offset: int) -> int:
        return bisect.bisect_right(self._starts, offset)


# A call whose name ends like an error type, or starts like a failure helper, carries error text.
_PY_ERROR_CALL = re.compile(
    r"(Error|Exception|Warning|Failure|Refusal|Violation)$"
    r"|^_?(warn|fail|error|raise|reject|refuse|abort)(_|$)", re.IGNORECASE)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _python_excluded_strings(tree: ast.AST) -> set[int]:
    """ids of the string constants that are docstrings, bare string statements or error text."""
    excluded: set[int] = set()

    def exclude_all(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                excluded.add(id(sub))

    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            excluded.add(id(node.value))  # a docstring or a bare string statement is a comment
        elif isinstance(node, ast.Raise):
            # the message of a raise is error text; the exception class is code and stays scanned
            for part in (node.exc, node.cause):
                if part is not None:
                    exclude_all(part)
        elif isinstance(node, ast.Assert) and node.msg is not None:
            exclude_all(node.msg)
        elif isinstance(node, ast.Call) and _PY_ERROR_CALL.search(_call_name(node.func) or "-"):
            for arg in list(node.args) + [keyword.value for keyword in node.keywords]:
                exclude_all(arg)
    return excluded


def _identifier_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_axis_link(node: ast.AST) -> bool:
    """``<name> + "_" + <axis>``: a component name spelled from an axis name (gap C19)."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return False
    left = node.left
    return ("axis" in split_words(_identifier_of(node.right))
            and isinstance(left, ast.BinOp) and isinstance(left.op, ast.Add)
            and isinstance(left.right, ast.Constant) and left.right.value == "_")


def python_tokens(text: str):
    tree = ast.parse(text)
    excluded = _python_excluded_strings(tree)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if _is_axis_link(node):
            yield line, AXIS_LINK, "link"
        if isinstance(node, ast.Name):
            yield line, node.id, "ident"
        elif isinstance(node, ast.Attribute):
            yield node.end_lineno, node.attr, "ident"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield line, node.name, "ident"
        elif isinstance(node, ast.arg):
            yield line, node.arg, "ident"
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.value.lineno, node.arg, "ident"
        elif isinstance(node, ast.alias):
            yield line, node.name, "ident"
            if node.asname:
                yield line, node.asname, "ident"
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield line, node.module, "ident"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in excluded:
            # a multi-line string reports each of its lines at its own line number
            for offset, piece in enumerate(node.value.split("\n")):
                if piece.strip():
                    yield line + offset, piece, "string"


_CPP_TOKEN = re.compile(
    r'(?P<comment>//[^\n]*|/\*.*?\*/)'
    r'|(?P<raw>R"(?P<delim>[^()\\\s]{0,16})\((?P<rawbody>.*?)\)(?P=delim)")'
    r'|(?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*")'
    r"|(?P<char>(?:u8|u|U|L)?'(?:\\.|[^'\\\n])+')"
    r'|(?P<number>\b[0-9][0-9A-Za-z_.\']*)'
    r'|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)'
    r'|(?P<punct>[;{}()])',
    re.DOTALL)
# An identifier that makes the rest of its statement error text: throw, static_assert, an error
# type (std::runtime_error, ...) or a failure helper (POPS_REQUIRE, fail_..., require_...).
_CPP_ERROR_IDENT = re.compile(
    r"^(throw|static_assert)$|(error|exception|failure|refusal)$"
    r"|^(POPS_)?(REQUIRE|ASSERT|EXPECT|FAIL|ABORT)|^(fail|reject|refuse|raise)(_|$)",
    re.IGNORECASE)


def cpp_tokens(text: str):
    """Identifiers and string literals of C++ source, outside comments and error text.

    A string literal is error text when an identifier of the same statement before it is
    ``throw``, ``static_assert``, an error type or a failure helper (``_CPP_ERROR_IDENT``). The
    statement ends at a ``;``, ``{`` or ``}`` at the parenthesis depth of that identifier, or at
    the ``)`` that closes the parentheses it appeared in; braces and semicolons nested inside its
    parentheses (a requires-expression, a lambda) do not end it.
    """
    line_of = _Lines(text)
    depth = 0  # parenthesis depth
    error_depth = None  # the depth of the error identifier of the current statement, if any
    for match in _CPP_TOKEN.finditer(text):
        kind = match.lastgroup
        if kind in ("comment", "char", "number"):
            continue
        line = line_of(match.start())
        if kind == "punct":
            punct = match.group("punct")
            if punct == "(":
                depth += 1
            elif punct == ")":
                depth -= 1
                if error_depth is not None and depth < error_depth:
                    error_depth = None
            elif error_depth is not None and depth <= error_depth:
                error_depth = None
        elif kind == "ident":
            ident = match.group("ident")
            if error_depth is None and _CPP_ERROR_IDENT.search(ident):
                error_depth = depth
            yield line, ident, "ident"
        elif kind == "string" and error_depth is None:
            literal = match.group("string")
            yield line, literal[literal.index('"') + 1:-1], "string"
        elif kind == "raw" and error_depth is None:
            for offset, piece in enumerate(match.group("rawbody").split("\n")):
                if piece.strip():
                    yield line + offset, piece, "string"


_HASH_TOKEN = re.compile(r'(?P<comment>#[^\n]*)|(?P<string>"(?:\\.|[^"\\\n])*")'
                         r'|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)')


def hash_comment_tokens(text: str):
    """CMake and TOML: identifiers and strings outside ``#`` comments."""
    line_of = _Lines(text)
    for match in _HASH_TOKEN.finditer(text):
        kind = match.lastgroup
        if kind == "comment":
            continue
        if kind == "string":
            yield line_of(match.start()), match.group("string")[1:-1], "string"
        else:
            yield line_of(match.start()), match.group("ident"), "ident"


_JSON_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')


def json_tokens(text: str):
    json.loads(text)  # a malformed schema is a defect, not an empty scan
    line_of = _Lines(text)
    for match in _JSON_STRING.finditer(text):
        yield line_of(match.start()), match.group(0)[1:-1], "string"


LEXERS = {"python": python_tokens, "cpp": cpp_tokens, "hash": hash_comment_tokens,
          "json": json_tokens}


# ---------------------------------------------------------------------------------------------
# Words.
# ---------------------------------------------------------------------------------------------
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def split_words(text: str) -> list[str]:
    """Lower-cased words of an identifier, a string or a path; digits stay glued to the word before.

    ``HLLCFlux`` -> [hllc, flux]; ``kRusanov`` -> [k, rusanov]; ``WENO5`` -> [weno5];
    ``ssprk2_step`` -> [ssprk2, step]; ``VariableRole::Density`` -> [variable, role, density].
    """
    words: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        if not chunk:
            continue
        merged: list[str] = []
        for part in _WORD.findall(chunk):
            if part.isdigit() and merged:
                merged[-1] += part
            else:
                merged.append(part)
        words.extend(part.lower() for part in merged)
    return words


# ---------------------------------------------------------------------------------------------
# CF2 vocabulary (spec 0.3, section 15.3, FC2-FC4 and CF2).
#
# Each row: (category, source rule, terms). A term is one word or a phrase of words written with
# "-" ("crank-nicolson" matches the words crank, nicolson in a row); every word may carry a trailing
# order or version number. A term marked "ident:" matches identifiers and paths only, never strings.
# ---------------------------------------------------------------------------------------------
CF2_VOCABULARY = (
    ("role", "CF2 role vocabulary used as a selector; FC3",
     "ident:variable-role ident:canonical-roles ident:role-of ident:roles-for"),
    ("role", "CF2 role machinery behind the selector (VariableRole aliases VariableSemantic); "
     "FC3, C3",
     "ident:variable-semantic ident:variable-roles ident:component-role ident:role-key "
     "ident:parse-role ident:role-token ident:role-canonical ident:role-from-name ident:role-name "
     "ident:roles-csv ident:energy-role ident:role-index ident:require-density ident:from-semantic "
     "ident:c-rho"),
    ("role", "15.3 physical roles; FC3",
     "momentum velocity pressure temperature enthalpy kinetic-energy internal-energy total-energy"),
    ("model", "15.3 named models; CF2 model and family names",
     "euler navier stokes poisson vlasov maxwell hall baer nunziato mhd boltzmann fokker planck "
     "cahn hilliard burgers schrodinger pitaevskii"),
    ("model", "15.3 named models: the transport, force and species routes of b6",
     "isothermal compressible exb ex-b electrostatic drift-diffusion maxwellian potential-force "
     "background-density multispecies ionization thermal species"),
    ("model", "15.3 physical quantities and forces of named models",
     "magnetic electric lorentz gravity gravitational charge-density electron ion sound-speed "
     "speed-of-sound acoustic mach"),
    ("closure", "15.3 equations of state and closures",
     "equation-of-state eos ideal-gas adiabatic hyqmom qmom fan-li levermore maxent gaussian "
     "hermite bgk"),
    ("method", "15.3 named numerical fluxes and wave-speed estimates; CF2 named methods",
     "rusanov hll hllc hlle hlld roe osher godunov lax-friedrichs lax-wendroff upwind harten "
     "einfeldt davis"),
    ("method", "15.3 named reconstructions or limiters; CF2 named limiters",
     "muscl weno eno minmod superbee van-leer vanleer van-albada vanalbada koren monotonized "
     "zhang"),
    ("method", "15.3 named time schemes; CF2 named schemes",
     "ssprk heun strang patankar bdf crank-nicolson imex dirk runge-kutta adams-bashforth "
     "adams-moulton rosenbrock ars imexrk lie"),
    ("method", "15.3 named discretizations; CF2 named methods",
     "scharfetter gummel"),
    ("method", "15.3 named field operators; FC2", "laplacian curl gradient divergence"),
    ("tagging", "15.3 named field operators in the refinement predicates of the tag-graph VM (C36)",
     "gradient-above gradient-below discrete-gradient-stencil"),
    ("boundary", "15.3 named boundary families",
     "dirichlet neumann robin inflow outflow reflective slip foextrap no-flux zero-flux "
     "constant-extrapolation"),
    ("interpolation", "CF2 named coarse-fine, temporal or regrid interpolation (C43)",
     "conservative-linear conservative-injection conservative-coarse-fine conservative-polynomial "
     "linear-time-interpolation divergence-preserving node-multilinear linear-prolongation "
     "coarse-fine-ghost-interpolation constant-injection linear-temporal-interpolation"),
    ("approximation", "CF2 a named approximation of a history between two instants (T7)",
     "linear-interpolation dense-output"),
    ("controller", "15.3 named controllers; controllers are library functions (section 15, 5.11)",
     "cfl courant"),
    ("route", "FC4 a route table or switch keyed on named methods or models (C21)",
     "ident:select-riemann ident:riemann-route ident:limiter-route ident:recon-route "
     "ident:time-route ident:transport-route ident:source-route-id ident:parse-source-route "
     "ident:source-route-tokens ident:elliptic-route ident:field-solver-route"),
    ("constant", "15.3 physical constants; CF2 a physical constant in an ABI",
     "gamma physical-default q-over-m four-pi-g vacuum-floor"),
)

# Exact string literals, compared whole and case-sensitively.
# A primitive selected by its spelling (CF2, the "p" test) and the role tokens of b6
# (python/pops/physics/aux.py) whose word also has a generic sense.
EXACT_LITERALS = (
    ("primitive", "CF2 a primitive selected by its spelling",
     ("p", "rho", "rho_u", "rho_v", "rho_w", "rho_E", "rhou", "rhov", "rhow", "E", "vx", "vy",
      "vz")),
    ("role", "CF2 role vocabulary used as a selector; FC3", ("density", "energy")),
    ("boundary", "15.3 named boundary families by their token", ("wall", "outlet", "inlet")),
    ("method", "15.3 the monotonized-central limiter by its route token", ("mc",)),
)
# Exact names, compared whole and case-sensitively with an identifier or a string literal: names
# too short or too generic as words (MC, n0) that are knowledge only spelled exactly so. Each is
# reported as `name`.
EXACT_NAMES = (
    ("role", "CF2 the members of the role enumeration; FC3",
     ("Density", "Energy", "kDensity", "kEnergy")),
    ("constant", "15.3 physical constants in an ABI (C20)", ("qom", "cs2", "n0", "rho0", "B0")),
    ("method", "15.3 the monotonized-central limiter by its short name", ("MC", "kMc")),
)
# A component name built from a variable name and an axis name, the <name>_<axis> links of b6 (C19).
AXIS_LINK = "name-axis-link"
# Case and profile identifiers (15.3), as whole identifiers or strings: M01-M28, T01-T12. L1, L2 are
# norms in the core (pops.linalg) and M00, M10, ... raw moments, so neither is refused.
CASE_IDENTIFIER = re.compile(r"^(M(0[1-9]|1[0-9]|2[0-8])|T(0[1-9]|1[0-2]))$")
# A term preceded by one of these words is another word: round_robin is a load-balancing order,
# not the Robin boundary family, and the conjugate gradient is a Krylov driver (FC6b), not the
# gradient operator.
NOT_AFTER = {"robin": frozenset({"round"}), "gradient": frozenset({"conjugate"})}


@dataclasses.dataclass(frozen=True)
class Term:
    label: str
    words: tuple[str, ...]
    category: str
    source: str
    ident_only: bool


def _vocabulary() -> tuple[Term, ...]:
    terms = []
    for category, source, spelled in CF2_VOCABULARY:
        for raw in spelled.split():
            ident_only = raw.startswith("ident:")
            label = raw.removeprefix("ident:")
            terms.append(Term(label, tuple(label.split("-")), category, source, ident_only))
    return tuple(terms)


VOCABULARY = _vocabulary()
_BY_FIRST_WORD: dict[str, list[Term]] = {}
for _term in VOCABULARY:
    _BY_FIRST_WORD.setdefault(_term.words[0], []).append(_term)
_EXACT = {literal: (category, source)
          for category, source, literals in EXACT_LITERALS for literal in literals}
_EXACT_NAME = {name: (category, source)
               for category, source, names in EXACT_NAMES for name in names}
TERM_INFO = {term.label: (term.category, term.source) for term in VOCABULARY}
TERM_INFO.update({'"%s"' % literal: info for literal, info in _EXACT.items()})
TERM_INFO.update({"`%s`" % name: info for name, info in _EXACT_NAME.items()})
TERM_INFO["case-identifier"] = ("case", "15.3 case or profile identifiers")
TERM_INFO[AXIS_LINK] = ("name", "FC4 names are presentation: a <name>_<axis> link (C19)")


def _strip_order(word: str) -> str:
    return word.rstrip("0123456789") or word


def terms_in(text: str, kind: str) -> set[str]:
    """Labels of the CF2 terms that @p text holds; @p kind is "ident", "string", "link" or "path"."""
    found: set[str] = set()
    if kind == "link":
        return {AXIS_LINK}
    if kind == "string" and text in _EXACT:
        found.add('"%s"' % text)
    if kind in ("ident", "string") and text in _EXACT_NAME:
        found.add("`%s`" % text)
    if kind != "path" and CASE_IDENTIFIER.match(text):
        found.add("case-identifier")
    words = [_strip_order(word) for word in split_words(text)]
    matches = []  # (first word, past-the-end word, label)
    for index, word in enumerate(words):
        for term in _BY_FIRST_WORD.get(word, ()):
            if term.ident_only and kind == "string":
                continue
            if index and words[index - 1] in NOT_AFTER.get(term.label, ()):
                continue
            if tuple(words[index:index + len(term.words)]) == term.words:
                matches.append((index, index + len(term.words), term.label))
    # A term inside a longer term is that longer term: divergence_preserving is the C43
    # interpolation, not the divergence operator.
    for start, end, label in matches:
        if not any(other_start <= start and end <= other_end
                   and other_end - other_start > end - start
                   for other_start, other_end, _ in matches):
            found.add(label)
    return found


# ---------------------------------------------------------------------------------------------
# Generic senses: sites where a refused spelling carries a generic mathematical meaning, so the
# site is not knowledge of section 15.3 and is not an exception either. Each entry names the file,
# the term, a regular expression that the whole source line must match, and the reason. The CF2 test
# fails on an entry that no longer matches any site, so no entry outlives its site. The pattern ".*"
# covers a whole file and is kept to files whose subject is that generic sense.
# ---------------------------------------------------------------------------------------------
GENERIC_SENSES = (
    ("include/pops/numerics/elliptic/mg/residual_roundoff.hpp", "gamma", r".*",
     "Higham's rounding bound gamma_n of floating-point error analysis (section 11 plumbing), "
     "not the ratio of specific heats; the whole file is that bound"),
    ("include/pops/numerics/elliptic/mg/geometric_mg.hpp", "gamma",
     r".*(gamma-bound@1|roundoff::stencil_gamma).*", "the same rounding bound gamma_n"),
    ("include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp", "gamma",
     r".*(roundoff::stencil_gamma|roundoff::gamma_for_operations|Scaled::from\(gamma\)).*",
     "the same rounding bound gamma_n"),
    ("python/pops/moments/closures/entropy_domain.py", "gamma",
     r"\s*(gamma = k\*u/\(1-k\*u\)|error = gamma\*sum\(.*)",
     "the rounding bound gamma_n = nu/(1 - nu) of a floating-point sum"),
    ("python/pops/solvers/elliptic/_cycle.py", "gamma", r".*binary64-five-point-gamma-bound.*",
     "the name of the rounding-bound policy"),
    ("python/pops/solvers/elliptic/_prepared_field_providers.py", "gamma",
     r".*binary64-five-point-gamma-bound.*", "the name of the rounding-bound policy"),
    ("include/pops/amr/tagging/berger_rigoutsos.hpp", "laplacian", r".*",
     "the second difference of the 1-D signature in Berger-Rigoutsos clustering (core machinery, "
     "section 15.2 item 5, DM13), not a field operator"),
    ("pyproject.toml", '"E"', r'\s*select = \["E", .*', "the pycodestyle error class of ruff"),
    ("python/pops/codegen/program_emit_principal.py", '"p"', r'\s*letter = "p" if primitive else "u"',
     "the local name of the emitted array argument, not a selected primitive"),
    ("include/pops/numerics/spatial/embedded_boundary/cut_geometry.hpp", "gradient", r".*",
     "the slope vector of the linear level-set fit that measures cut-cell fractions (geometry of "
     "the support, section 4.3), not a field operator; the whole file is that geometry"),
    ("src/runtime/amr/amr_system.cpp", "divergence",
     r'\s*rollback(_active_field_candidate_collectively)?\((lane, )?'
     r'"[a-z ]*divergence rollback"\);',
     "a disagreement between MPI ranks before a collective rollback, not the divergence operator"),
    ("python/pops/time/_program/inspection.py", "pressure",
     r'\s*"many live scratch buffers: GPU memory pressure / allocator churn; consider reuse"\),',
     "memory pressure on a GPU, not the physical pressure"),
)


def generic_sense(rel: str, term: str, source_line: str):
    """The GENERIC_SENSES entry that covers this site, or None."""
    for entry in GENERIC_SENSES:
        path, entry_term, pattern, _ = entry
        if path == rel and entry_term == term and re.fullmatch(pattern, source_line):
            return entry
    return None


# ---------------------------------------------------------------------------------------------
# Occurrences.
# ---------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, order=True)
class Occurrence:
    path: str
    line: int  # 0 when the term is in the file path itself
    term: str
    excerpt: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.term

    def render(self) -> str:
        return "%s:%d %s | %s" % (self.path, self.line, self.term, self.excerpt)


def _excerpt(lines: list[str], line: int) -> str:
    if line == 0:
        return "(path)"
    if 1 <= line <= len(lines):
        return " ".join(lines[line - 1].split())[:110]
    return ""


def cf2_sites_in_text(rel: str, text: str, kind: str):
    """(Occurrence, generic-sense entry or None) for every CF2 site of one file."""
    lines = text.splitlines()
    found: dict[tuple[int, str], Occurrence] = {}
    for term in terms_in(rel, "path"):
        found[(0, term)] = Occurrence(rel, 0, term, "(path)")
    for line, token, token_kind in LEXERS[kind](text):
        for term in terms_in(token, token_kind):
            found.setdefault((line, term), Occurrence(rel, line, term, _excerpt(lines, line)))
    for occurrence in sorted(found.values()):
        source_line = lines[occurrence.line - 1] if occurrence.line else ""
        yield occurrence, generic_sense(rel, occurrence.term, source_line)


def cf2_sites(rel: str, root: pathlib.Path = REPO_ROOT):
    kind = scanned_kind(rel)
    if kind is None:
        found = [(Occurrence(rel, 0, term, "(path)"), None) for term in terms_in(rel, "path")]
        return sorted(found)
    text = (root / rel).read_text(encoding="utf-8")
    return list(cf2_sites_in_text(rel, text, kind))


def scan_cf2(root: pathlib.Path = REPO_ROOT, engine: FrozenEngine | None = None):
    """(occurrences that are knowledge, generic-sense entries that matched at least once)."""
    occurrences: list[Occurrence] = []
    used = set()
    for rel in frozen_engine_files(engine, root):
        for occurrence, sense in cf2_sites(rel, root):
            if sense is None:
                occurrences.append(occurrence)
            else:
                used.add(sense)
    return occurrences, used


if __name__ == "__main__":
    for found in scan_cf2()[0]:
        print(found.render())
