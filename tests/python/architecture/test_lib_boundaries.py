"""ADC-566/ADC-693, rewritten by Linear N2 (spec 0.3): pops.lib is the home of numerical methods.

Spec 0.3 inverts the rule this file used to enforce. The b6 version pinned ``HLL``, ``Rusanov``,
``MUSCL``, ``WENO5``, ``GeometricMG`` and ``FFT`` to homes in the core and refused them in
``pops.lib``. Under spec 0.3 a named numerical method (a flux, a reconstruction, a limiter, a scheme,
an interpolation) is a ``pops.lib`` word written with the public levels (section 2.6.9, A4, P11),
and the core must not know it (section 15.3, FC2, FC4); a solver is a reference in ``pops.lib.solve``
whose natives are qualified substitutions (section 15, SV19). So:

  * gate 2a -- ``pops.lib`` may hold the modules of section 2.6.9 (``math``, ``models``,
    ``numerics``, ``output``, ``solve``, ``time``) beside the b6 packages ``amr`` and ``initial``;
  * gate 2b -- ``pops.lib`` is still a leaf: no ``_pops``/codegen/runtime import at any scope (CF0b,
    whose full scan with its b6 exception list is the next lot of N2, makes this stricter);
  * gate 2c -- the central objects of the core (layouts, runtime parameters, programs, modules,
    field operator protocols) keep one home outside ``pops.lib``; method names are no longer among
    them;
  * gate 2e (the inverse rule) -- no numerical method has a home in the core: a class, function,
    type or module-level binding of the core paths named after one is a CF2 exception of b6
    (``frozen_engine/cf2_exceptions.txt``), and their number only goes down (MG6).

It EXTENDS, and does not re-implement, ``test_no_runtime_imports.py`` (module-scope runtime imports)
and ``test_import_graph.py`` (the lib -> package edges). Source-only AST scans plus one
installed-package proof, which fails when the promised ``pops`` installation is unavailable; an
acceptance gate must never turn that defect into a skip. ASCII only.
"""
import ast
import collections
import pathlib
import re

import pytest

from tests.python.architecture.frozen_engine import cf2, ratchet, scan


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "python" / "pops" / "lib"

# pops.lib may hold the b6 packages and the modules of section 2.6.9 of spec 0.3. A child outside this
# set is reviewed here first.
_B6_LIB_CHILD_DIRS = {"amr", "initial", "models", "time"}
_SPEC_LIB_MODULES = {"math", "models", "numerics", "output", "solve", "time"}
_ALLOWED_LIB_CHILD_DIRS = _B6_LIB_CHILD_DIRS | _SPEC_LIB_MODULES

# The HARD refusal list: lib is a leaf, so it must import none of these at ANY scope (module-scope OR
# lazily inside a function). This is the delta over test_no_runtime_imports.py (module-scope only).
_FORBIDDEN_IMPORT_ROOTS = ("_pops", "pops.codegen", "pops.runtime", "pops._pops")

# The descriptor/authoring packages lib MAY import (compose descriptors). Everything under pops.* that
# is NOT in this set and NOT forbidden is flagged: a new pops.<other> edge from lib must be reviewed.
_ALLOWED_POPS_IMPORT_ROOTS = (
    "pops",  # the public facade used by ready-to-use compositions
    "pops._ir",
    "pops.math",
    "pops.model",
    "pops.time",
    "pops.physics",
    "pops.moments",
    "pops.mesh",
    "pops.numerics",
    "pops.solvers",
    "pops.linalg",
    "pops.descriptors",
    "pops.params",
    "pops.lib",  # intra-lib composition
)

# Central objects of the core that have exactly ONE home outside lib; lib must neither DEFINE nor
# RE-EXPORT them. Named numerical methods (HLL, HLLC, Roe, Rusanov, MUSCL, WENO5) and solvers
# (GeometricMG, FFT) are no longer here: their home is pops.lib (gate 2e).
_CANONICAL_NAMES = {
    "FieldOperator", "FieldDiscretization", "AMR", "RuntimeParam", "Program", "Module",
}

# lib/__init__.py must stay thin; a fat __init__ that concentrates logic fails.
_LIB_INIT_LINE_CAP = 40

# Categories of the CF2 vocabulary (frozen_engine/scan.py) that name a numerical method: methods and
# field operators, interpolations, approximations of histories, controllers, refinement predicates
# and solve algorithms.
_METHOD_CATEGORIES = ("method", "interpolation", "approximation", "controller", "tagging", "solver")

# The number of core definitions named after a numerical method at b6e64c1d (gate 2e), counted with
# the vocabulary that the early audit of N2 completed on 29/10 and the solve algorithms that its
# independent review added on 30/10 (GeometricMG, FFT, CG, GMRES, Newton, ...). It may only go down
# (MG6); never raise it.
_B6_METHOD_HOMES_IN_CORE = 296


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _read(path):
    return path.read_text(encoding="utf-8")


def _parse(path):
    return ast.parse(_read(path), filename=str(path))


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _is_forbidden(modname):
    return any(modname == root or modname.startswith(root + ".") for root in _FORBIDDEN_IMPORT_ROOTS)


def _is_allowed_pops(modname):
    return any(modname == root or modname.startswith(root + ".") for root in _ALLOWED_POPS_IMPORT_ROOTS)


def _import_targets(tree):
    """Yield (lineno, target, is_lazy) for every import in @p tree (module-scope AND in-function)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, node.col_offset > 0
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # absolute imports only
                yield node.lineno, node.module, node.col_offset > 0


# ---------------------------------------------------------------------------------------------
# Gate 2a -- DIRECTORY fence: every lib child is a b6 package or a module of section 2.6.9.
# ---------------------------------------------------------------------------------------------
def test_lib_child_directories_are_the_allowed_set():
    children = {
        p.name for p in LIB.iterdir()
        if p.is_dir() and p.name != "__pycache__" and (p / "__init__.py").is_file()
    }
    extra = children - _ALLOWED_LIB_CHILD_DIRS
    missing = _B6_LIB_CHILD_DIRS - children
    assert not extra, (
        "pops.lib may hold the b6 packages %s and the modules of spec 0.3 section 2.6.9 %s; a new "
        "child package is reviewed here first: %s"
        % (sorted(_B6_LIB_CHILD_DIRS), sorted(_SPEC_LIB_MODULES), sorted(extra)))
    assert not missing, "pops.lib is missing an expected package: %s" % sorted(missing)


# ---------------------------------------------------------------------------------------------
# Gate 2b -- IMPORT fence: lib is a leaf, no runtime/codegen import even lazily.
# ---------------------------------------------------------------------------------------------
def test_lib_never_imports_runtime_or_codegen_even_lazily():
    forbidden_hits = []
    off_allowlist = []
    for path in _py_files(LIB):
        rel = _rel(path)
        for lineno, target, is_lazy in _import_targets(_parse(path)):
            if _is_forbidden(target):
                scope = "lazily" if is_lazy else "at module scope"
                forbidden_hits.append("%s:%d imports %s %s" % (rel, lineno, target, scope))
            elif target.startswith("pops.") or target == "pops":
                if not _is_allowed_pops(target):
                    off_allowlist.append("%s:%d imports %s (not a descriptor/authoring package)"
                                         % (rel, lineno, target))

    assert not forbidden_hits, (
        "pops.lib is a leaf: it must not import _pops/codegen/runtime at ANY scope (this gate adds "
        "the lazy in-function case beyond the module-scope test_no_runtime_imports.py):\n  "
        + "\n  ".join(forbidden_hits))
    assert not off_allowlist, (
        "pops.lib may compose only descriptor/authoring packages %s; a new pops.<other> edge must be "
        "reviewed:\n  " % (sorted(_ALLOWED_POPS_IMPORT_ROOTS),) + "\n  ".join(off_allowlist))


# ---------------------------------------------------------------------------------------------
# Gate 2c -- CONTENT fence: no central object of the core DEFINED or RE-EXPORTED under lib.
# ---------------------------------------------------------------------------------------------
def test_lib_defines_no_canonical_central_class():
    violations = []
    for path in _py_files(LIB):
        rel = _rel(path)
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef) and node.name in _CANONICAL_NAMES:
                violations.append("%s:%d defines canonical class %r" % (rel, node.lineno, node.name))
    assert not violations, (
        "central objects (%s) have exactly one home elsewhere; pops.lib must not define them:\n  "
        % ", ".join(sorted(_CANONICAL_NAMES)) + "\n  ".join(violations))


def test_lib_reexports_no_canonical_central_name():
    """A ``from pops.time import Program`` (or an __all__ surfacing it) under lib is a second path to
    a central object -- refused via a MODULE-SCOPE ImportFrom + __all__ scan.

    Only a module-scope import binds a name on the lib module's public surface (a re-export). A LAZY
    (in-function) import of a canonical name -- e.g. ``_helpers.py`` importing ``Program`` inside the
    program-macro decorator to BUILD a program -- is internal use, not a re-export, so it is left
    alone (col_offset > 0)."""
    violations = []
    for path in _py_files(LIB):
        rel = _rel(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.col_offset == 0:
                for alias in node.names:
                    imported = alias.asname or alias.name
                    if imported in _CANONICAL_NAMES:
                        violations.append(
                            "%s:%d re-imports canonical name %r from %s at module scope"
                            % (rel, node.lineno, imported, node.module or "."))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if (isinstance(elt, ast.Constant)
                                        and elt.value in _CANONICAL_NAMES):
                                    violations.append(
                                        "%s:%d __all__ surfaces canonical name %r"
                                        % (rel, node.lineno, elt.value))
    assert not violations, (
        "pops.lib must not re-export a central object (that is a second path):\n  "
        + "\n  ".join(violations))


def test_canonical_homes_are_outside_lib():
    """Belt-and-braces: the central objects DO live in their documented non-lib home, so the content
    fence is guarding a real single-home invariant, not an empty set."""
    homes = {
        "AMR": REPO_ROOT / "python/pops/layouts/__init__.py",
        "RuntimeParam": REPO_ROOT / "python/pops/params/runtime.py",
        "FieldOperator": REPO_ROOT / "python/pops/fields/operator.py",
        "FieldDiscretization": REPO_ROOT / "python/pops/fields/discretization.py",
        "Program": REPO_ROOT / "python/pops/time/_program/api.py",
    }
    missing = []
    for name, path in homes.items():
        if not path.exists():
            missing.append("%s: expected home %s does not exist" % (name, _rel(path)))
            continue
        defined = any(isinstance(node, ast.ClassDef) and node.name == name
                      for node in ast.walk(_parse(path)))
        if not defined:
            missing.append("%s: not defined in its expected home %s" % (name, _rel(path)))
    assert not missing, "canonical single-home invariant broken:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------------------------------------
# Gate 2d -- lib.__init__ stays thin; models lower runtime-free.
# ---------------------------------------------------------------------------------------------
def test_lib_init_stays_thin():
    init = LIB / "__init__.py"
    lines = len(_read(init).splitlines())
    assert lines <= _LIB_INIT_LINE_CAP, (
        "pops.lib.__init__ must stay thin (<= %d lines); a fat __init__ concentrating logic is "
        "refused, got %d" % (_LIB_INIT_LINE_CAP, lines))


def test_lib_models_lower_to_physics_without_runtime():
    """Installed-package gate: a provided model lowers to a pops.model/physics object whose manifest
    carries NO runtime/compiled fields (no .so path, no abi_key)."""
    from pops.lib.models import Gaussian, HyQMOM15
    from pops.frames import Cartesian2D

    for factory in (lambda: HyQMOM15.vlasov_poisson_magnetic(
                        order=4, frame=Cartesian2D()),
                    lambda: Gaussian.transport(frame=Cartesian2D())):
        model = factory()
        module = getattr(model, "module", model)
        # The lowered object is an authoring Module (typed operators), not a runtime handle.
        assert hasattr(module, "operator_registry") or hasattr(module, "manifest"), (
            "a provided model must lower to a pops.model/physics authoring object, got %r"
            % type(model))
        # No compiled/runtime leakage on the authoring object.
        for runtime_attr in ("so_path", "abi_key"):
            assert not getattr(model, runtime_attr, None), (
                "a provided model must lower runtime-free; %r leaked %s" % (model, runtime_attr))


# ---------------------------------------------------------------------------------------------
# Gate 2e -- the inverse rule: no numerical method has a home in the core.
# ---------------------------------------------------------------------------------------------
_METHOD_TERMS = frozenset(label for label, (category, _) in scan.TERM_INFO.items()
                          if category in _METHOD_CATEGORIES)
_CPP_DEFINITION = re.compile(r"\b(?:struct|class|using|enum\s+class|enum)\s+([A-Za-z_]\w*)")
_CPP_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def method_homes_in_core():
    """(path, line, term, name) of every core definition named after a numerical method: a Python
    class, function or module-level binding, a C++ struct, class, enum or type alias. In the native
    tree, a qualified substitution may be named after its generic algorithm (scan.CORE_ONLY, FC6);
    a name with a generic sense (scan.GENERIC_SENSES) is no method."""
    engine = scan.load_cf0()
    homes = []
    for rel in scan.frozen_engine_files(engine):
        kind = scan.scanned_kind(rel)
        if kind not in ("python", "cpp"):
            continue
        refused = _METHOD_TERMS - scan.CORE_ONLY if engine.is_native(rel) else _METHOD_TERMS
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        lines = text.splitlines()
        if kind == "python":
            tree = ast.parse(text)
            named = [(node.lineno, node.name) for node in ast.walk(tree)
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            named += [(node.lineno, target.id) for node in tree.body if isinstance(node, ast.Assign)
                      for target in node.targets if isinstance(target, ast.Name)]
        else:
            blanked = _CPP_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
            named = [(blanked.count("\n", 0, match.start()) + 1, match.group(1))
                     for match in _CPP_DEFINITION.finditer(blanked)]
        homes.extend((rel, line, term, name) for line, name in named
                     for term in scan.terms_in(name, "ident") & refused
                     if scan.generic_sense(rel, term, lines[line - 1]) is None)
    return homes


def test_no_numerical_method_has_a_home_in_the_core():
    homes = method_homes_in_core()
    spelled = collections.defaultdict(set)  # (path, term) -> the names that its entries spell
    for entry in ratchet.read_list(cf2.EXCEPTIONS)[1]:
        spelled[entry.key].update(entry.excerpt.split())
    unlisted = ["%s:%d %s (%s)" % home for home in homes
                if home[3] not in spelled.get((home[0], home[2]), ())]
    assert not unlisted, (
        "spec 0.3 (15.3, FC2, FC4): a numerical method is a pops.lib word, never a core definition; "
        "these core definitions are not b6 exceptions of CF2:\n  " + "\n  ".join(unlisted))
    assert len(homes) <= _B6_METHOD_HOMES_IN_CORE, (
        "MG6: the core holds %d definitions named after a numerical method, more than the %d of b6; "
        "write the method in pops.lib instead:\n  %s"
        % (len(homes), _B6_METHOD_HOMES_IN_CORE, "\n  ".join("%s:%d %s (%s)" % h for h in homes)))


def test_method_homes_scan_sees_the_b6_natives():
    """Control: the scan finds the hand-written fluxes that spec 0.3 section 1.6 cites, and the
    solvers that the b6 version of this test pinned in the core (GeometricMG, FFT)."""
    names = {(path, name) for path, _, _, name in method_homes_in_core()}
    assert ("include/pops/numerics/fv/numerical_flux.hpp", "RusanovFlux") in names
    assert ("include/pops/numerics/fv/numerical_flux.hpp", "HLLFlux") in names
    assert ("python/pops/solvers/elliptic/_descriptor.py", "GeometricMG") in names
    assert ("python/pops/solvers/elliptic/_descriptor.py", "FFT") in names
    assert ("python/pops/solvers/krylov/__init__.py", "BiCGStab") in names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
