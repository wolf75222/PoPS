"""ADC-566: pops.lib is a leaf, locked to models / time / presets.

``pops.lib`` holds ONLY ready-to-use things (provided models, provided time schemes, compose-and-go
presets). It must stay a leaf of the import graph and must never become a second home for a central
object (a flux, a solver, a field problem, an AMR descriptor, a runtime param). This file fences all
four boundaries.

It EXTENDS, and does not re-implement, two existing fences (cited inline so the boundary is visible):

  * ``test_no_runtime_imports.py`` catches a MODULE-SCOPE (``col_offset == 0``) import of
    ``_pops`` / ``pops.codegen`` / ``pops.runtime`` in the symbolic layers, lib included. THIS file's
    gate 2b adds the LAZY (in-function, ``col_offset > 0``) case for lib, which that fence leaves
    alone by design.
  * ``test_import_graph.py`` locks the lib -> {ir, model, time, physics, moments} module edges. THIS
    file asserts CONTENT (no central class defined or re-exported under lib) and the DIRECTORY set,
    which the layering fence does not check.

Source-only AST scans (run without the native extension) plus skip-clean functional proofs that
import ``pops``. ASCII only.
"""
import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "python" / "pops" / "lib"

# The strict directory allow: pops.lib has exactly these immediate child packages. A new
# python/pops/lib/riemann/ or .../solvers/ would grow this set and fail (an ALLOW of 3 named dirs,
# not a broad pattern).
_ALLOWED_LIB_CHILD_DIRS = {"models", "time", "presets"}

# The HARD refusal list: lib is a leaf, so it must import none of these at ANY scope (module-scope OR
# lazily inside a function). This is the delta over test_no_runtime_imports.py (module-scope only).
_FORBIDDEN_IMPORT_ROOTS = ("_pops", "pops.codegen", "pops.runtime", "pops._pops")

# The descriptor/authoring packages lib MAY import (compose descriptors). Everything under pops.* that
# is NOT in this set and NOT forbidden is flagged: a new pops.<other> edge from lib must be reviewed.
# (numerics / solvers / mesh are descriptor catalogs used by the time schemes and presets.)
_ALLOWED_POPS_IMPORT_ROOTS = (
    "pops",  # the facade (import pops) -- presets compose through it
    "pops.ir",
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

# Central object names that have exactly ONE public home elsewhere; lib must neither DEFINE nor
# RE-EXPORT any of them (a second path). Canonical homes (asserted below): mesh AMR, elliptic
# GeometricMG, params RuntimeParam, fields PoissonProblem/FieldProblem, numerics HLL/MUSCL/...,
# time Program/Module.
_CANONICAL_NAMES = {
    "HLL", "HLLC", "Roe", "Rusanov", "MUSCL", "WENO5",
    "PoissonProblem", "FieldProblem", "GeometricMG", "FFT",
    "AMR", "RuntimeParam", "Program", "Module",
}

# lib/__init__.py must stay thin (currently 31 lines); a fat __init__ that concentrates logic fails.
_LIB_INIT_LINE_CAP = 40


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
# Gate 2a -- DIRECTORY fence: lib's child packages are exactly {models, time, presets}.
# ---------------------------------------------------------------------------------------------
def test_lib_child_directories_are_the_strict_allow_set():
    children = {p.name for p in LIB.iterdir()
                if p.is_dir() and p.name != "__pycache__"}
    extra = children - _ALLOWED_LIB_CHILD_DIRS
    missing = _ALLOWED_LIB_CHILD_DIRS - children
    assert not extra, (
        "pops.lib may only hold ready-to-use packages %s; a new child package is a second home for a "
        "central object -- refused: %s"
        % (sorted(_ALLOWED_LIB_CHILD_DIRS), sorted(extra)))
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
# Gate 2c -- CONTENT fence: no canonical object DEFINED or RE-EXPORTED under lib.
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
    """A ``from pops.solvers import GeometricMG`` (or an __all__ surfacing it) under lib is a second
    path to a central object -- refused via a MODULE-SCOPE ImportFrom + __all__ scan.

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
    """Belt-and-braces: the canonical objects DO live in their documented non-lib home, so the
    content fence is guarding a real single-home invariant, not an empty set."""
    homes = {
        "AMR": REPO_ROOT / "python/pops/mesh/layouts/__init__.py",
        "GeometricMG": REPO_ROOT / "python/pops/solvers/elliptic/_descriptor.py",
        "RuntimeParam": REPO_ROOT / "python/pops/params/runtime.py",
        "PoissonProblem": REPO_ROOT / "python/pops/fields/poisson.py",
        "FieldProblem": REPO_ROOT / "python/pops/fields/problem.py",
        "Program": REPO_ROOT / "python/pops/time/program.py",
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
# Gate 2d -- lib.__init__ stays thin; macros return Program; models lower runtime-free.
# ---------------------------------------------------------------------------------------------
def test_lib_init_stays_thin():
    init = LIB / "__init__.py"
    lines = len(_read(init).splitlines())
    assert lines <= _LIB_INIT_LINE_CAP, (
        "pops.lib.__init__ must stay thin (<= %d lines); a fat __init__ concentrating logic is "
        "refused, got %d" % (_LIB_INIT_LINE_CAP, lines))


def test_lib_time_macros_return_a_core_program():
    """Functional (skip-clean): each typed lib.time macro produces a pops.time.Program.

    The remaining macros (strang / imex / bdf / predictor_corrector) require extra scheme arguments;
    they share the same @program_macro dispatch (lib/time/_helpers.py), so these four schemes are a
    representative proof that lib.time lowers to the core Program, not a lib stepper.  The fixture
    deliberately supplies the authoritative BlockHandle and model state Handle: a display label is
    never promoted into semantic ownership."""
    try:
        import pops.lib.time as lib_time
        from pops.model import Module
        from pops.problem import Problem
        from pops.time import Program
    except Exception as exc:  # pragma: no cover - bare source tree without importable pops.
        pytest.skip("pops import unavailable: %s" % exc)

    module = Module("architecture-time-schemes")
    state_space = module.state_space("U", ("u",))
    state = module.state_handle(state_space)
    block = Problem(name="architecture-time-case").add_block("plasma", module)
    for name in ("forward_euler", "ssprk2", "ssprk3", "rk4"):
        program = getattr(lib_time, name)(block, state, sources=(), flux=False)
        assert isinstance(program, Program), (
            "pops.lib.time.%s must return a pops.time.Program, got %r" % (name, type(program)))


def test_lib_models_lower_to_physics_without_runtime():
    """Functional (skip-clean): a provided model lowers to a pops.model/physics object whose manifest
    carries NO runtime/compiled fields (no .so path, no abi_key)."""
    try:
        from pops.lib.models import Gaussian, HyQMOM15
    except Exception as exc:  # pragma: no cover - bare source tree without importable pops.
        pytest.skip("pops import unavailable: %s" % exc)

    for factory in (lambda: HyQMOM15.vlasov_poisson_magnetic(order=4),
                    lambda: Gaussian.transport()):
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
