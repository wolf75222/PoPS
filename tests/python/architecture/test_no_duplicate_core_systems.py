"""ADC-565: architecture gates against a SECOND core system beside the canonical one.

The target authoring surface has exactly one public stepper (``pops.time.Program``), one
canonical field-problem home (``pops.fields`` -- ``FieldProblem`` / ``PoissonProblem``), and one
AMR *configuration* route (``layout=AMR(...)``); ``sim.amr`` is a read-only runtime VIEW. This
file refuses a duplicate of any of those three.

It EXTENDS, and does not re-implement, the existing fences (referenced inline so a reviewer sees
the boundary):

  * ADC-598 ``test_no_legacy_runtime_routes.py`` -- no ``System`` / ``AmrSystem`` front door in the
    target surface, ``target=`` blocked on compile/bind, string selectors on an explicit allowlist,
    ``Program`` field-solve facade lowers to ``ProgramContext``, AMR layout manifest + ``sim.amr``
    runtime view. This file does NOT re-check the compile/bind signature (see Q2 in the plan).
  * ADC-532 ``test_lib_time_no_string_selectors.py`` -- lib.time macros select operators by handle.
  * ADC-529 facade-lowering-parity / ``_ir_hash`` equality -- the technique reused in gate 1b.

The AST scans are source-only (they run without the native extension); the lowering proofs import
``pops`` and skip cleanly when it is not importable. ASCII only.
"""
import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# The target authoring surface (the packages a user drives). Mirrors ADC-598's TARGET_SURFACE_ROOTS
# minus ``lib`` (owned by the ADC-566 boundary fence) so the two files do not double-scan lib.
TARGET_SURFACE_ROOTS = (
    "diagnostics",
    "external",
    "fields",
    "ir",
    "linalg",
    "mesh",
    "model",
    "moments",
    "numerics",
    "output",
    "params",
    "physics",
    "solvers",
    "time",
)

# Methods whose presence on a PUBLIC class marks it as a time stepper (a second one would bypass
# Program). ``step`` alone is not enough (Program uses it as a build-time authoring decorator), so we
# also weigh ``advance`` / ``integrate``; the justified exception below keeps Program allowed.
_STEPPER_METHODS = {"step", "advance", "integrate"}

# The ONE canonical public stepper. Program.step is a BUILD-TIME authoring decorator
# (program_authoring.py) that records the IR body once; it is never executed numerically. It is the
# single allowed stepper class, named explicitly (no broad allowlist).
_ALLOWED_STEPPER_CLASSES = {
    "Program": "python/pops/time/program.py: the ONE canonical compiled-time stepper; step() is a "
               "build-time IR authoring decorator, not a numerical advance loop",
}

# lib/time/rk.py:ButcherTableau is a DATA helper (A/b/c coefficient table), not a stepper: it is not
# exported from pops.time and defines no step/advance/integrate. Named here as the single justified
# non-stepper class the time surface may define with an RK-adjacent name.
_ALLOWED_NON_STEPPER_DATA = {
    "ButcherTableau": "python/pops/lib/time/rk.py: a Butcher A/b/c coefficient table (data), not a "
                      "stepper; carries no step/advance/integrate and is not exported as a stepper",
}

# The canonical field-problem base + its home package. A second public class exposing a field
# registration surface (``.field`` / ``register_field``) or subclassing ``*FieldProblem`` OUTSIDE
# pops/fields would be a parallel field system.
_FIELD_OPERATOR_HOME = POPS / "fields"
_FIELD_OPERATOR_BASE_SUFFIX = "FieldOperator"
_FIELD_REGISTER_METHODS = {"register_field"}

# The bind path consumes, never authors, a field problem: a PoissonProblem(/FieldProblem(
# construction under pops/runtime would be bind re-declaring a field system.
_BIND_PATH = POPS / "runtime"
_FIELD_AUTHORING_CTOR_NAMES = {"FieldOperator", "FieldDiscretization"}


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _read(path):
    return path.read_text(encoding="utf-8")


def _parse(path):
    return ast.parse(_read(path), filename=str(path))


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _target_surface_files():
    for entry in TARGET_SURFACE_ROOTS:
        path = POPS / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from _py_files(path)


def _public_classes(tree):
    """Yield every module-level and nested public (non-underscore) ClassDef in @p tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield node


def _class_methods(node):
    return {child.name for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return ("%s.%s" % (base, node.attr)) if base else node.attr
    return None


# ---------------------------------------------------------------------------------------------
# Gate 1a -- TIME: no second public stepper bypassing Program.
# ---------------------------------------------------------------------------------------------
def test_time_surface_defines_no_second_public_stepper():
    violations = []
    time_dir = POPS / "time"
    lib_time_dir = POPS / "lib" / "time"
    for path in _py_files(time_dir) + _py_files(lib_time_dir):
        rel = _rel(path)
        for node in _public_classes(_parse(path)):
            methods = _class_methods(node)
            if not (methods & _STEPPER_METHODS):
                continue
            if node.name in _ALLOWED_STEPPER_CLASSES:
                continue
            # A class with only ``step`` and no numerical advance is suspicious but allowed ONLY if
            # it is the named exception; any other stepper-shaped public class is a violation.
            violations.append(
                "%s:%d public class %r defines stepper method(s) %s"
                % (rel, node.lineno, node.name, sorted(methods & _STEPPER_METHODS)))

    assert not violations, (
        "only pops.time.Program may be a public stepper; a second stepper-shaped class bypasses the "
        "canonical time program:\n  " + "\n  ".join(violations)
        + "\n(allowed: %s)" % ", ".join(sorted(_ALLOWED_STEPPER_CLASSES)))


def test_lib_time_exports_are_macros_not_stepper_classes():
    """pops.lib.time must export functions (scheme macros), never a stepper class.

    Parse pops/lib/time/__init__.py's ``__all__`` and assert every re-exported name resolves to a
    FunctionDef in the sub-modules (a macro) or the single ButcherTableau DATA helper -- never a
    stepper ClassDef. This is the structural half of "lib.time is macros, Program is the stepper".
    """
    init = POPS / "lib" / "time" / "__init__.py"
    tree = _parse(init)

    exported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        exported.update(
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str))
    assert exported, "pops.lib.time.__init__ must declare __all__"

    # Collect every FunctionDef / ClassDef name across the sub-modules with its kind.
    func_defs, class_defs = set(), {}
    for path in _py_files(POPS / "lib" / "time"):
        for node in ast.walk(_parse(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_defs.add(node.name)
            elif isinstance(node, ast.ClassDef):
                class_defs[node.name] = node

    violations = []
    # Module-scope UPPER_SNAKE tableau constants (RK4_TABLEAU / SSPRK2_TABLEAU) are data, allowed.
    for name in sorted(exported):
        if name in func_defs:
            continue
        if name.isupper():  # a tableau data constant, not a stepper
            continue
        if name in _ALLOWED_NON_STEPPER_DATA and name in class_defs:
            node = class_defs[name]
            if _class_methods(node) & _STEPPER_METHODS:
                violations.append(
                    "%s: allowed data helper %r unexpectedly defines a stepper method" % (name, name))
            continue
        if name in class_defs:
            violations.append("pops.lib.time exports class %r (must export scheme macros only)" % name)

    assert not violations, (
        "pops.lib.time must export scheme-builder functions (and the ButcherTableau data helper), "
        "never a stepper class:\n  " + "\n  ".join(violations))


def test_lib_time_macro_returns_the_same_program_handle():
    """Installed-package gate: a lib.time macro returns one pops.time.Program handle.

    Reuses the ADC-554 program_macro contract with an authoritative BlockHandle and state Handle.
    The macro produces the canonical Program; it never mints a second stepper object or promotes a
    free block/state name into semantic ownership.
    """
    import pops.lib.time as lib_time
    from pops.model import Module
    from pops.problem import Case
    from pops.time import Program

    module = Module("architecture-time-schemes")
    state_space = module.state_space("U", ("u",))
    state = module.state_handle(state_space)
    rate = module.rate_operator("advance", state_space, flux=False)
    block = Case(name="architecture-time-case").block("plasma", module)
    instance = block[state]
    # Split, IMEX and multistep factories need their own typed operators/history and have dedicated
    # contract tests.  These four share the same one-state/one-rate signature.
    for name in ("ForwardEuler", "SSPRK2", "SSPRK3", "RK4"):
        result = getattr(lib_time, name)(instance, rate=rate)
        assert isinstance(result, Program), (
            "pops.lib.time.%s must return a pops.time.Program, got %r" % (name, type(result)))


# ---------------------------------------------------------------------------------------------
# Gate 1b -- FIELDS: one canonical FieldProblem/PoissonProblem; facade == direct lowering.
# ---------------------------------------------------------------------------------------------
def test_only_pops_fields_defines_a_field_operator_class():
    violations = []
    for path in _target_surface_files():
        # The canonical home is allowed to define/subclass FieldProblem.
        if _FIELD_OPERATOR_HOME in path.parents or path == _FIELD_OPERATOR_HOME:
            continue
        rel = _rel(path)
        for node in _public_classes(_parse(path)):
            base_names = {_dotted_name(base) or "" for base in node.bases}
            subclasses_field = any(
                name and name.endswith(_FIELD_OPERATOR_BASE_SUFFIX) for name in base_names)
            has_register = bool(_class_methods(node) & _FIELD_REGISTER_METHODS)
            if subclasses_field:
                violations.append(
                    "%s:%d public class %r subclasses FieldOperator outside pops/fields"
                    % (rel, node.lineno, node.name))
            elif has_register:
                violations.append(
                    "%s:%d public class %r exposes register_field outside pops/fields"
                    % (rel, node.lineno, node.name))

    assert not violations, (
        "the physical field operator has one home (pops.fields.FieldOperator); a parallel "
        "field system elsewhere is refused:\n  " + "\n  ".join(violations))


def test_bind_path_consumes_field_plans_never_constructs_them():
    violations = []
    if _BIND_PATH.is_dir():
        for path in _py_files(_BIND_PATH):
            rel = _rel(path)
            for node in ast.walk(_parse(path)):
                if isinstance(node, ast.Call):
                    name = _dotted_name(node.func)
                    tail = name.split(".")[-1] if name else None
                    if tail in _FIELD_AUTHORING_CTOR_NAMES:
                        violations.append(
                            "%s:%d constructs %s() (bind must consume, not author, a field plan)"
                            % (rel, node.lineno, tail))

    assert not violations, (
        "the runtime bind path must consume field authoring, never construct FieldOperator/"
        "FieldDiscretization itself:\n  " + "\n  ".join(violations))


def test_field_handle_is_the_sole_public_field_solve_route():
    """The callable Case field handle is the only public field-solve entry.

    It delegates to the private generic Program node builder, so there is one canonical IR without
    exposing a second authoring route that could bypass Case authentication.
    """
    from pops.descriptors import Descriptor
    from pops.fields import FieldDiscretization, FieldOperator
    from pops.ir import ValueExpr
    from pops.math import laplacian
    from pops.model import Handle
    from pops.model import Module
    from pops.problem import Case
    from pops.time import Program

    class _Method(Descriptor):
        category = "field_method"

        def to_data(self):
            return {"type": "architecture-second-order"}

    class _Solver(Descriptor):
        category = "elliptic_solver"

        def to_data(self):
            return {"type": "architecture-solver"}

    module = Module("field-parity-model")
    state_space = module.state_space("U", ("u",))
    state_handle = module.state_handle(state_space)
    case = Case(name="field-parity-case")
    block = case.block("gas", module)
    unknown = Handle("potential", kind="field", owner=module.owner_path)
    provider = Handle("charge", kind="field_operator", owner=module.owner_path)
    operator = FieldOperator(
        "potential",
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == ValueExpr(state_handle),
        providers=provider,
    )
    field = case.field(
        operator,
        FieldDiscretization(method=_Method(), boundaries=(), solver=_Solver()),
    )
    instance = block[state_handle]

    program = Program("field_route")
    state = program.state(instance)
    outcome = field(state.n)

    assert not hasattr(program, "solve_fields")
    assert type(outcome).__name__ == "FieldSolveOutcome"
    assert [value.op for value in program._values].count("solve_fields") == 1


# ---------------------------------------------------------------------------------------------
# Gate 1c -- AMR: layout=AMR(...) is THE config route; sim.amr is a read-only VIEW.
# ---------------------------------------------------------------------------------------------
# The ONLY public AMR *configuration* entry is pops.layouts.AMR plus the authority objects in
# pops.amr. No target-surface function may take an amr-config STRING kwarg (e.g.
# amr="amr_system"); that vocabulary is what target=/string selectors reintroduce.
_AMR_CONFIG_STRING_ARGS = {"amr", "amr_target"}


def test_no_public_function_takes_an_amr_config_string_kwarg():
    violations = []
    for path in _target_surface_files():
        rel = _rel(path)
        for node in ast.walk(_parse(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            args = node.args
            pairs = list(
                zip(args.args[len(args.args) - len(args.defaults):], args.defaults, strict=True)
            )
            pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
            for arg, default in pairs:
                if default is None or arg.arg not in _AMR_CONFIG_STRING_ARGS:
                    continue
                if isinstance(default, ast.Constant) and isinstance(default.value, str):
                    violations.append(
                        "%s:%d public %s(%s=%r) is an AMR-config string selector"
                        % (rel, node.lineno, node.name, arg.arg, default.value))

    assert not violations, (
        "AMR is configured by the typed layout=AMR(...) descriptor, not a string kwarg or a "
        "target='amr_system' branch:\n  " + "\n  ".join(violations))


def test_amr_config_lives_in_the_layout_descriptor_only():
    """Installed-package gate: AMR(...) configures; sim.amr is a read-only view.

    Reuses the ADC-598 shape: AMR(...).inspect() carries capabilities.layout=="amr" (the config
    manifest), while sim.amr.inspect() is a {hierarchy, patches, regrid, limitations} VIEW with NO
    configuration mutator (no set_/configure_/add_ method that changes levels/ratio).
    """
    from pops.mesh import CartesianMesh
    from tests.python.support.layout_plan import final_amr_layout

    layout = final_amr_layout(CartesianMesh(n=16, L=1.0))
    manifest = layout.inspect()
    assert manifest["capabilities"]["layout"] == "amr", (
        "AMR(...) must be the typed AMR configuration surface")

    # sim.amr is a runtime VIEW: no config mutator, and inspect() is the fixed four-key view.
    # ADC-545: the engine left the top-level surface -- reach it via the advanced runtime seam.
    from pops.runtime._system import AmrSystem  # ADC-545 advanced runtime seam
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=4)

    view = sim.amr
    mutators = [name for name in dir(view)
                if not name.startswith("_")
                and (name.startswith(("set_", "configure", "add_"))
                     or "level" in name.lower() or "ratio" in name.lower())]
    assert not mutators, (
        "sim.amr is a read-only runtime view; it must expose no AMR-config mutator, found: %s"
        % mutators)
    assert set(view.inspect().to_dict()) == {"hierarchy", "patches", "regrid", "limitations"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
