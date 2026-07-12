"""ADC-479 Spec 5 C1-System: named elliptic fields lower on the uniform System.

PURE-PYTHON tests of the install-seam routing for a NAMED elliptic field (m.elliptic_field): a
field a block's model DECLARES routes through the shared system elliptic solver (set_poisson), while
an UNDECLARED field name is rejected LOUD against the declared set (never a silent drop). The
declared set is collected exclusively from the ``CompiledModel.elliptic_field_names`` records in an
exact immutable phase records; bind never consults a raw authoring model.

Runs both under pytest and as a plain script; the CI runner executes it via the __main__ guard.
"""
import sys

try:
    import pops
    from pops.runtime._system_unified_install import _SystemUnifiedInstall
    from pops.codegen._plans import BindInputs, InstallPlan, ResolvedBlock, ResolvedSimulationPlan
    from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
    from pops.codegen.loader import CompiledModel
    from pops.model.bind_schema import BindSchema
    from pops.model import Module
    from pops.problem._snapshot import AuthoringSnapshot
except Exception as exc:  # noqa: BLE001
    print("skip test_unified_install_named_field_system (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check


def _compiled_model(*fields):
    """Return an inert, runtime-ready model carrying only immutable install metadata."""
    from pops.codegen._compiled_model_identity import compiled_model_identity
    model = CompiledModel(
        so_path="/fake.so", backend="production", adder="add_native_block",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20", target="system", elliptic_field_names=list(fields))
    model.definition_identity = compiled_model_identity(model_hash="h")
    return model


def _install_instances(**models):
    """Assemble runtime instances through the exact resolve/compile/bind records."""
    snapshot = AuthoringSnapshot({"kind": "named-field-system", "blocks": tuple(models)})
    schema = BindSchema()
    source = {"kind": "named-field-system", "blocks": tuple(models)}
    resolved = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="system",
        backend="production",
        layout=None,
        time={"method": "inert"},
        blocks=tuple(ResolvedBlock(name, source, None, "production") for name in models),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={},
        capabilities={},
    )
    artifact = CompiledSimulationArtifact(
        plan=resolved,
        program=_compiled_model(),
        blocks=tuple(
            CompiledBlockArtifact(name, model, None) for name, model in models.items()
        ),
    )
    inputs = BindInputs()
    plan = InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={name: {"model": model, "spatial": None} for name, model in models.items()},
        params=schema.resolve_bind({}, compile_values=resolved.compile_values),
        aux={},
    )
    return plan.instances


class _SolverHarness(_SystemUnifiedInstall):
    """Captures set_poisson calls without a real System (no engine, no Kokkos)."""

    def __init__(self):
        self.calls = []

    def set_poisson(self, **kw):
        self.calls.append(kw)


def test_compiled_model_carries_elliptic_field_names():
    """C1: CompiledModel carries the declared named-field set (elliptic_field_names)."""
    cm = CompiledModel(
        so_path="/fake.so", backend="production", adder="add_native_block",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20", elliptic_field_names=["psi", "chi"])
    _check(cm.elliptic_field_names == ["psi", "chi"], "CompiledModel keeps the declared names")
    # default: empty when none declared
    cm0 = CompiledModel(
        so_path="/fake.so", backend="production", adder="add_native_block",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20")
    _check(cm0.elliptic_field_names == [], "no declared field -> empty list")
    print("ok test_compiled_model_carries_elliptic_field_names")


def test_declared_elliptic_fields_collected_from_install_plan_instances():
    """C1: the declared set is the union of per-instance ``CompiledModel`` metadata."""
    instances = _install_instances(
        ne=_compiled_model("chi"),
        ni=_compiled_model("psi"),
    )
    declared = _SystemUnifiedInstall._declared_elliptic_fields(None, instances)
    _check(declared == {"psi", "chi"},
           "union of InstallPlan CompiledModel fields (got %r)" % declared)
    declared2 = _SystemUnifiedInstall._declared_elliptic_fields(
        None, _install_instances(b=_compiled_model("theta")))
    _check(declared2 == {"theta"},
           "CompiledModel.elliptic_field_names collected (got %r)" % declared2)
    print("ok test_declared_elliptic_fields_collected_from_install_plan_instances")


def test_install_solver_routes_default_poisson_field():
    """C1: the default Poisson field still routes to set_poisson (regression)."""
    h = _SolverHarness()
    h._install_solver("phi", "geometric_mg", frozenset())
    _check(h.calls and h.calls[0]["solver"] == "geometric_mg", "default field -> set_poisson")
    print("ok test_install_solver_routes_default_poisson_field")


def test_install_solver_routes_declared_named_field():
    """C1: a DECLARED named elliptic field routes through the shared elliptic solver (set_poisson)."""
    h = _SolverHarness()
    h._install_solver("psi", "geometric_mg", frozenset({"psi"}))
    _check(h.calls and h.calls[0]["solver"] == "geometric_mg",
           "a declared named field routes to set_poisson (shared solver)")
    print("ok test_install_solver_routes_declared_named_field")


def test_install_solver_rejects_undeclared_field():
    """C1: an UNDECLARED field name is a typo -- rejected LOUD, naming the declared set."""
    h = _SolverHarness()
    try:
        h._install_solver("psii", "geometric_mg", frozenset({"psi"}))
        raise AssertionError("an undeclared field name must raise")
    except ValueError as exc:
        _check("psii" in str(exc), "the reject names the offending field")
        _check("psi" in str(exc), "the reject names the declared set")
        _check("elliptic_field" in str(exc), "the reject points at m.elliptic_field")
    _check(not h.calls, "no set_poisson call on a rejected field")
    print("ok test_install_solver_rejects_undeclared_field")


def test_case_validate_accepts_named_field():
    """C1: a Problem with a named non-Poisson field VALIDATES (the whitelist reject is removed)."""
    from pops.fields import FieldProblem
    from pops.math import laplacian

    class _Solver:
        name = "GeometricMG"
        scheme = "geometric_mg"
        options = {}

    model = Module("named-field-validation")
    model.state_space("U", ("rho",))
    problem = pops.Problem().block("ne", physics=model)
    problem.field(FieldProblem(
        name="psi", unknown="psi", equation=(-laplacian("psi") == "rho"), solver=_Solver()))
    _check(problem.validate() is True, "a named non-Poisson field validates (C1-System)")
    print("ok test_case_validate_accepts_named_field")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
