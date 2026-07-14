"""ADC-428 Named elliptic fields on the AMR layout (codegen + install seam).

PR #379 lowered named elliptic fields (m.elliptic_field) on the UNIFORM System; ADC-428 wires the AMR
equivalent: the AMR native loader emits the same register_elliptic_field + set_block_elliptic_field
calls (on the AmrSystem facade), the AmrSystem _install_solver routes a model-declared named field and
rejects a typo, and AmrRuntime owns a dedicated coarse solve per named field (readable via
sim.field(name)). The deep numerical validation of the second AMR solve lives in the C++ test
tests/cpp/integration/amr/test_amr_named_field.cpp (engine-level parity vs the default Poisson; no Kokkos .so gate).

Section A (pure Python, always runs): the AMR native loader EMITS the named-field registration on the
AmrSystem facade (no longer the ADC-428 NotImplementedError), and the AmrSystem install seam routes a
declared named field while rejecting an undeclared one.

The typed Case integration matrix owns end-to-end coverage through the public structured descriptor.
This focused module deliberately does not construct a native runtime from a retired layout API.
"""
import sys


from tests.python.support.assertions import _check
from tests.python.support.layout_plan import resolved_layout_contract
from pops.runtime._system import AmrSystem  # ADC-545 advanced runtime seam


def _raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure
        return False
    return False


# =================== Section A: pure Python (codegen + install seam) ===================
try:
    from pops.codegen._plans import BindInputs, InstallPlan, ResolvedBlock, ResolvedSimulationPlan
    from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
    from pops.codegen.loader import CompiledModel
    from pops.model.bind_schema import BindSchema
    from pops.params import ConstParam
    from pops.ir.ops import sqrt
    from pops.physics._facade import Model
    from pops.problem._snapshot import AuthoringSnapshot
    from pops.runtime._amr_system import AmrSystem
except Exception as exc:  # noqa: BLE001  -- pops not importable -> skip, never fake
    print("skip test_amr_named_field (pops unavailable: %s)" % exc)
    sys.exit(0)


_Q = -1.0  # charge sign (f = q * rho), like pops::ChargeDensity


def _isothermal_block(m):
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.value(m.param(ConstParam("cs2", 0.5)))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    m.elliptic_rhs(_Q * rho)  # default Poisson coupling f = q * rho
    return rho


def _named_model(name="amr_named", scale=1.0):
    """Default Poisson PLUS a named field 'psi' with rhs = scale * (default RHS)."""
    m = Model(name)
    rho = _isothermal_block(m)
    g2x = m.aux_field("g2x")
    g2y = m.aux_field("g2y")
    m.aux_field("psi")  # the named field's potential slot (written C++-side)
    m.elliptic_field("psi", rhs=(scale * _Q) * rho, aux=["psi", "g2x", "g2y"])
    # the source reads the named field's gradient so the field is genuinely consumed.
    m.source([0.0 * rho, -rho * g2x, -rho * g2y])
    return m


def test_amr_loader_emits_named_field_registration():
    """A1: the AMR native loader EMITS register_elliptic_field + set_block_elliptic_field on the
    AmrSystem facade (replacing the ADC-428 NotImplementedError), exactly like the uniform loader."""
    loader = _named_model("amr_emit")._m.emit_cpp_native_loader(target="amr_system")
    _check('register_elliptic_field("psi"' in loader,
           "the AMR named field registers its aux components")
    _check("set_block_elliptic_field" in loader and "make_poisson_rhs" in loader,
           "the AMR named field attaches its RHS closure (make_poisson_rhs of the brick)")
    _check("pops::AmrSystem*" in loader, "the registration targets the AmrSystem facade")
    _check("Ell_psi" in loader, "the named elliptic RHS brick is emitted")
    print("ok test_amr_loader_emits_named_field_registration")


def test_amr_loader_no_named_field_unchanged():
    """A2: a default-only model still emits the AMR loader with NO named-field registration (the named
    code path is inert -> the default AMR install is bit-identical)."""
    m = Model("amr_plain")
    _isothermal_block(m)
    loader = m._m.emit_cpp_native_loader(target="amr_system")
    _check("register_elliptic_field" not in loader,
           "a default-only model emits no named-field registration on the AMR loader")
    _check("pops_install_native_amr" in loader, "the AMR loader is still emitted")
    print("ok test_amr_loader_no_named_field_unchanged")


def _compiled_model(*fields):
    """Return an inert AMR loader record with detached named-field metadata."""
    from pops.codegen._compiled_model_identity import compiled_model_identity
    model = CompiledModel(
        so_path="/fake.so", backend="production",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20", target="amr_system",
        elliptic_field_names=list(fields))
    model.definition_identity = compiled_model_identity(model_hash="h")
    return model


def _install_instances(**models):
    """Assemble runtime inputs from the exact AMR resolve/compile/bind contract."""
    snapshot = AuthoringSnapshot({"kind": "named-field-amr", "blocks": tuple(models)})
    schema = BindSchema()
    source = {"kind": "named-field-amr", "blocks": tuple(models)}
    layout_plan, layout_coverage = resolved_layout_contract(
        None, target="amr_system", block_names=models)
    resolved = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="amr_system",
        backend="production",
        layout=None,
        layout_plan=layout_plan,
        time=None,
        blocks=tuple(
            ResolvedBlock(name, source, None, "production", ("U",))
            for name in models
        ),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_plans={},
        libraries=(),
        requirements={},
        capabilities={"amr": True},
        lowering_coverage=layout_coverage,
    )
    artifact = CompiledSimulationArtifact(
        plan=resolved,
        program=None,
        blocks=tuple(
            CompiledBlockArtifact(name, model, None, ("U",))
            for name, model in models.items()
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


class _SolverHarness(AmrSystem):
    """Captures set_poisson without building a real AMR engine (no Kokkos)."""

    def __init__(self):
        self.calls = []

    def set_poisson(self, **kw):
        self.calls.append(kw)


def test_amr_declared_elliptic_fields_collected():
    """A3: the declared set is gathered from per-instance ``CompiledModel`` metadata."""
    instances = _install_instances(
        plasma=_compiled_model("psi"),
        beam=_compiled_model("chi"),
    )
    declared = AmrSystem._declared_elliptic_fields(instances)
    _check(declared == {"psi", "chi"}, "union of the instance declared fields (got %r)" % declared)
    theta = _install_instances(b=_compiled_model("theta"))
    _check(AmrSystem._declared_elliptic_fields(theta) == {"theta"},
           "elliptic_field_names is collected from a real CompiledModel")
    print("ok test_amr_declared_elliptic_fields_collected")


def test_amr_install_solver_routes_declared_named_field():
    """A4: a DECLARED named elliptic field routes through the shared AMR elliptic solver (set_poisson)."""
    h = _SolverHarness()
    h._install_solver("psi", "geometric_mg", frozenset({"psi"}))
    _check(h.calls and h.calls[0]["solver"] == "geometric_mg",
           "a declared named field routes to set_poisson (shared solver)")
    # the default Poisson field still routes (regression).
    h2 = _SolverHarness()
    h2._install_solver("phi", "geometric_mg", frozenset())
    _check(h2.calls and h2.calls[0]["solver"] == "geometric_mg", "default field -> set_poisson")
    print("ok test_amr_install_solver_routes_declared_named_field")


def test_amr_install_solver_rejects_undeclared_field():
    """A5: an UNDECLARED field name is a typo -- rejected LOUD, naming the declared set (not deferred)."""
    h = _SolverHarness()
    try:
        h._install_solver("psii", "geometric_mg", frozenset({"psi"}))
        raise AssertionError("an undeclared field name must raise")
    except ValueError as exc:
        _check("psii" in str(exc), "the reject names the offending field")
        _check("psi" in str(exc), "the reject names the declared set")
        _check("elliptic_field" in str(exc), "the reject points at m.elliptic_field")
    _check(not h.calls, "no set_poisson call on a rejected field")
    print("ok test_amr_install_solver_rejects_undeclared_field")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d Section-A test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
