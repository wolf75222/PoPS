"""ADC-428 Named elliptic fields on the AMR layout (codegen + install-seam + optional end-to-end).

PR #379 lowered named elliptic fields (m.elliptic_field) on the UNIFORM System; ADC-428 wires the AMR
equivalent: the AMR native loader emits the same register_elliptic_field + set_block_elliptic_field
calls (on the AmrSystem facade), the AmrSystem _install_solver routes a model-declared named field and
rejects a typo, and AmrRuntime owns a dedicated coarse solve per named field (readable via
sim.field(name)). The deep numerical validation of the second AMR solve lives in the C++ test
tests/cpp/integration/amr/test_amr_named_field.cpp (engine-level parity vs the default Poisson; no Kokkos .so gate).

Section A (pure Python, always runs): the AMR native loader EMITS the named-field registration on the
AmrSystem facade (no longer the ADC-428 NotImplementedError), and the AmrSystem install seam routes a
declared named field while rejecting an undeclared one.

Section B (gated, self-skip): the typed Problem end-to-end -- a default phi + a named field on an AMR
layout -- compiles (production .so), binds, runs a few steps, and sim.field(name) returns a solved,
non-trivial second field distinct from the default potential. Skips cleanly (exit 0) without _pops / a
compiler / a visible Kokkos -- never fakes the engine.
"""
import sys


from tests.python.support.assertions import _check
from pops.runtime.system import AmrSystem  # ADC-545 advanced runtime seam


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
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops.runtime.amr_system import AmrSystem
except Exception as exc:  # noqa: BLE001  -- pops not importable -> skip, never fake
    print("skip test_amr_named_field (pops unavailable: %s)" % exc)
    sys.exit(0)


_Q = -1.0  # charge sign (f = q * rho), like pops::ChargeDensity


def _isothermal_block(m):
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
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


class _RawModel:
    """A raw physics/dsl model stand-in exposing the _elliptic_fields mapping (m.elliptic_field)."""

    def __init__(self, fields=()):
        self._elliptic_fields = {n: {} for n in fields}


class _SolverHarness(AmrSystem):
    """Captures set_poisson without building a real AMR engine (no Kokkos)."""

    def __init__(self):
        self.calls = []

    def set_poisson(self, **kw):
        self.calls.append(kw)


def test_amr_declared_elliptic_fields_collected():
    """A3: the declared named-field set is gathered from the per-instance models (raw _elliptic_fields
    or a CompiledModel.elliptic_field_names)."""
    instances = {"plasma": {"model": _RawModel(fields=("psi",))},
                 "beam": {"model": _RawModel(fields=("chi",))}}
    declared = AmrSystem._declared_elliptic_fields(instances)
    _check(declared == {"psi", "chi"}, "union of the instance declared fields (got %r)" % declared)
    # a model exposing elliptic_field_names (CompiledModel shape) is read too.
    cm = type("CM", (), {"elliptic_field_names": ["theta"]})()
    _check(AmrSystem._declared_elliptic_fields({"b": {"model": cm}}) == {"theta"},
           "elliptic_field_names is collected from a compiled handle shape")
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


# =================== Section B: gated end-to-end on the AMR layout ===================
def _run_section_b():
    try:
        import numpy as np

        import pops
        from pops.mesh.cartesian import CartesianMesh
        from pops.mesh.layouts import AMR
        from pops.numerics.reconstruction import FirstOrder
        from pops.numerics.riemann import Rusanov
    except Exception as exc:  # noqa: BLE001
        print("-- (B) skipped: numpy/_pops/mesh unavailable: %s --" % exc)
        return True

    # _pops must expose the AMR named-field read-back binding (rebuild _pops if not).
    if not hasattr(AmrSystem(n=8, L=1.0, periodic=True), "named_field_values"):
        print("-- (B) skipped: _pops lacks AmrSystem.named_field_values (rebuild _pops) --")
        return True

    N = 32

    def _ic():
        x = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, x, indexing="ij")
        rho = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
        rho = rho - (rho.mean() - 1.0)  # zero-mean perturbation -> periodic Poisson solvable
        mx = 0.1 * rho
        my = -0.05 * rho
        return np.stack([rho, mx, my])

    # Build the named model and compile it for the AMR target (production .so). Kokkos-gated.
    model = _named_model("amr_e2e", scale=2.0)  # rhs = 2 * default -> psi = 2 * default phi
    try:
        compiled = model.compile(backend="production", target="amr_system")
    except (RuntimeError, NotImplementedError) as exc:
        print("-- (B) skipped: production AMR compile unavailable: %s --" % str(exc)[:160])
        return True

    # Install the compiled block on a native AmrSystem (the AMR-route runtime). The .so loader runs
    # register_elliptic_field + set_block_elliptic_field at add time, so the named field "psi" is
    # registered with its own coarse GeometricMG; sim.run() solves it each step (default Poisson
    # first, then the named field), and sim.field("psi") reads the solved coarse potential.
    from pops.runtime._bind_adapters import _amr_config_from_layout  # ADC-583: moved from codegen
    layout = AMR(CartesianMesh(n=N, L=1.0, periodic=True))
    sim = AmrSystem(_amr_config_from_layout(layout))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit())
    sim.set_density("plasma", _ic()[0])  # the coarse density (n x n); momentum starts at rest
    for _ in range(2):
        sim.step(1e-3)

    phi = np.asarray(sim.potential())
    psi = np.asarray(sim.field("psi"))
    _check(psi.shape == phi.shape, "sim.field('psi') returns the coarse (n, n) shape")
    _check(np.isfinite(psi).all(), "the named field is finite")
    _check(float(np.abs(psi - psi.mean()).max()) > 1e-6,
           "the named field is non-trivial (genuinely solved, not a zero no-op)")
    # psi = 2 * default phi (rhs = 2 * default, Poisson linear) -> psi is DISTINCT from phi.
    dphi = float(np.abs(phi - phi.mean()).max())
    dpsi = float(np.abs(psi - psi.mean()).max())
    _check(dphi > 1e-6 and abs(dpsi - 2.0 * dphi) < 1e-2 * dpsi,
           "the named field (rhs=2*default) solves to ~2x the default potential (distinct + scaled)")
    print("ok test_amr_named_field_end_to_end (psi span %.3e == 2 x phi span %.3e)" % (dpsi, dphi))
    return True


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d Section-A test(s) passed" % len(funcs))
    _run_section_b()


if __name__ == "__main__":
    _run_all()
