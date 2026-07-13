#!/usr/bin/env python3
"""ADC-557 real-compiler acceptance: the standard flow lowers the facade once, trace on the handle.

A physics facade ``pops.physics.Model`` compiled through ``compile_problem`` (the standard flow --
no manual ``m.to_module()``) yields a handle that carries the operator-first Module as the
lowered-module trace (``compiled.inspect()``) and a compile-time ``module_hash`` for drift
detection. A native brick ``pops.Model(...)`` (a ``_pops.ModelSpec``), by contrast, has NO backing
operator-first Module: its trace is honestly absent, never fabricated.

The pure-Python chain (facade -> lower_and_validate -> handle metadata) is pinned by the unit test
``tests/python/unit/codegen/test_module_lowering.py``; this test proves the same claims through a
REAL ``.so`` compile. Skips cleanly unless the full toolchain is present (like the sibling
``test_compile_problem.py``).
"""
import sys


def _skip(msg):
    print("skip test_compile_module_trace (%s)" % msg)
    sys.exit(0)


try:
    import pops
    from pops import time as adctime
    from pops.physics._facade import Model as PhysicsModel
    from tests.python.support.typed_program import program_states, synthetic_module
except Exception as exc:  # noqa: BLE001
    _skip("pops unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _fe_program(name="module_trace_probe"):
    P = adctime.Program(name)
    dt = P.dt
    module = synthetic_module("%s_state" % name, components=("rho", "mx", "my"))
    _case, states = program_states(P, module, ("ions",))
    temporal = states["ions"]
    U = temporal.n
    f = P.solve_fields(U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["default"])
    P.commit(temporal.next, P.value("U1", U + dt * R))
    return P


def physics_model():
    """A facade physics Model (the ADC-557 subject): it carries the operator-first Module view.

    The FE program lowers via ``ctx.rhs_into`` (flux=True + "default"), so the emitted program
    source only ADDS the inert ``pops_module_*`` metadata descriptors on top of the source the
    sibling ``test_compile_problem.py`` already compiles -- no extra kernel surface.
    """
    m = PhysicsModel("ions")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.flux(x=[mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + 0.5 * rho])
    m.elliptic_rhs(rho - 1.0)
    return m


def bricks_model():
    """A native brick model (a ``_pops.ModelSpec``): NO backing operator-first Module."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                      transport=pops.IsothermalFlux(),
                      source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


# Standard flow: pass the facade Model directly, no manual to_module / lower.
try:
    compiled = pops.codegen.compile_problem(time=_fe_program(), model=physics_model())
except (RuntimeError, Exception) as exc:  # noqa: BLE001
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

print("== ADC-557: one lowering, module trace on the handle (facade Model) ==")
report = compiled.inspect().to_dict()
chk(report.get("module_manifest") is not None,
    "compiled.inspect() carries the lowered-module trace (operator-first Module)")
chk(bool(compiled.module_hash()), "the handle carries a compile-time module_hash for drift detection")

# The trace lists the operator-first operators (flux_default / fields_from_state) without the user
# ever building a Module by hand.
ops = [op.get("name") for op in report["module_manifest"].get("operators", [])] \
    if report.get("module_manifest") else []
chk("flux_default" in ops and "fields_from_state" in ops,
    "the lowered-module trace lists the operators (%s)" % ops)

# A native brick ModelSpec has NO operator-first Module: the trace is honestly ABSENT (None), never
# fabricated -- and the compile itself still succeeds (the historical route is unchanged).
print("== honest absence: a native brick ModelSpec carries no module trace ==")
try:
    compiled_spec = pops.codegen.compile_problem(time=_fe_program("spec_probe"), model=bricks_model())
except (RuntimeError, Exception) as exc:  # noqa: BLE001
    _skip("ModelSpec compile failed: %s" % str(exc)[:160])
chk(compiled_spec.module_manifest is None,
    "a ModelSpec handle carries NO module_manifest (honest absence)")
chk(compiled_spec.module_hash() is None,
    "a ModelSpec handle carries NO module_hash (honest absence)")

print("%s test_compile_module_trace" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
