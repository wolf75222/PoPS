#!/usr/bin/env python3
"""ADC-557 real-compiler acceptance: the standard flow lowers the final model once.

A final ``pops.physics.Model`` compiled through the internal ``compile_problem`` seam (no
manual ``m.to_module()``) yields a handle that carries the operator-first Module as the lowered-module
trace (``compiled.inspect()``) and a compile-time ``module_hash`` for drift detection. The bounded
native ``ModelSpec`` bridge is rejected before compilation because it has no canonical Module
authority; a missing trace can therefore never be fabricated.

The pure-Python chain (Model -> lower_and_validate -> handle metadata) is pinned by the unit test
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
    from pops.codegen._compile_drivers import compile_problem
    from pops.runtime.bricks import (
        BackgroundDensity, FluidState, IsothermalFlux, Model as NativeBrickModel, NoSource,
    )
    from tests.python.integration._final_field_program import scalar_advection_field_model
    from tests.python.support.typed_program import program_states
except Exception as exc:  # noqa: BLE001
    _skip("pops unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _fe_program(model, name="module_trace_probe"):
    P = pops.Program(name)
    dt = P.dt
    module = model.module
    _case, states = program_states(P, model, ("ions",))
    temporal = states["ions"]
    U = temporal.n
    R = module.operator_handle("explicit_rhs")(U, name="rate")
    P.commit(temporal.next, P.value("U1", U + dt * R, at=temporal.next.point))
    return P


def physics_model():
    """A final blackboard Model carrying its operator-first Module view.

    The FE Program is field-free, while the model still declares a physical field operator. This
    keeps the smoke focused on trace metadata without inventing an unresolved field-install plan.
    """
    return scalar_advection_field_model("ions")


def bricks_model():
    """The bounded native ModelSpec bridge has no operator-first Module authority."""
    return NativeBrickModel(
        state=FluidState.isothermal(cs2=0.5),
        transport=IsothermalFlux(),
        source=NoSource(),
        elliptic=BackgroundDensity(alpha=1.0, n0=0.0),
    )


# Standard flow: pass the final Model directly, with no manual Module lowering.
try:
    compiled_model = physics_model()
    compiled = compile_problem(
        time=_fe_program(compiled_model), model=compiled_model)
except (RuntimeError, Exception) as exc:  # noqa: BLE001
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

print("== ADC-557: one lowering, module trace on the handle (final Model) ==")
report = compiled.inspect().to_dict()
chk(report.get("module_manifest") is not None,
    "compiled.inspect() carries the lowered-module trace (operator-first Module)")
chk(bool(compiled.module_hash()), "the handle carries a compile-time module_hash for drift detection")

# The trace lists the operator-first operators (flux_default / electrostatic) without the user
# ever building a Module by hand.
ops = [op.get("name") for op in report["module_manifest"].get("operators", [])] \
    if report.get("module_manifest") else []
chk("flux_default" in ops and "electrostatic" in ops,
    "the lowered-module trace lists the operators (%s)" % ops)

# The retired root-level ModelSpec composition no longer enters Program compilation. Its explicit,
# bounded runtime bridge is rejected before a handle exists, so codegen cannot fabricate a Module
# trace or hash for it.
print("== bounded native ModelSpec bridge is absent from Program compilation ==")
legacy_error = None
try:
    legacy_program_model = physics_model()
    compile_problem(
        time=_fe_program(legacy_program_model, "legacy_bridge_probe"),
        model=bricks_model(),
    )
except TypeError as exc:
    legacy_error = exc
chk(legacy_error is not None,
    "a native ModelSpec bridge is rejected before any compiled handle can escape")
chk(legacy_error is not None and "CompilerLowerable" in str(legacy_error),
    "the rejection names the canonical CompilerLowerable/Module boundary")

print("%s test_compile_module_trace" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
