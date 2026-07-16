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

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)

_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip("test_compile_module_trace: %s" % _native_missing)


try:
    import pops
    import pops.lib.time as libtime
    from pops.codegen._compile_drivers import compile_problem
    from pops.runtime._engine_descriptors import (
        BackgroundDensity, FluidState, IsothermalFlux, Model as NativeBrickModel, NoSource,
    )
    from tests.python.integration._final_field_program import (
        resolve_periodic_field_program,
        scalar_advection_field_model,
    )
    from tests.python.support.typed_program import program_states
except Exception as exc:  # noqa: BLE001
    require_native_or_skip("test_compile_module_trace imports unavailable: %s" % exc)

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
compiled_model = physics_model()
resolved = resolve_periodic_field_program(
    compiled_model,
    lambda state, rate, _fields: libtime.ForwardEuler(state, rate=rate),
    name="module-trace",
    block_name="ions",
    target="system",
    n=8,
)
compiled = pops.compile(resolved)

print("== ADC-557: one lowering, module trace on the handle (final Model) ==")
report = compiled.inspect().to_dict()
expected_module_hash = compiled_model.module.module_hash()
compile_frozen_manifest = compiled.module_manifest
chk(compile_frozen_manifest is not None,
    "the compiled handle retains its immutable compile-frozen module manifest")
manifest_data = {} if compile_frozen_manifest is None else compile_frozen_manifest.to_dict()
ops = [op.get("name") for op in manifest_data.get("operators", [])]
transport_alias = manifest_data.get("operator_aliases", {}).get("transport", {})
chk(
    "flux_default" in ops
    and "electrostatic" in ops
    and transport_alias.get("target") == "flux_default"
    and transport_alias.get("handle", {}).get("registered_operator_name") == "flux_default",
    "the trace preserves authored flux 'transport' as the authenticated alias of the canonical "
    "flux_default native route (operators=%s, alias=%s)" % (ops, transport_alias),
)
module_hashes = [
    dict(block.model.definition_identity).get("module_hash")
    for block in compiled.blocks
]
chk(module_hashes and all(value == expected_module_hash for value in module_hashes),
    "every qualified block carries its authenticated compile-time module hash")
cache_hashes = report.get("options", {}).get("cache_key", {}).get("model_hashes", [])
chk(bool(cache_hashes), "compiled.inspect() exposes the model hashes used by the cache key")

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
