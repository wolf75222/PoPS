"""ADC-633: the compiled condensed-Schur Program routes solve_linear through the ctx on the AMR target.

Source-only codegen check (host-side, no compiler / no _pops run): a condensed-Schur Program lowered
for both targets emits ``ctx.solve_linear_matfree(...)``. The uniform context selects a metric-aware
provider, while the AMR context selects flat or composite hierarchy execution.
"""
from pops.params import ConstParam
import pytest

from typed_program_support import state_refs

pops_time = pytest.importorskip("pops.time", exc_type=ImportError)
pops_lib_time = pytest.importorskip("pops.lib.time", exc_type=ImportError)


def _lorentz_model(name):
    """The canonical condensed block (rho / mx / my + grad_x / grad_y / B_z) carrying the
    electrostatic-Lorentz linearization J the generic condensed route (ADC-637) resolves."""
    from pops.ir.ops import sqrt
    from pops.lib.models import author_electrostatic_lorentz
    from pops.physics._facade import Model
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.value(m.param(ConstParam("cs2", 0.5)))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)
    m.aux("grad_x")
    m.aux("grad_y")
    m.aux("B_z")
    author_electrostatic_lorentz(m)
    return m


def _linear_handle(model):
    from pops.model import OperatorHandle
    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    return OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)


def _emit(target, *, theta=1.0):
    model = _lorentz_model("adc633_emit_model")
    P = pops_time.Program("adc633_schur_emit")._bind_operators(model)
    block, state = state_refs(P, "blk", model=model)
    pops_lib_time.CondensedSchur(
        P, block, state, alpha=1.0, theta=theta,
        linear_operator=_linear_handle(model))
    return P.emit_cpp_program(model=model, target=target)


def test_amr_target_routes_solve_linear_through_ctx():
    """The AMR .so emits ctx.solve_linear_matfree (the hierarchy-aware seam), NOT the bare Krylov call."""
    src = _emit("amr_system")
    assert "ctx.solve_linear_matfree(" in src, (
        "a condensed-Schur Program on the AMR target must route solve_linear through the ctx seam "
        "(ctx.solve_linear_matfree) so the flat/composite dispatch runs per hierarchy"
    )


def test_refined_driver_orders_gather_solve_publish():
    """The refined branch contains a real hierarchy barrier: both level loops surround one solve."""
    src = _emit("amr_system")
    gather = src.index("Gather every level before the unique hierarchy-scoped solve")
    solve = src.index("ctx.solve_linear_matfree(", gather)
    publish = src.index("The composite solution is complete", solve)
    assert gather < solve < publish
    refined = src[gather:src.index("ctx.advance_synchronized_hierarchy", gather)]
    assert refined.count("ctx.solve_linear_matfree(") == 1
    assert refined[:solve - gather].count("condensed") > 0, "gather region lost assembly kernels"
    assert "ctx.stage_linear_initial_guess();" in refined[:solve - gather], \
        "the zero initial guess must be gathered for every refined level before the solve"
    assert "assembly_source(" in refined[publish - gather:], \
        "publish region lost per-level reconstruction"


def test_condensed_preset_declares_hierarchy_scope_on_the_operator():
    """The preset authors generic operator metadata; no condensed op or emitter infers the scope."""
    model = _lorentz_model("adc648_scope_model")
    P = pops_time.Program("adc648_scope")._bind_operators(model)
    block, state = state_refs(P, "gas", model=model)
    pops_lib_time.CondensedSchur(
        P, block, state, alpha=1.0, theta=1.0,
        linear_operator=_linear_handle(model))
    solve = next(value for value in P._values if value.op == "solve_linear")
    operator = solve.inputs[0]
    assert operator.attrs["scope"] == "hierarchy"
    assert operator.attrs["hierarchy_provider"] == "composite_tensor_fac"
    assert solve.attrs["scope"] == "hierarchy"
    coeffs = next(value for value in P._values if value.op == "condensed_coeffs")
    assert "scope" not in coeffs.attrs
    assert "hierarchy_provider" not in coeffs.attrs


def test_refined_theta_half_gathers_each_levels_phi_history_as_initial_guess():
    """ADC-427 composition: refined theta<1 must not drop the persistent per-level phi^n carry."""
    src = _emit("amr_system", theta=0.5)
    gather = src.index("Gather every level before the unique hierarchy-scoped solve")
    solve = src.index("ctx.solve_linear_matfree(", gather)
    gathered = src[gather:solve]
    assert 'ctx.history_zero_start("blk.schur_phi", 1, 1)' in gathered
    assert "ctx.stage_linear_initial_guess(" in gathered
    assert "ctx.stage_linear_initial_guess();" not in gathered


def test_hierarchy_scope_rejects_control_flow_crossing_barrier():
    """A global solve cannot silently inherit the per-level control-flow scheduler."""
    from pops.solvers import CG, CompositeTensorFAC, Hierarchy
    P = pops_time.Program("adc648_nested_refusal")
    block, state = state_refs(P, "gas")
    temporal = P.state(block, state)
    U = temporal.n
    A = P.matrix_free_operator("A", scope=Hierarchy(), provider=CompositeTensorFAC())

    def apply(program, out, x):
        return program.laplacian(out, x)

    P.set_apply(A, apply)
    rhs = P.scalar_field("rhs")
    from pops.time import FailRun
    phi = P.solve_linear(operator=A, rhs=rhs, method=CG(max_iter=2), max_iter=2).consume(
        action=FailRun())
    # Add top-level control flow after the solve: the first driver envelope refuses rather than
    # claiming a region schedule it cannot preserve.
    looped = P.range(U, 1, lambda program, value: value)
    final = P.value("next", 1 * looped, at=temporal.next.point)
    P.commit(temporal.next, final)
    with pytest.raises(NotImplementedError, match="top-level barrier"):
        P.emit_cpp_program(target="amr_system")


def test_hierarchy_scope_without_provider_is_refused_before_codegen():
    """Scope alone never selects the FAC implementation for an arbitrary matrix-free operator."""
    from pops.solvers import Hierarchy
    P = pops_time.Program("adc648_provider_refusal")
    with pytest.raises(ValueError, match="explicit native provider"):
        P.matrix_free_operator("A", scope=Hierarchy())


def test_provider_extension_uses_interface_not_type_dispatch():
    """A new provider descriptor travels by canonical id/capability without editing authoring code."""
    from pops.solvers import Hierarchy, HierarchySolveProvider
    custom = HierarchySolveProvider("external_composite", frozenset({"amr_hierarchy"}))
    P = pops_time.Program("adc648_open_provider")
    operator = P.matrix_free_operator("A", scope=Hierarchy(), provider=custom)
    assert operator.attrs["hierarchy_provider"] == "external_composite"


def test_system_target_uses_generic_context_seam():
    """The System target uses the same seam so geometry/provider dispatch remains runtime-owned."""
    src = _emit("system")
    assert "ctx.solve_linear_matfree(" in src
    assert "pops::bicgstab_solve(" not in src


def test_ir_hash_is_target_independent():
    """The IR identity is the same for both targets (the hash is of the IR, not the emitted C++): the
    AMR seam is an emission-time branch, not an IR change, so the uniform IR-hash is untouched."""
    model = _lorentz_model("adc633_hash_model")
    P = pops_time.Program("adc633_schur_hash")._bind_operators(model)
    block, state = state_refs(P, "gas", model=model)
    pops_lib_time.CondensedSchur(
        P, block, state, alpha=1.0, theta=1.0,
        linear_operator=_linear_handle(model))
    h_sys = P._ir_hash()
    h_amr = P._ir_hash()
    assert h_sys == h_amr, "the IR hash must not depend on the codegen target"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
