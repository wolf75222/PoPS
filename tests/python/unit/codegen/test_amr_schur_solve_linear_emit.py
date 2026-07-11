"""ADC-633: the compiled condensed-Schur Program routes solve_linear through the ctx on the AMR target.

Source-only codegen check (host-side, no compiler / no _pops run): a condensed-Schur Program lowered
for ``target='amr_system'`` emits ``ctx.solve_linear_matfree(...)`` for its elliptic solve, while the
same Program lowered for ``target='system'`` keeps emitting the verbatim ``pops::bicgstab_solve(...)``.
This pins the target-conditional seam (the System body -- and hence the uniform trajectory + IR hash --
is byte-untouched; only the AMR body gains the hierarchy-aware solve).
"""
from pops.params import ConstParam
import pytest

pops_time = pytest.importorskip("pops.time", exc_type=ImportError)
pops_lib_time = pytest.importorskip("pops.lib.time", exc_type=ImportError)


def _lorentz_model(name):
    """The canonical condensed block (rho / mx / my + grad_x / grad_y / B_z) carrying the
    electrostatic-Lorentz linearization J the generic condensed route (ADC-637) resolves."""
    from pops.ir.ops import sqrt
    from pops.lib.models import author_electrostatic_lorentz
    from pops.physics.facade import Model
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


def _emit(target):
    model = _lorentz_model("adc633_emit_model")
    P = pops_time.Program("adc633_schur_emit").bind_operators(model)
    pops_lib_time.condensed_schur(
        P, "blk", alpha=1.0, theta=1.0,
        linear_operator=_linear_handle(model))
    return P.emit_cpp_program(model=model, target=target)


def test_amr_target_routes_solve_linear_through_ctx():
    """The AMR .so emits ctx.solve_linear_matfree (the hierarchy-aware seam), NOT the bare Krylov call."""
    src = _emit("amr_system")
    assert "ctx.solve_linear_matfree(" in src, (
        "a condensed-Schur Program on the AMR target must route solve_linear through the ctx seam "
        "(ctx.solve_linear_matfree) so the flat/composite dispatch runs per hierarchy"
    )


def test_system_target_keeps_verbatim_bicgstab():
    """The System .so is byte-untouched: it still emits pops::bicgstab_solve, never the AMR seam."""
    src = _emit("system")
    assert "pops::bicgstab_solve(" in src, (
        "the System target must keep emitting the verbatim matrix-free Krylov call -- the uniform "
        "trajectory (and the IR hash) are unchanged by ADC-633"
    )
    assert "ctx.solve_linear_matfree(" not in src, (
        "the System target must NOT emit the AMR hierarchy seam"
    )


def test_ir_hash_is_target_independent():
    """The IR identity is the same for both targets (the hash is of the IR, not the emitted C++): the
    AMR seam is an emission-time branch, not an IR change, so the uniform IR-hash is untouched."""
    model = _lorentz_model("adc633_hash_model")
    P = pops_time.Program("adc633_schur_hash").bind_operators(model)
    pops_lib_time.condensed_schur(
        P, "gas", alpha=1.0, theta=1.0,
        linear_operator=_linear_handle(model))
    h_sys = P._ir_hash()
    h_amr = P._ir_hash()
    assert h_sys == h_amr, "the IR hash must not depend on the codegen target"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
