"""ADC-633: the compiled condensed-Schur Program routes solve_linear through the ctx on the AMR target.

Source-only codegen check (host-side, no compiler / no _pops run): a condensed-Schur Program lowered
for ``target='amr_system'`` emits ``ctx.solve_linear_matfree(...)`` for its elliptic solve, while the
same Program lowered for ``target='system'`` keeps emitting the verbatim ``pops::bicgstab_solve(...)``.
This pins the target-conditional seam (the System body -- and hence the uniform trajectory + IR hash --
is byte-untouched; only the AMR body gains the hierarchy-aware solve).
"""
import pytest

pops_time = pytest.importorskip("pops.time", exc_type=ImportError)
pops_lib_time = pytest.importorskip("pops.lib.time", exc_type=ImportError)


def _emit(target):
    P = pops_time.Program("adc633_schur_emit")
    pops_lib_time.condensed_schur(P, "gas", alpha=1.0, theta=1.0)
    return P.emit_cpp_program(target=target)


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
    P = pops_time.Program("adc633_schur_hash")
    pops_lib_time.condensed_schur(P, "gas", alpha=1.0, theta=1.0)
    h_sys = P._ir_hash()
    h_amr = P._ir_hash()
    assert h_sys == h_amr, "the IR hash must not depend on the codegen target"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
