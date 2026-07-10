#!/usr/bin/env python3
"""Generic condensed-implicit authoring ops + compile-time refusals (ADC-637).

``P.condensed_coeffs`` / ``P.condensed_rhs`` / ``P.condensed_reconstruct`` author the three stages of
the generic condensed-implicit solve, each carrying an authored linear operator J (m.local_linear_map)
and a coupled momentum subset. They lower to the inline block_inverse<2> emitters (program_emit_condensed)
parallel to the P.schur_* ops.

This test pins the authoring surface + the design-section-5 refusals (pure Python, no compile): the
builders record + validate their operands, produce the right vtype, and serialize; a subset that is not
distinct / not a tuple / whose size differs from the native spatial dimension (the subset IS the
velocity vector eliminated against grad(phi)/div(F); dimension=2, the ADC-294 core invariant)
raises with a precise message; a non-operator handle raises; a cons/prim-dependent J is refused UPSTREAM
at m.local_linear_map registration (the block-local-linearization contract). Real engine only; skips
(exit 0) if pops is unavailable, never faking. Runs under pytest and as a script.
"""
import sys

try:
    import pytest
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_condensed_authoring (pops unavailable: %s)" % exc)
    sys.exit(0)


def _model():
    m = Model("lorentz_condensed")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    m.aux("grad_x")
    m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    jh = m.local_linear_map("lorentz_J", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    return m, rho, jh


def test_builders_record_the_condensed_ops():
    """The three builders produce the condensed_coeffs / condensed_rhs / condensed_reconstruct ops with
    the authored operator name + the coupled subset, and the program validates + hashes."""
    m, _, jh = _model()
    P = adctime.Program("cs").bind_operators(m)
    U = P.state("blk")
    P.solve_fields(U)
    coeffs = P.condensed_coeffs(state=U, linear_operator=jh, subset=(1, 2), c=1.0 * P.dt * P.dt,
                                th_dt=1.0 * P.dt, c_rho=0)
    assert coeffs.vtype == "condensed_coeffs"
    assert coeffs.attrs["linear_operator"] == "lorentz_J" and coeffs.attrs["subset"] == (1, 2)
    phi_n = P.scalar_field("blk.phi_n")
    rhs = P.scalar_field("blk.rhs")
    r = P.condensed_rhs(out=rhs, phi_n=phi_n, state=U, linear_operator=jh, subset=(1, 2),
                        th_dt=1.0 * P.dt, g=1.0 * P.dt)
    assert r.op == "condensed_rhs" and r.vtype == "scalar_field"
    recon = P.condensed_reconstruct(state=U, phi=phi_n, linear_operator=jh, subset=(1, 2),
                                    th_dt=1.0 * P.dt, c_rho=0)
    assert recon.vtype == "state" and recon.attrs["subset"] == (1, 2)
    P.commit(P.state("U", block="blk").next, recon)
    assert P.validate() is True and P._ir_hash()
    print("OK  condensed_coeffs/rhs/reconstruct record + validate + hash")


def test_subset_size_must_equal_the_spatial_dimension():
    """The subset is the spatial velocity block, so its size must equal the native dimension (2,
    the ADC-294 core invariant) on EVERY condensed op -- a 3-component subset in a 2D engine is
    ill-posed, not unimplemented (ValueError, never NotImplementedError)."""
    m, _, jh = _model()
    P = adctime.Program("cs").bind_operators(m)
    U = P.state("blk")
    with pytest.raises(ValueError, match="spatial velocity block"):
        P.condensed_coeffs(state=U, linear_operator=jh, subset=(0, 1, 2), c=1.0, th_dt=1.0)
    print("OK  condensed_coeffs refuses a size-3 subset via the spatial-dimension contract")


def test_no_dense_capacity_bound_only_the_dimension_contract():
    """There is NO dense-inverse capacity bound (block_inverse<N>/mat_inverse<N> are unbounded in
    N): a size-9 subset is refused by the SAME spatial-dimension contract as size 3, and the
    message must not invent a capacity."""
    m, _, jh = _model()
    P = adctime.Program("cs").bind_operators(m)
    U = P.state("blk")
    big = tuple(range(9))
    with pytest.raises(ValueError, match="dimension=2") as excinfo:
        P.condensed_reconstruct(state=U, phi=P.scalar_field("p"), linear_operator=jh, subset=big,
                                th_dt=1.0)
    assert "bound" not in str(excinfo.value)
    print("OK  size-9 refused by the dimension contract, no invented capacity bound")


def test_subset_must_be_distinct_nonnegative_ints():
    """A subset with a repeated / negative / non-int component raises a precise ValueError."""
    m, _, jh = _model()
    P = adctime.Program("cs").bind_operators(m)
    U = P.state("blk")
    with pytest.raises(ValueError, match="distinct"):
        P.condensed_reconstruct(state=U, phi=P.scalar_field("p"), linear_operator=jh, subset=(1, 1),
                                th_dt=1.0)
    with pytest.raises(ValueError, match="non-negative ints"):
        P.condensed_reconstruct(state=U, phi=P.scalar_field("p2"), linear_operator=jh, subset=(1, -1),
                                th_dt=1.0)
    print("OK  a non-distinct / negative subset is refused")


def test_non_operator_handle_is_refused():
    """linear_operator must be an authored operator (handle or name), not an arbitrary object."""
    m, _, _ = _model()
    P = adctime.Program("cs").bind_operators(m)
    U = P.state("blk")
    with pytest.raises(TypeError, match="OperatorHandle"):
        P.condensed_coeffs(state=U, linear_operator=object(), subset=(1, 2), c=1.0, th_dt=1.0)
    print("OK  a non-operator linear_operator is refused")


def test_scalar_coeffs_and_c_rho_are_validated():
    """The coefficients must be numbers or dt-polynomials, c_rho a non-negative int."""
    m, _, jh = _model()
    P = adctime.Program("cs").bind_operators(m)
    U = P.state("blk")
    with pytest.raises(ValueError, match=r"exact scalar or a dt-polynomial"):
        P.condensed_coeffs(state=U, linear_operator=jh, subset=(1, 2), c="not-a-number", th_dt=1.0)
    with pytest.raises(ValueError, match="c_rho"):
        P.condensed_coeffs(state=U, linear_operator=jh, subset=(1, 2), c=1.0, th_dt=1.0, c_rho=-1)
    print("OK  scalar coefficients + c_rho are validated")


def test_cons_or_prim_dependent_J_refused_upstream():
    """The block-local-linearization contract: a J coefficient depending on a conservative / primitive
    variable is refused at m.local_linear_map registration with an actionable message (so the condensed
    ops only ever reference a validated eliminable operator)."""
    m = Model("bad")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("B_z")
    with pytest.raises(ValueError,
                       match="must not depend on conservative or primitive variables"):
        m.local_linear_map("badJ", [[0.0, 0.0, 0.0], [0.0, 0.0, rho], [0.0, -rho, 0.0]])
    print("OK  a cons/prim-dependent J is refused upstream at registration")


def _run():
    fns = [test_builders_record_the_condensed_ops,
           test_subset_size_must_equal_the_spatial_dimension,
           test_no_dense_capacity_bound_only_the_dimension_contract,
           test_subset_must_be_distinct_nonnegative_ints,
           test_non_operator_handle_is_refused,
           test_scalar_coeffs_and_c_rho_are_validated,
           test_cons_or_prim_dependent_J_refused_upstream]
    for fn in fns:
        fn()
    print("PASS test_condensed_authoring (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
    sys.exit(0)
