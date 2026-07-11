#!/usr/bin/env python3
"""Generic condensed-implicit-solve codegen (ADC-637).

The condensed-implicit pattern authors a source's per-cell block linearization ``J`` (via
``m.local_linear_operator`` / ``m.local_linear_map`` on a coupled momentum subset) and a gradient-linear
elliptic coupling, and the codegen emits the three condensed stages -- the tensor coefficient assembly
``A = I + c*rho*M^{-1}`` (M = I - theta*dt*J), the fused RHS ``-Lap(phi^n) - g*div(M^{-1}(mx,my))`` and
the velocity reconstruction ``v = M^{-1}(v^n - theta*dt*grad phi)`` -- as INLINE ``for_each_cell``
kernels that invert M once per cell with ``pops::detail::block_inverse<2>`` (block_inverse.hpp). No call
into ``coupling/schur/**``; the block inverse is computed from the AUTHORED J, generic in the operator.

This is a SOURCE-ONLY golden of the emitted C++ (no compile, no _pops runtime call): it pins the
block_inverse reduction text (``M_ = I - th_dt*J``, ``block_inverse<2>``, ``A = 1 + cr*Mi_``), that the
Lorentz J entries lower through the shared Expr.to_cpp machinery (``B_z = auxA(i, j, 3)``,
``- th_dt_ * (B_z)`` / ``(-B_z)``), the R2 rho/M split (rho in the outer factor, never in M), and the
block_inverse.hpp include gating. Real engine only; skips (exit 0) if pops is not importable, never
faking. Runs under pytest and as a script.
"""
import sys

try:
    from pops.physics.facade import Model
    from pops.solvers.krylov import BiCGStab
    from pops import time as adctime
    from pops.time.value_metadata import CoeffPolynomial
    from typed_program_support import typed_state
except Exception as exc:  # pops not importable here (no built extension) -> skip, never fake
    print("skip test_condensed_generic_codegen (pops unavailable: %s)" % exc)
    sys.exit(0)


def _lorentz_condensed_program():
    """A rho/mx/my block with the Lorentz linearization J = [[0,B_z],[-B_z,0]] authored on the momentum
    subset and a Poisson coupling, lowered through the generic condensed_* ops -- the codegen instance
    the emitters lower (the PR-2 macro will author this same IR ergonomically)."""
    m = Model("lorentz_condensed")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    m.aux("grad_x")
    m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    # J on the full state; the coupled block is the momentum subset (1, 2). Coefficients are aux-only
    # (B_z), so the block is constant in U|_K -- the eliminable class.
    m.local_linear_map("lorentz_J", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)

    P = adctime.Program("cs_generic")
    temporal = typed_state(P, "blk", state_name="U", model=m)
    U = temporal.n
    P.solve_fields(U)
    coeffs = P._new("condensed_coeffs", "condensed_coeffs", (U,),
                    {"linear_operator": "lorentz_J", "subset": (1, 2),
                     "c": CoeffPolynomial({2: 1.0}),
                     "th_dt": CoeffPolynomial({1: 1.0}), "c_rho": 0},
                    "cs_coeffs", temporal.block)
    phi_n = P.scalar_field("blk.phi_n")
    rhs = P.scalar_field("blk.rhs")
    P._new("scalar_field", "condensed_rhs", (rhs, phi_n, U),
           {"linear_operator": "lorentz_J", "subset": (1, 2),
            "th_dt": CoeffPolynomial({1: 1.0}), "g": CoeffPolynomial({1: 1.0})},
           "cs_rhs", temporal.block)
    rhs2 = P._values[-1]
    A = P.matrix_free_operator("blk.op")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.apply_laplacian_coeff(lap, x, coeffs)
        return -1.0 * lap

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=rhs2, method=BiCGStab(max_iter=50), tol=1e-10, max_iter=50)
    out = P._new("state", "condensed_reconstruct", (U, phi),
                 {"linear_operator": "lorentz_J", "subset": (1, 2),
                  "th_dt": CoeffPolynomial({1: 1.0}), "c_rho": 0},
                 "cs_recon", temporal.block, space=U.space, state_ref=U.state_ref)
    P.commit(temporal.next, out)
    return P, m


def _emit(target="system"):
    P, m = _lorentz_condensed_program()
    return P.emit_cpp_program(model=m, target=target)


def test_coeffs_emit_block_inverse_reduction():
    """The coefficient assembly builds M = I - th_dt*J from the authored J, inverts it with
    block_inverse<2>, and writes A = I + c*rho*M^{-1} into the four tensor fields."""
    src = _emit()
    for frag in ("pops::Real M_[2][2];",
                 "M_[0][0] = pops::Real(1) - th_dt_ * (0.0);",
                 "M_[0][1] = pops::Real(0) - th_dt_ * (B_z);",
                 "M_[1][0] = pops::Real(0) - th_dt_ * ((-B_z));",
                 "M_[1][1] = pops::Real(1) - th_dt_ * (0.0);",
                 "pops::detail::block_inverse<2>(M_, Mi_);",
                 "exA(i, j, 0) = pops::Real(1) + cr * Mi_[0][0];",
                 "eyA(i, j, 0) = pops::Real(1) + cr * Mi_[1][1];",
                 "axyA(i, j, 0) = cr * Mi_[0][1];",
                 "ayxA(i, j, 0) = cr * Mi_[1][0];"):
        assert frag in src, "condensed_coeffs must emit %r\n%s" % (frag, src)
    print("OK  condensed_coeffs emits M = I - th_dt*J, block_inverse<2>, A = I + c*rho*M^-1")


def test_j_entries_lower_through_shared_expr_machinery():
    """The authored Lorentz J references aux('B_z'), lowered by the SAME Expr.to_cpp + _cell_locals
    machinery the model kernels use: B_z is bound from the aux at its canonical component 3."""
    src = _emit()
    assert "const pops::Real B_z = auxA(i, j, 3);" in src, "B_z aux binding missing\n%s" % src
    print("OK  the authored J lowers via the shared Expr machinery (B_z from aux component 3)")


def test_r2_rho_split_out_of_M():
    """R2: rho (a conservative var) enters only the OUTER factor cr = c*rho, never the block M (which
    is J-only). The coeff kernel reads rho from the state and builds cr; M has no stateA read."""
    src = _emit()
    assert "const pops::Real cr = (" in src and ") * rho;" in src, "cr = c*rho missing\n%s" % src
    # M assembly references only th_dt_ and the aux J locals -- never rho or a stateA read.
    coeff = src.split("pops::Real M_[2][2];", 1)[1].split("block_inverse<2>", 1)[0]
    assert "rho" not in coeff and "stateA" not in coeff, "M must not read rho / state (R2)\n%s" % coeff
    print("OK  R2: rho is the outer c*rho factor; M = I - th_dt*J is J-only")


def test_rhs_and_reconstruct_use_block_apply_inverse():
    """The fused RHS flux and the velocity reconstruction apply M^{-1} to a VECTOR with the FACTORED
    ``block_apply_inverse<2>`` (one reciprocal out of the bracket, bit-for-bit the retiring brick's
    LorentzEliminator::apply_Binv), NOT the pre-divided block_inverse entries -- that spelling would
    round differently and drift the trajectory off np.array_equal (ADC-637 PR-2 parity crux). The RHS
    fuses -Lap phi^n with the centered divergence of M^{-1}(m)."""
    src = _emit()
    assert "pops::detail::block_apply_inverse<2>(M_, cond_v_, cond_mv_);" in src, \
        "vector apply must use the factored block_apply_inverse\n%s" % src
    assert "fA(i, j, 0) = cond_fx_;" in src, "flux M^-1 apply write missing\n%s" % src
    assert "rhsA(i, j, 0) = nlA(i, j, 0) - " in src, "fused -Lap - g*div(F) missing\n%s" % src
    assert "stateA(i, j, 1) = rho * nx_;" in src, "mom = rho*v write missing\n%s" % src
    # exactly one block_inverse<2> site (coeffs: A reads the entries directly) and two factored
    # block_apply_inverse<2> sites (flux + reconstruct: the vector apply).
    assert src.count("pops::detail::block_inverse<2>(M_, Mi_);") == 1, \
        "expected 1 block_inverse (coeffs) site\n%s" % src
    assert src.count("pops::detail::block_apply_inverse<2>(M_, cond_v_, cond_mv_);") == 2, \
        "expected 2 block_apply_inverse (flux + reconstruct) sites\n%s" % src
    print("OK  condensed_rhs + condensed_reconstruct apply M^{-1} with the factored block_apply_inverse")


def test_block_inverse_header_included_and_no_schur_tokens():
    """The generated .so includes block_inverse.hpp (only when a condensed op is present) and carries NO
    coupling/schur token -- neither the include path NOR the C++ namespace: a generic-only Program must
    compile without coupling/schur/** (its matrix-free apply lowers the condensed bundle to
    ctx.fill_boundary + the pops::apply_laplacian coefficient floor, and the coefficient halos are
    filled through the ctx seam, as the brick's assemble_schur_coeffs did natively)."""
    src = _emit()
    assert "#include <pops/numerics/linalg/block_inverse.hpp>" in src, "block_inverse include missing"
    for forbidden in ("coupling/schur", "coupling::schur", "LorentzEliminator", "assemble_schur",
                      "SchurOperator", "schur_reconstruct"):
        assert forbidden not in src, "generic condensed path must not name %r\n%s" % (forbidden, src)
    # The generic coefficiented apply: in-halos via the ctx seam, then the SAME apply_laplacian floor
    # the brick's wrapper forwarded to (bit-identical operator arithmetic).
    assert "pops::apply_laplacian(" in src, "generic apply must call the apply_laplacian floor\n%s" % src
    assert "ctx.fill_boundary(const_cast<pops::MultiFab&>(in));" in src, \
        "generic apply must fill the in-halos via the ctx seam\n%s" % src
    # The four coefficient fields get their halos filled after assembly (the brick's eps_bc fill).
    assert src.count("ctx.fill_boundary(*ceps_x") == 1 and src.count("ctx.fill_boundary(*ca_yx") == 1, \
        "condensed_coeffs must fill the coefficient halos\n%s" % src
    print("OK  block_inverse.hpp included; no coupling/schur vocabulary in the generic path")


def test_schur_free_program_omits_block_inverse_header():
    """A Program with no condensed op does NOT include block_inverse.hpp (the include is gated)."""
    P = adctime.Program("plain")
    temporal = typed_state(P, "blk", state_name="U")
    U = temporal.n
    P.commit(temporal.next,
             P.linear_combine("id", 1.0 * U))
    src = P.emit_cpp_program()
    assert "block_inverse.hpp" not in src, "a condensed-free Program must not include block_inverse.hpp"
    print("OK  block_inverse.hpp is gated: absent from a condensed-free Program")


def test_assembly_redirect_present_on_both_targets_and_no_schur():
    """The per-level write/read redirect (ADC-637 section 2) emits ctx.assembly_target /
    ctx.assembly_source on BOTH the System and the AMR target -- IDENTITY at runtime on System / flat
    AMR (the seam returns the field unchanged), per-level on a refined hierarchy. The emitted C++ text
    is the same; the divergence is a runtime property of the two contexts. Neither target names
    coupling/schur (the brick is retired)."""
    for target in ("system", "amr_system"):
        src = _emit(target)
        assert "ctx.assembly_target(" in src, (
            "the condensed emitters must redirect their coefficient / RHS / flux writes through "
            "ctx.assembly_target on target=%r" % target)
        assert "ctx.assembly_source(" in src, (
            "the condensed reconstruction must redirect its potential read through ctx.assembly_source "
            "on target=%r" % target)
        assert "pops::runtime::program::kEpsX" in src, (
            "the redirect must name the AssemblyFieldRole roles on target=%r" % target)
        assert "coupling/schur" not in src and "coupling::schur" not in src, (
            "no coupling/schur include or namespace on target=%r (the brick is retired)" % target)
    # The AMR target routes the elliptic solve through the ctx seam (flat/composite dispatch).
    assert "ctx.solve_linear_matfree(" in _emit("amr_system"), \
        "the AMR target must route solve_linear through the ctx seam"
    print("OK  assembly_target/assembly_source redirect on both targets; no coupling/schur")


def _run():
    fns = [test_coeffs_emit_block_inverse_reduction,
           test_j_entries_lower_through_shared_expr_machinery,
           test_r2_rho_split_out_of_M,
           test_rhs_and_reconstruct_use_block_apply_inverse,
           test_block_inverse_header_included_and_no_schur_tokens,
           test_schur_free_program_omits_block_inverse_header,
           test_assembly_redirect_present_on_both_targets_and_no_schur]
    for fn in fns:
        fn()
    print("PASS test_condensed_generic_codegen (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
    sys.exit(0)
