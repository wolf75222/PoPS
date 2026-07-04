#!/usr/bin/env python3
"""ADC-644 -- the wired GeometricMG preconditioner options participate in the program IR / emit.

A default ``preconditioners.GeometricMG()`` lowers to NO ``precond_options`` IR attr, so the program
IR hash AND the emitted C++ (``GeometricMgPreconditioner()``) are BYTE-IDENTICAL to the pre-644 form.
A configured preconditioner (V-cycle-shape knobs) adds the attr, busts the IR hash, and emits the
explicit ctor ``GeometricMgPreconditioner(nu1, nu2, nbottom, min_coarse, n_vcycles)``.

Source-only: the emit + IR hash are exercised at the authoring/lowering layer (no _pops runtime, no
compile). Runs under pytest AND standalone. Skips (never fakes) if pops is not importable.
"""
import sys

import pytest


def _solve_program(preconditioner=None):
    """A minimal GMRES solve_linear program (GMRES takes the preconditioner ApplyFn slot)."""
    import pops.time as t
    from pops.solvers.krylov import GMRES
    P = t.Program("p")
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - 0.1 * lap

    P.set_apply(A, apply)
    kw = dict(operator=A, rhs=U, method=GMRES(max_iter=200), tol=1e-10, max_iter=200, restart=8)
    if preconditioner is not None:
        kw["preconditioner"] = preconditioner
    phi = P.solve_linear(**kw)
    P.commit("blk", phi)
    return P


def test_default_geometric_mg_precond_leaves_ir_hash_byte_identical():
    from pops.solvers.preconditioners import preconditioners
    # A default GeometricMG() preconditioner and the emitting-only default (identity is not equivalent:
    # it takes a different branch, so we compare the DEFAULT MG to itself and confirm no attr leaks).
    default_a = _solve_program(preconditioners.GeometricMG())
    default_b = _solve_program(preconditioners.GeometricMG())
    assert default_a._ir_hash() == default_b._ir_hash()
    node = default_a._commits["blk"]
    # Omit-when-default: no precond_options attr on the node for a default GeometricMG().
    assert "precond_options" not in node.attrs
    assert node.attrs["preconditioner"] == "geometric_mg"


def test_configured_precond_busts_ir_hash_and_adds_attr():
    from pops.solvers.preconditioners import preconditioners
    default = _solve_program(preconditioners.GeometricMG())
    override = _solve_program(preconditioners.GeometricMG(n_vcycles=3, pre_sweeps=1))
    assert default._ir_hash() != override._ir_hash()
    node = override._commits["blk"]
    assert node.attrs["precond_options"] == {"n_vcycles": 3, "pre_sweeps": 1}


def test_default_precond_emits_historical_ctor():
    default = _solve_program()  # None -> Identity() (unpreconditioned)
    mg_default = _solve_program(_mg())
    src_default = mg_default.emit_cpp_program()
    # The default GeometricMG preconditioner emits the no-arg ctor, byte-identical to pre-644.
    assert "GeometricMgPreconditioner>();" in src_default
    assert "GeometricMgPreconditioner>(2, 2, 50" not in src_default


def test_configured_precond_emits_explicit_ctor():
    override = _solve_program(_mg(n_vcycles=3, pre_sweeps=1, post_sweeps=1, bottom_sweeps=80,
                                  min_coarse=4))
    src = override.emit_cpp_program()
    # nu1, nu2, nbottom, min_coarse, n_vcycles in fixed positional order.
    assert "GeometricMgPreconditioner>(1, 1, 80, 4, 3)" in src


def _mg(**kw):
    from pops.solvers.preconditioners import preconditioners
    return preconditioners.GeometricMG(**kw)


def main():
    try:
        import pops  # noqa: F401
        import pops.time  # noqa: F401
    except Exception as exc:  # pragma: no cover - no built extension here
        print("SKIP  ADC-644 precond IR hash (pops not importable: %s)" % exc)
        return 0
    test_default_geometric_mg_precond_leaves_ir_hash_byte_identical()
    test_configured_precond_busts_ir_hash_and_adds_attr()
    test_default_precond_emits_historical_ctor()
    test_configured_precond_emits_explicit_ctor()
    print("OK  ADC-644 precond options IR hash + emit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
