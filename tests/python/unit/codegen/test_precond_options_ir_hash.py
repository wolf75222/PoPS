#!/usr/bin/env python3
"""Prepared GeometricMG options are complete provider-owned IR and native input.

Defaults are canonicalized by the provider before entering IR.  The compiler therefore has no
provider-name branch and no omit-when-default path; configured options still change identity and
the explicit native constructor.

Source-only: the emit + IR hash are exercised at the authoring/lowering layer (no _pops runtime, no
compile). Runs under pytest AND standalone. Skips (never fakes) if pops is not importable.
"""
from tests.python.support.requirements import require_native_or_skip
from pops.codegen.program_codegen import emit_cpp_program
import sys

import pytest

from typed_program_support import typed_state


def _solve_program(preconditioner=None):
    """A minimal typed LinearProblem solved by GMRES with an optional preconditioner."""
    import pops.time as t
    from pops.linalg import LinearProblem
    from pops.solvers.krylov import GMRES
    P = t.Program("p")
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - 0.1 * lap

    P.set_apply(A, apply)
    endpoint = typed_state(P, "blk", state_name="U").next
    rhs = P.value("rhs", U, at=endpoint.point)
    solver = GMRES(
        max_iter=200, rel_tol=1e-10, restart=8, preconditioner=preconditioner)
    phi = P.solve(
        LinearProblem(A, rhs, nullspace=None), solver=solver,
    ).consume(action=t.FailRun())
    P.commit(endpoint, phi)
    return P


def test_default_geometric_mg_precond_has_stable_provider_identity_and_canonical_options():
    from pops.solvers.preconditioners import preconditioners
    default_a = _solve_program(preconditioners.GeometricMG())
    default_b = _solve_program(preconditioners.GeometricMG())
    assert default_a._ir_hash() == default_b._ir_hash()
    node = next(value for value in default_a._values if value.op == "solve_linear")
    assert node.attrs["preconditioner_provider"]["provider_id"] == (
        "pops.preconditioner.geometric-mg"
    )
    assert node.attrs["preconditioner_options"] == {
        "pre_sweeps": 2,
        "post_sweeps": 2,
        "bottom_sweeps": 50,
        "min_coarse": 2,
        "n_vcycles": 1,
    }


def test_configured_precond_busts_ir_hash_and_adds_attr():
    from pops.solvers.preconditioners import preconditioners
    default = _solve_program(preconditioners.GeometricMG())
    override = _solve_program(preconditioners.GeometricMG(n_vcycles=3, pre_sweeps=1))
    assert default._ir_hash() != override._ir_hash()
    node = next(value for value in override._values if value.op == "solve_linear")
    assert node.attrs["preconditioner_options"] == {
        "pre_sweeps": 1,
        "post_sweeps": 2,
        "bottom_sweeps": 50,
        "min_coarse": 2,
        "n_vcycles": 3,
    }


def test_default_precond_emits_explicit_provider_defaults():
    mg_default = _solve_program(_mg())
    src_default = emit_cpp_program(mg_default)
    assert "GeometricMgPreconditioner>(2, 2, 50, 2, 1)" in src_default


def test_configured_precond_emits_explicit_ctor():
    override = _solve_program(_mg(n_vcycles=3, pre_sweeps=1, post_sweeps=1, bottom_sweeps=80,
                                  min_coarse=4))
    src = emit_cpp_program(override)
    # nu1, nu2, nbottom, min_coarse, n_vcycles in fixed positional order.
    assert "GeometricMgPreconditioner>(1, 1, 80, 4, 3)" in src


def test_partial_precond_options_fill_defaults_from_the_shared_schema():
    src = emit_cpp_program(_solve_program(_mg(bottom_sweeps=1)))
    assert "GeometricMgPreconditioner>(2, 2, 1, 2, 1)" in src


@pytest.mark.parametrize(
    "name",
    ["n_vcycles", "pre_sweeps", "post_sweeps", "bottom_sweeps", "min_coarse"],
)
def test_codegen_rejects_forged_preconditioner_integer_overflow(name):
    program = _solve_program(_mg())
    solve = next(value for value in program._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    attrs["preconditioner_options"] = {name: 1 << 31}
    object.__setattr__(solve, "attrs", attrs)

    with pytest.raises(ValueError, match=name):
        emit_cpp_program(program)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("pre_sweeps", -1),
        ("post_sweeps", -1),
        ("bottom_sweeps", 0),
        ("min_coarse", 0),
        ("n_vcycles", 0),
    ],
)
def test_codegen_rejects_forged_preconditioner_below_native_minimum(name, bad):
    program = _solve_program(_mg())
    solve = next(value for value in program._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    attrs["preconditioner_options"] = {name: bad}
    object.__setattr__(solve, "attrs", attrs)

    with pytest.raises(ValueError, match=name):
        emit_cpp_program(program)


def test_codegen_rejects_forged_preconditioner_bool_without_int_coercion():
    program = _solve_program(_mg())
    solve = next(value for value in program._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    attrs["preconditioner_options"] = {"pre_sweeps": True}
    object.__setattr__(solve, "attrs", attrs)

    with pytest.raises(TypeError, match="pre_sweeps"):
        emit_cpp_program(program)


def _mg(**kw):
    from pops.solvers.preconditioners import preconditioners
    return preconditioners.GeometricMG(**kw)


def main():
    try:
        import pops  # noqa: F401
        import pops.time  # noqa: F401
    except Exception as exc:  # pragma: no cover - no built extension here
        require_native_or_skip('SKIP  ADC-644 precond IR hash (pops not importable: %s)' % exc)
        return 0
    test_default_geometric_mg_precond_has_stable_provider_identity_and_canonical_options()
    test_configured_precond_busts_ir_hash_and_adds_attr()
    test_default_precond_emits_explicit_provider_defaults()
    test_configured_precond_emits_explicit_ctor()
    print("OK  ADC-644 precond options IR hash + emit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
