"""DIRK coefficients, stage equations, failure consumption and native disposition."""
from fractions import Fraction

import pytest
import pops.lib.time as lt
from pops.time import Program, TimePoint
from typed_program_support import state_refs


def test_dirk_order_depends_on_exact_coefficients_and_has_no_polynomial_claim():
    midpoint = lt.IMPLICIT_MIDPOINT_TABLEAU
    renamed = lt.DiagonallyImplicitRungeKuttaTableau([[Fraction(1, 2)]], [1], name="other")
    impostor = lt.DiagonallyImplicitRungeKuttaTableau([[1]], [1], name="implicit_midpoint")
    assert midpoint.certificate == renamed.certificate
    assert midpoint.properties.order == 2
    assert impostor.properties.order == 1
    assert midpoint.properties.stability_status == "unverified"
    assert not hasattr(midpoint.properties, "stability_polynomial")
    with pytest.raises(ValueError, match="lower-triangular"):
        lt.DiagonallyImplicitRungeKuttaTableau([[1, 1], [0, 1]], [0, 1])


def test_dirk_stages_are_consumed_before_one_endpoint_commit():
    from test_time_std_imex_lie_ab import _authoring
    state, _, linear, _ = _authoring("dirk")
    tableau = lt.DiagonallyImplicitRungeKuttaTableau(
        [[Fraction(1, 4)], [Fraction(1, 2), Fraction(1, 4)]], [Fraction(1, 2)] * 2)
    program = lt.DIRK(state, implicit_operator=linear, tableau=tableau)
    assert program.validate()
    nodes = program._values
    solves = [v for v in nodes if v.op == "solve_local_linear"]
    assert len(solves) == 2
    assert len([v for v in nodes if v.op == "solve_outcome"]) == 2
    assert len(program.commits()) == 1
    assert tuple(v.point.time.offset for v in solves) == tuple(
        TimePoint(program.clock, c).offset for c in tableau.c)


def test_dirk_nonlinear_stage_uses_residual_and_explicit_solver():
    from pops.physics._facade import Model
    from pops.solvers.nonlinear import LocalNewton
    model = Model("nonlinear_dirk")
    (u,) = model.conservative_vars("u")
    source = model.source_term("relaxation", [-u * u])
    block, state = state_refs(Program("refs"), "material", model=model)
    with pytest.raises(ValueError, match="explicit implicit_solver"):
        lt.DIRK(block[state], implicit_operator=source)
    program = lt.DIRK(block[state], implicit_operator=source,
                      tableau=lt.IMPLICIT_MIDPOINT_TABLEAU,
                      implicit_solver=LocalNewton(tolerance=1e-12, max_iterations=25,
                                                  finite_difference_step=1e-7))
    assert program.validate()
    solve = next(v for v in program._values if v.op == "solve_local_nonlinear")
    assert any(v.op == "source" for v in solve.attrs["residual_block"])
    consume = next(v for v in program._serialize()["nodes"] if v["op"] == "solve_outcome")
    assert consume["attrs"]["action"]["kind"] == "fail_run"
