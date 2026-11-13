"""IR6 (G02 lot L, ROM-20): a named body whose elaboration reaches itself is refused.

Before lot L such a recursion ended as a Python ``RecursionError``. One elaboration stack
(``pops._ir.elaboration``) is entered by a GraphFunction while it is decorated, by a
``@pops.solver`` (``pops.surface.solver`` until the cut-over) while it is lowered and by the body of
a ``@pops.scheme`` while it runs; re-entering a definition raises ``RecursiveGraphFunction`` with
the cycle and the Python origin of each call. A plain Python helper that recurses while it builds a
body only unrolls a finite body and is not refused.
"""
from __future__ import annotations

import pytest

import pops
import pops.surface as surface
from pops._ir.refusals import RecursiveGraphFunction
from pops.graph_functions import graph
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.lib.solve import cg
from pops.surface import _solver
from pops.time.variant_types import Float64, Product


def test_a_graph_function_that_decorates_itself_again_is_refused():
    def builder(p, a):
        graph(inputs=Product(x=Float64), output=Float64)(builder)
        return a["x"]

    with pytest.raises(RecursiveGraphFunction) as refused:
        graph(inputs=Product(x=Float64), output=Float64)(builder)
    names = [name for name, _ in refused.value.cycle]
    assert names == [builder.__qualname__, builder.__qualname__]
    assert all(__file__ in origin for _, origin in refused.value.cycle)
    assert "a recursion is written as iterate" in str(refused.value)
    assert refused.value.rule == "IR6" and refused.value.phase == "validate"


def test_a_python_helper_that_unrolls_is_not_refused():
    def power(p, x, n):
        return x if n == 1 else x * power(p, x, n - 1)

    f = graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: power(p, a["x"], 4))
    assert f.identity


def _problem():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    frame = CartesianDomain("ir6", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("ir6", frame=frame)
    state = model.state("u", components=("c0",), sampling="cell_average")
    block = pops.Case("ir6").block("q", model)
    program = pops.Program("ir6")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator("A", domain="state", range_="state", ncomp=1)
    program.set_apply(operator, lambda p, out, x: x - 0.1 * p.laplacian(out, x))
    linear = LinearProblem(operator, temporal.n, nullspace=None,
                           properties=LinearOperatorProperties.symmetric_positive_definite())
    return program, linear


def test_solvers_that_lower_each_other_are_refused_with_the_cycle():
    program, linear = _problem()
    reference = cg(max_iterations=4).with_request(rtol=1e-8, atol=0.0)

    @surface.solver
    def first(problem, region, depth=1):
        _solver.lower_solver(second().with_request(rtol=1e-8, atol=0.0), program, linear)
        return _solver.lower_solver(reference, program, linear)

    @surface.solver
    def second(problem, region, depth=1):
        _solver.lower_solver(first().with_request(rtol=1e-8, atol=0.0), program, linear)
        return None

    with pytest.raises(RecursiveGraphFunction) as refused:
        _solver.lower_solver(first().with_request(rtol=1e-8, atol=0.0), program, linear)
    assert [name for name, _ in refused.value.cycle] == [
        "@pops.solver first", "@pops.solver second", "@pops.solver first"]
    assert all(__file__ in origin for _, origin in refused.value.cycle)


def test_the_same_solver_lowered_twice_in_sequence_is_not_a_cycle():
    program, linear = _problem()
    reference = cg(max_iterations=4).with_request(rtol=1e-8, atol=0.0)
    _solver.lower_solver(reference, program, linear)
    _solver.lower_solver(reference, program, linear)


def test_a_scheme_whose_body_runs_itself_again_is_refused():
    runs = {}

    @surface.scheme
    def again(face, term):
        return runs["scheme"]._run({"face": face, "term": term}, None)

    def realize():
        runs["scheme"] = again()
        return runs["scheme"]._run({"face": object(), "term": object()}, None)

    with pytest.raises(RecursiveGraphFunction) as refused:
        realize()
    assert [name for name, _ in refused.value.cycle] == ["@pops.scheme again"] * 2
    assert all(__file__ in origin for _, origin in refused.value.cycle)
