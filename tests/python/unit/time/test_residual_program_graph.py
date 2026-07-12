"""Program/immutable-graph integration for canonical residual operators."""
from __future__ import annotations

import pytest

from pops.model import Module
from pops.problem import Problem
from pops.time import Program
from pops.time.graph import ProgramGraph, ResidualEvaluation, ResidualSolve, StateRead, ValueRef
from pops.time.points import Clock, TimePoint
from pops.time.residual import (
    Dt, EquationSpace, FiniteDifferenceJVP, IdentityTerm, PreconditionerContract,
    PreconditionerDomain, ResidualOperator, UnknownSpace,
)


def _operator(*components):
    equations = tuple(
        "eq::%d_%s" % (index, item.split("::")[-1])
        for index, item in enumerate(components))
    return ResidualOperator(
        "implicit",
        UnknownSpace("Y", components),
        EquationSpace("F", equations),
        tuple(IdentityTerm(equation, unknown)
              for equation, unknown in zip(equations, components, strict=True)),
        Dt(),
        FiniteDifferenceJVP(),
    )


def test_residual_product_and_solve_are_exact_immutable_graph_references():
    program = Program("residual-step")
    u = program.scalar_field("u")
    p = program.scalar_field("p")
    operator = _operator("fluid::u", "fluid::p")

    residual = program.residual(
        operator, {"fluid::p": p, "fluid::u": u}, name="F")
    solution = program.solve_residual(residual, initial=(u, p), name="newton")
    before = program._ir_hash()

    graph = program.to_graph()
    evaluation = next(node for node in graph.nodes if type(node) is ResidualEvaluation)
    solve = next(node for node in graph.nodes if type(node) is ResidualSolve)

    assert evaluation.unknowns == (ValueRef(u.id), ValueRef(p.id))
    assert evaluation.operator.to_data() == operator.to_data()
    assert solve.residual == ValueRef(residual.id)
    assert solve.initial == (ValueRef(u.id), ValueRef(p.id))
    assert len(solution) == 2
    assert tuple(value.op for value in solution) == (
        "residual_solution_component", "residual_solution_component")
    assert tuple(value.block for value in solution) == (u.block, p.block)
    assert tuple(value.space for value in solution) == (u.space, p.space)
    token = next(value for value in program._values if value.op == "solve_residual")
    assert all(value.inputs == (token,) for value in solution)
    assert program._ir_hash() == before
    assert graph.graph_hash == program.to_graph().graph_hash
    assert graph.to_data() == program.to_graph().to_data()
    assert program._rebuild(lambda _value: True)._ir_hash() == program._ir_hash()


def test_residual_rejects_product_and_initial_space_mismatches_before_compile():
    program = Program("bad-residual")
    u = program.scalar_field("u")
    p = program.scalar_field("p")
    q = program.scalar_field("q", ncomp=2)
    operator = _operator("fluid::u", "fluid::p")
    before = program._ir_hash()

    with pytest.raises(ValueError, match="mapping keys must exactly match"):
        program.residual(operator, {"fluid::u": u, "fluid::q": q})
    with pytest.raises(ValueError, match="expects 2 unknown component"):
        program.residual(operator, (u,))
    assert program._ir_hash() == before

    residual = program.residual(operator, (u, p))
    with pytest.raises(ValueError, match="initial product arity"):
        program.solve_residual(residual, initial=(u,))
    solution = program.solve_residual(residual, initial={"fluid::p": p, "fluid::u": u})
    assert tuple(value.space for value in solution) == (u.space, p.space)
    with pytest.raises(ValueError, match="mapping keys must exactly match"):
        program.solve_residual(residual, initial={"fluid::u": u, "fluid::q": q})
    with pytest.raises(TypeError, match="solver must be an immutable typed descriptor"):
        program.solve_residual(residual, initial=(u, p), solver="newton")
    incompatible = PreconditionerContract(
        "bad_pc",
        PreconditionerDomain(
            EquationSpace("wrong_F", ("wrong::u", "wrong::p")),
            UnknownSpace("wrong_Y", ("wrong::u", "wrong::p")),
        ),
    )
    with pytest.raises(ValueError, match="preconditioner.*space"):
        program.solve_residual(residual, initial=(u, p), preconditioner=incompatible)


def test_residual_graph_nodes_keep_clock_and_region_validation_rules():
    slow = Clock("slow")
    fast = Clock("fast")
    operator = _operator("fluid::u")
    source = StateRead(0, {"state": "fluid::u"}, slow, TimePoint(slow))
    residual = ResidualEvaluation(
        1, operator, (ValueRef(0),), fast, TimePoint(fast))
    with pytest.raises(ValueError, match="cross-clock read slow -> fast"):
        ProgramGraph("bad-clock", (source, residual), clocks=(slow, fast))

    bad_operator = ResidualEvaluation(
        2, {"kind": "not_residual_operator"}, (ValueRef(0),), slow, TimePoint(slow))
    with pytest.raises(ValueError, match="operator must be a residual_operator"):
        ProgramGraph("bad-operator", (source, bad_operator), clocks=(slow,))

    mismatched = ResidualEvaluation(
        3, operator, (ValueRef(0), ValueRef(0)), slow, TimePoint(slow))
    with pytest.raises(ValueError, match="unknown product arity"):
        ProgramGraph("bad-arity", (source, mismatched), clocks=(slow,))


def test_residual_solution_preserves_each_unknown_block_and_space():
    model = Module("coupled")
    space = model.state_space("U", ("u",))
    state = model.state_handle(space)
    problem = Problem("case")
    left = problem.add_block("left", model)
    right = problem.add_block("right", model)
    program = Program("coupled-residual")
    left_state = program.state(left, state)
    right_state = program.state(right, state)
    u_left = left_state.n
    u_right = right_state.n
    operator = _operator("left::U", "right::U")

    residual = program.residual(operator, (u_left, u_right))
    solution = program.solve_residual(
        residual, initial=(u_left, u_right),
        at=(left_state.next.point, right_state.next.point))

    assert tuple(value.block for value in solution) == (left, right)
    assert tuple(value.space for value in solution) == (space, space)
    assert tuple(value.point for value in solution) == (
        left_state.next.point, right_state.next.point)
    program.commit_many({left_state.next: solution[0], right_state.next: solution[1]})
