"""Program/immutable-graph integration for canonical residual operators."""
from __future__ import annotations

import pytest

from pops.model import Module
from pops.problem import Problem
from pops.time import FailRun, Program, RejectAttempt
from pops.time.graph import (
    Commit, ProgramGraph, ProgramValue, ResidualEvaluation, ResidualSolve, StateRead, ValueRef,
)
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
    outcome = program.solve_residual(residual, initial=(u, p), name="newton")
    with pytest.raises(TypeError, match="not iterable"):
        tuple(outcome)
    solution = outcome.consume(action=RejectAttempt(statuses=("iteration_limit",)))
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
        "solve_outcome_component", "solve_outcome_component")
    assert tuple(value.block for value in solution) == (u.block, p.block)
    assert tuple(value.space for value in solution) == (u.space, p.space)
    token = next(value for value in program._values if value.op == "solve_residual")
    consumed = next(value for value in program._values if value.op == "solve_outcome")
    assert consumed.inputs == (token,)
    assert consumed.attrs["action"].to_data()["kind"] == "reject_attempt"
    assert all(value.inputs == (consumed,) for value in solution)
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
    solution = program.solve_residual(
        residual, initial={"fluid::p": p, "fluid::u": u}).consume(action=FailRun())
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
        at=(left_state.next.point, right_state.next.point)).consume(action=FailRun())

    assert tuple(value.block for value in solution) == (left, right)
    assert tuple(value.space for value in solution) == (space, space)
    assert tuple(value.point for value in solution) == (
        left_state.next.point, right_state.next.point)
    program.commit_many({left_state.next: solution[0], right_state.next: solution[1]})


def test_graph_rejects_direct_residual_solve_reads_before_consumed_outcome():
    clock = Clock("macro")
    point = TimePoint(clock)
    operator = _operator("fluid::u")
    source = StateRead(0, {"state": "fluid::u"}, clock, point)
    residual = ResidualEvaluation(1, operator, (ValueRef(0),), clock, point)
    solve = ResidualSolve(
        2, ValueRef(1), (ValueRef(0),), clock, point,
        attrs={"unknown_count": 1})

    direct = ProgramValue(3, "bad", "state", "copy", (ValueRef(2),), clock, point)
    with pytest.raises(ValueError, match="unconsumed solve token"):
        ProgramGraph("bad-direct", (source, residual, solve, direct), clocks=(clock,))

    commit = Commit(4, {"state": "fluid::u"}, ValueRef(2), clock, TimePoint(clock, step=1))
    with pytest.raises(ValueError, match="unconsumed solve token"):
        ProgramGraph("bad-commit", (source, residual, solve, commit), clocks=(clock,))

    outcome = ProgramValue(
        5, "accepted", "solve_outcome", "solve_outcome", (ValueRef(2),), clock, point,
        attrs={"action": FailRun()})
    component = ProgramValue(
        6, "u", "state", "solve_outcome_component", (ValueRef(5),), clock, point,
        attrs={"index": 0})
    graph = ProgramGraph(
        "good-consume", (source, residual, solve, outcome, component), clocks=(clock,))
    assert graph.nodes[-1].to_data()["op"] == "solve_outcome_component"

    missing_action = ProgramValue(
        7, "missing", "solve_outcome", "solve_outcome", (ValueRef(2),), clock, point)
    with pytest.raises(ValueError, match="requires explicit action"):
        ProgramGraph("bad-action", (source, residual, solve, missing_action), clocks=(clock,))
