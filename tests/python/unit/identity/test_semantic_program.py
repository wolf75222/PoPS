from __future__ import annotations

from fractions import Fraction

import pops.lib.time as libtime
from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.physics._facade import Model
from pops.problem import Case
from pops.time import Program, StagePoint, TimePoint


def _references():
    model = Model("advection")
    model.conservative_vars("u")
    rate = model.rate("explicit", flux=False, sources=())
    block = Case(name="case").block("tracer", model)
    state = next(
        item for item in model.declaration_index().records() if item.kind == "state")
    return block[state], rate


def _stage(program, rate, state, *, at):
    return program.value("stage-rate", rate(state), at=at)


def test_manual_and_library_ssprk2_have_one_semantic_identity():
    state_ref, rate = _references()
    library = libtime.SSPRK2(state_ref, rate=rate)

    manual = Program("my-presentation-name")
    state = manual.state(state_ref)
    u0 = state.n
    point0 = StagePoint("stage-zero", {"main": TimePoint(manual.clock, 0)})
    point1 = StagePoint("stage-one", {"main": TimePoint(manual.clock, 1)})
    k0 = _stage(manual, rate, u0, at=point0)
    u1 = manual.value(
        "predictor-display-name", u0 + manual.dt * k0, at=point1)
    k1 = _stage(manual, rate, u1, at=point1)
    result = manual.value(
        "corrector-display-name",
        u0 + Fraction(1, 2) * manual.dt * k0 + Fraction(1, 2) * manual.dt * k1,
        at=state.next.point,
    )
    manual.commit(state.next, result)

    assert semantic_identity_of(program=library) == semantic_identity_of(program=manual)
    assert program_semantic_data(library) == program_semantic_data(manual)


def test_program_and_node_presentation_names_do_not_enter_semantic_data():
    state_ref, _ = _references()
    left = Program("left")
    left_state = left.state(state_ref)
    left.commit(
        left_state.next,
        left.value("left-node", left_state.n, at=left_state.next.point),
    )

    right = Program("right")
    right_state = right.state(state_ref)
    right.commit(
        right_state.next,
        right.value("right-node", right_state.n, at=right_state.next.point),
    )

    assert left._ir_hash() != right._ir_hash()
    assert semantic_identity_of(program=left) == semantic_identity_of(program=right)
