from __future__ import annotations

from fractions import Fraction

import pops.lib.time as libtime
from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.model import Module
from pops.problem import Problem
from pops.time import Program


def _references():
    module = Module("advection")
    state = module.state_space("U", ("u",))
    block = Problem(name="case").add_block("tracer", module)
    return block, module.state_handle(state)


def _stage(program, state):
    return program._rhs_legacy(
        state=state, fields=program.solve_fields(state), flux=True, sources=["default"])


def test_manual_and_library_ssprk2_have_one_semantic_identity():
    block, state_handle = _references()
    library = libtime.SSPRK2(block, state_handle)

    manual = Program("my-presentation-name")
    state = manual.state(block, state_handle)
    u0 = state.n
    k0 = _stage(manual, u0)
    u1 = manual.linear_combine("predictor-display-name", u0 + manual.dt * k0)
    k1 = _stage(manual, u1)
    result = manual.linear_combine(
        "corrector-display-name",
        Fraction(1, 2) * u0 + Fraction(1, 2) * (u1 + manual.dt * k1),
    )
    manual.commit(state.next, result)

    assert semantic_identity_of(program=library) == semantic_identity_of(program=manual)
    assert program_semantic_data(library) == program_semantic_data(manual)


def test_program_and_node_presentation_names_do_not_enter_semantic_data():
    block, state_handle = _references()
    left = Program("left")
    left_state = left.state(block, state_handle)
    left.commit(left_state.next, left.linear_combine("left-node", left_state.n))

    right = Program("right")
    right_state = right.state(block, state_handle)
    right.commit(right_state.next, right.linear_combine("right-node", right_state.n))

    assert left._ir_hash() != right._ir_hash()
    assert semantic_identity_of(program=left) == semantic_identity_of(program=right)
