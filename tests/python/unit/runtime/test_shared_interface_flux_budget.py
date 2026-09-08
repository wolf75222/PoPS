"""Shared sample retention is bounded by real endpoint RHS/history provenance."""
import re

import pytest

from pops.codegen.program_emit_amr import _emit_flux_expression_budget
from pops.numerics.terms import Flux
from pops.time import Program
from tests.python.support.typed_program import program_states, synthetic_module


def _bounds(program):
    source = _emit_flux_expression_budget(program)
    return tuple(int(re.search(
        rf"pops_program_interface_coupling_{name}\(\) \{{ return UINT64_C\((\d+)\)",
        source).group(1)) for name in ("application_bound", "identity_character_bound"))


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_shared_sample_upper_bound_counts_both_endpoint_retained_rings(lag):
    program = Program("shared_bound_%d" % lag)
    _, states = program_states(program, synthetic_module("shared_bound_model", components=("u",)),
                               ("left", "right"))
    for name, temporal in states.items():
        state = temporal.n
        rate = program.rhs(state=state, terms=[Flux()])
        expression = state + program.dt * rate
        if lag:
            program.store_history(name + ".rate", rate, depth=lag)
            previous = program.history(name + ".rate", lag=lag, space=rate.space,
                                       block=state.block, state_ref=state.state_ref)
            expression = expression + program.dt * previous
        value = program.value(name + "_next", expression, at=temporal.next.point)
        program.commit(temporal.next, value)
    samples, characters = _bounds(program)
    assert samples == 2 * (lag + 1)
    assert characters == len(program._ir_hash()) + len("shared-rhs/") + 10 + len("program-rhs-group")


def test_pure_state_history_grants_no_shared_sample_capacity():
    program = Program("no_shared_rhs")
    _, states = program_states(program, synthetic_module("no_shared_rhs_model", components=("u",)),
                               ("left", "right"))
    for name, temporal in states.items():
        state = temporal.n
        program.store_history(name, state, depth=3)
        program.history(name, lag=3, space=state.space, block=state.block, state_ref=state.state_ref)
    assert _bounds(program) == (0, 0)
