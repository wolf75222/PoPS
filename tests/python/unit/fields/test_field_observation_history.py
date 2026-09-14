"""Consumed independent fields own their scalar history width without a species surrogate."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.fields._observation_contract import validate_field_observation
from pops.runtime._checkpoint_resource_budget import _history_capacity
from pops.time import FailRun
from tests.python.unit.fields.test_program_field_problem import field_case


def _observation():
    _case, field, problem, program, values, point = field_case(joint=True)
    solution = program.solve(field, values=values, at=point).consume(action=FailRun())
    return program, field.observe(solution)[field[problem.unknowns[0]]]


def test_consumed_field_store_supplies_independent_scalar_component_authority():
    program, observed = _observation()
    program.store_history("phi", observed, depth=1)
    assert program._histories_ncomp == {"phi": 1}
    assert "phi" not in program._history_blocks
    assert validate_field_observation(observed)[:2] == (2, 0)
    names, _bytes, evidence = _history_capacity(
        program, cells=(16 * 16,), amr=False, block_nvars={})
    assert "history_phi_0" in names
    assert evidence == (("phi", 1, 1),)


@pytest.mark.parametrize("mutation, error", (
    ({"ncomp": 2}, "one consumed packed source"),
    ({"component": 2}, "outside its solved unknown tuple"),
    ({"component": 1}, "does not select its declared solved unknown"),
))
def test_field_history_cannot_invent_width_or_select_a_foreign_unknown(mutation, error):
    _program, observed = _observation()
    altered = SimpleNamespace(op=observed.op, vtype=observed.vtype, prog=observed.prog,
                             inputs=observed.inputs, attrs=dict(observed.attrs) | mutation)
    with pytest.raises(ValueError, match=error):
        validate_field_observation(altered)


def test_ordinary_unowned_scalar_history_keeps_strict_component_refusal():
    program, _observed = _observation()
    program.store_history("unknown", program.scalar_field("unknown"), depth=1)
    with pytest.raises(ValueError, match="has no component authority"):
        _history_capacity(program, cells=(16 * 16,), amr=False, block_nvars={})
