"""Current equation input frontier excludes earlier field solves in stage ancestry."""
from __future__ import annotations

import pytest

from pops.time import FailRun
from tests.python.unit.fields.test_program_field_problem import field_case


def test_publication_retains_current_both_state_inputs_and_rejects_conflicting_one():
    case, field, problem, program, values, first_point = field_case(publication_fields=True)
    handles = tuple(values)
    first, second = (case.blocks()[name] for name in ("first", "second"))
    model = case._block_registry.spec("first")["model"]
    module = model.module
    carrier = first[module.field_handle(module.field_spaces()["fields"])]

    first_observed = field.observe(program.solve(field, values=values, at=first_point)
                                   .consume(action=FailRun()))
    first_context = first_observed.publish({
        (carrier, "observed_phi"): first_observed[field[problem.unknowns[0]]],
    })
    stage = program.stage("second physical stage", c=1)
    current_first = program.value("current first", 1.1 * values[handles[0]], at=stage)
    current_second = program.value("current second", 1.2 * values[handles[1]], at=stage)
    observed = field.observe(program.solve(field, values={
        handles[0]: current_first, handles[1]: current_second}, at=stage)
        .consume(action=FailRun()))
    binding = {(carrier, "observed_phi"): observed[field[problem.unknowns[0]]]}
    context = observed.publish(binding)
    assert dict(context.field_context.stage_sources) == {
        first: current_first.id, second: current_second.id,
    }
    assert context.field_context != first_context.field_context

    contradictory = program.value("contradictory first", 1.3 * values[handles[0]], at=stage)
    with pytest.raises(ValueError, match="conflicting or repeated block mappings"):
        observed.publish(binding, states={handles[0]: contradictory})

    # A previously published field is never permitted for the new stage,
    # even though the physical field handle and numeric timestamp still match.
    with pytest.raises(ValueError, match="incompatible field context"):
        program.rhs(state=current_first, fields=first_context, terms=[])
