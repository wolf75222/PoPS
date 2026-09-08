"""Explicit consumed observations own exact provider inputs and their stage context."""
from __future__ import annotations

import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FailRun, FixedDt
from tests.python.unit.fields.test_program_field_problem import field_case


def publication_case():
    case, field, problem, program, values, point = field_case(publication_fields=True)
    model = case._block_registry.spec("first")["model"]
    module = model.module
    target = case.blocks()["first"][module.field_handle(module.field_spaces()["fields"])]
    solution = field.observe(program.solve(field, values=values, at=point).consume(action=FailRun()))
    gradient = solution.gradient(field[problem.unknowns[0]], dimension=2)
    context = solution.publish({
        (target, "observed_phi"): solution[field[problem.unknowns[0]]],
        (target, "observed_gx"): (gradient, 0),
        (target, "observed_gy"): (gradient, 1),
    })
    for handle in values:
        state = program.state(handle)
        program.commit(state.next, program.value("unchanged_" + state.n.name, 1 * state.n, at=state.next.point))
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    frame = model._frame
    layout = Uniform(CartesianGrid(frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, context, target


def test_consumed_field_publication_resolves_exact_existing_provider_and_emits_transaction():
    case, layout, program, context, target = publication_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    plan = resolved.blocks[0].resolved_operations
    claims = plan.provider_evidence["program_field_publications"]
    assert len(claims) == 3 and claims[0]["key"]["space_name"] == target.local_id
    assert claims[0]["producer"] == context.attrs["field_problem_identity"]
    assert context.field_context.stage_sources
    code = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "ctx.publish_field_components(" in code
    assert "provider:pops.field-problem" in code
    publication = code.index("ctx.publish_field_components(")
    assert "ctx.prepare_provider_values(" in code[:publication]
    assert "observed_static" in code


def test_publication_rejects_unconsumed_or_unqualified_input():
    case, field, problem, program, values, point = field_case()
    outcome = program.solve(field, values=values, at=point)
    with pytest.raises(TypeError, match="consumed"):
        field.observe(outcome)
    solution = field.observe(outcome.consume(action=FailRun()))
    with pytest.raises(TypeError, match="block-qualified"):
        solution.publish({field[problem.unknowns[0]]: solution[field[problem.unknowns[0]]]})


@pytest.mark.parametrize("transport", (False, True))
def test_actual_source_and_flux_consumers_resolve_field_signature(transport):
    from tests.python.integration.runtime.test_public_field_consumers import consumer_case
    case, layout = consumer_case(16, transport=transport)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(case._time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "ctx.publish_field_components(" in code
    assert "potential_grad_y" in code


@pytest.mark.parametrize("corruption", ("component", "space", "source_component"))
def test_publication_node_authenticates_destination_and_selected_width(corruption):
    from pops.fields._program_publication import validate_field_publication
    case, layout, program, context, target = publication_case()
    # IR replacement is a deliberate counterexample to authoring checks: admission reauthenticates.
    rows = tuple(dict(row) for row in context.attrs["bindings"])
    if corruption == "component":
        rows[0]["component"] = "not_declared"
    elif corruption == "source_component":
        rows[0]["source_component"] = 1
    if corruption == "space":
        program._replace_value(context, space=None)
    else:
        program._replace_value(context, attrs={**context.attrs, "bindings": rows})
    replaced = next(value for value in program._values if value.id == context.id)
    with pytest.raises(ValueError, match="component|field space"):
        validate_field_publication(replaced)
