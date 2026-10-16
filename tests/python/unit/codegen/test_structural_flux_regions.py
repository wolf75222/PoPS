"""Structural provenance follows selected fields, never a Result's storage template."""
import pytest
from pops.time import Program
from pops.numerics.terms import Flux
from pops.codegen.program_lowerability import _check_amr_flux_weights
from typed_program_support import typed_state

from pops.codegen.program_structural_flux import field_projection_sources
from tests.python.unit.time.test_realization_contributions import fixture


def _ids(value):
    sources = field_projection_sources(value)
    assert sources is not None
    return {node.id for node in sources}


def test_nested_fold_projection_keeps_only_the_selected_field():
    _, p, q, _ = fixture()
    unused = p.value("unrelated", 4*q.n)
    initial = p.product(selected=q.n, unrelated=unused)
    output = p.fold(initial, 2, lambda b, carry, index: b.product(
        selected=carry["selected"], unrelated=carry["unrelated"]))
    assert _ids(output["selected"]) == {q.n.id}


def test_iteration_projects_initial_and_successful_body_without_error_template():
    _, p, q, _ = fixture()
    transformed = p.value("transformed", 2*q.n)
    template = p.value("storage_only", 9*q.n)
    initial = p.product(field=q.n, step=p.int64(0))
    def body(b, carry):
        return b.branch(carry["step"] == 0,
            lambda z: z.success(z.product(field=transformed, step=carry["step"]+1)),
            lambda z: z.failure(z.product(field=template, step=carry["step"]), 7))
    output = p.iterate(initial, body, lambda b, carry: carry["step"] == 1, max_steps=2)
    selected = p.require_success(output)["state"]["field"]
    assert _ids(selected) == {q.n.id, transformed.id}


def test_legacy_capture_and_result_match_keep_exact_selected_success():
    _, p, q, _ = fixture()
    success = p.value("success", 2*q.n)
    error_template = p.value("error_template", 3*q.n)
    fallback = p.value("fallback", 4*q.n)
    result = p.capture(lambda b: b.branch(b.boolean(True),
        lambda z: z.success(z.product(field=success)),
        lambda z: z.failure(z.product(field=error_template), 5)))
    assert _ids(p.require_success(result)["field"]) == {success.id}
    selected = p.match(result, ok=lambda b, value: value["field"], err=lambda b, error: fallback)
    assert _ids(selected) == {success.id, fallback.id}


@pytest.mark.parametrize("representation", ("field", "product"))
@pytest.mark.parametrize("capture", (False, True))
def test_iteration_and_capture_keep_integrated_flux(representation, capture):
    _check_amr_flux_weights(_iterated_flux(1, representation, capture))


@pytest.mark.parametrize("power", (0, 2))
@pytest.mark.parametrize("representation", ("field", "product"))
def test_iteration_and_capture_cannot_hide_wrong_flux_power(power, representation):
    with pytest.raises(ValueError, match="dt powers"):
        _check_amr_flux_weights(_iterated_flux(power, representation, True))


@pytest.mark.parametrize("representation", ("field", "product"))
def test_iteration_cannot_hide_an_unproved_transform_of_carried_flux(representation):
    with pytest.raises(ValueError, match="without exact flux-weight propagation"):
        _check_amr_flux_weights(_iterated_flux(1, representation, True, opaque=True))


def _iterated_flux(power, representation, capture, *, opaque=False):
    program = Program("iteration_flux_proof")
    state = typed_state(program, "fluid", state_name="U")
    initial = program.value("initial", 1*state.n, at=state.next.point)
    coefficient = (1, program.dt, program.dt*program.dt)[power]
    carry = program.product(field=initial) if representation == "product" else initial

    def update(body, carried):
        value = carried["field"] if representation == "product" else carried
        rate = body.rhs(state=value, terms=[Flux()])
        result = body.value("next", value + coefficient*rate, at=value.point)
        if opaque:
            result = body._new("state", "unproved_transform", (result,), {},
                               "unproved", result.block, space=result.space)
        return body.product(field=result) if representation == "product" else result

    def invoke(body):
        outcome = body.iterate(carry, update, lambda b, value: b.boolean(True), max_steps=3)
        selected = body.require_success(outcome)["state"]
        return selected["field"] if representation == "product" else selected

    selected = program.require_success(program.capture(invoke)) if capture else invoke(program)
    program.commit(state.next, selected)
    return program
