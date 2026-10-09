"""Actual frozen-capture baseline guards; no future candidate provider qualification."""

from copy import deepcopy

import pytest


@pytest.fixture(scope="module")
def legacy():
    from test_sol61_captured_diffusion_independent import captured_case

    return captured_case(width=3, order=(2, 0, 1), uniform=True)


def test_capture_contract_is_distinct_from_candidate_evaluation(legacy):
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    _, _, program, token, _ = legacy
    validate_nonlinear_field_request(program, token)
    assert token.attrs["contract"] == "pops.spatial-field-residual@2"
    assert token.attrs["solve_request"]["schema_version"] == 2
    assert token.attrs["solve_request"]["realization"] == {
        "coefficient_face_policy": "pops.field.face-mean.arithmetic@1"
    }
    assert program._serialize()["version"] == 10
    captures = token.inputs[2 : 2 + token.attrs["capture_count"]]
    assert len(token.inputs[1].inputs) == len(captures)
    assert all(a is b for a, b in zip(token.inputs[1].inputs, captures, strict=True))


@pytest.mark.parametrize(
    "attack",
    [
        "capture_id",
        "capture_point",
        "reverse_captures",
        "coefficient_point",
        "face_policy",
        "future_contract",
    ],
)
def test_actual_capture_phase_and_identity_mutations_refuse_without_authoring_changes(
    legacy, attack
):
    from test_sol61_captured_diffusion_independent import clone
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    from pops.time import SolveRequestError
    from pops.time._program.serialization import _json_ready

    _, _, program, token, _ = legacy
    coefficient = token.inputs[1]
    captures = token.inputs[2 : 2 + token.attrs["capture_count"]]
    inputs = list(token.inputs)
    attrs = deepcopy(_json_ready(token.attrs))
    coefficient_attrs = deepcopy(_json_ready(coefficient.attrs))
    if attack in ("capture_id", "capture_point"):
        replacement = clone(captures[0])
        if attack == "capture_id":
            object.__setattr__(replacement, "id", replacement.id + 991)
        else:
            object.__setattr__(replacement, "point", program.stage("foreign-capture-phase", c=1))
        inputs[2] = replacement
        # Coordinate both input lists: this is stronger than one stale coefficient edge.
        inputs[1] = clone(coefficient, inputs=(replacement, *captures[1:]))
    elif attack == "reverse_captures":
        inputs[2 : 2 + len(captures)] = reversed(captures)
        inputs[1] = clone(coefficient, inputs=tuple(reversed(captures)))
    elif attack == "coefficient_point":
        inputs[1] = clone(coefficient, point=program.stage("foreign-coefficient-phase", c=1))
    elif attack == "face_policy":
        coefficient_attrs["coefficient_face_policy"] = "pops.field.face-mean.harmonic@1"
        inputs[1] = clone(coefficient, attrs=coefficient_attrs)
    else:
        # A tag change cannot retroactively receive new code or native science.
        attrs["contract"] = "pops.spatial-field-residual@3"
    forged = clone(token, attrs=attrs, inputs=tuple(inputs))
    before = program._ir_hash()
    if attack == "capture_id":
        # Scientific input validation need not authenticate a local SSA id; the
        # actual graph boundary must reject its undeclared input reference.
        validate_nonlinear_field_request(program, forged)
        coefficient_inputs, token_inputs = coefficient.inputs, token.inputs
        try:
            # Preserve the genuinely authored coefficient/token objects. Inject
            # the foreign capture into both live edges, then restore both edges.
            object.__setattr__(coefficient, "inputs", inputs[1].inputs)
            object.__setattr__(token, "inputs", (token_inputs[0], coefficient, *inputs[2:]))
            with pytest.raises(ValueError, match="was not authored by this Program"):
                program.validate()
        finally:
            object.__setattr__(coefficient, "inputs", coefficient_inputs)
            object.__setattr__(token, "inputs", token_inputs)
    else:
        with pytest.raises((SolveRequestError, ValueError)):
            validate_nonlinear_field_request(program, forged)
    assert program._ir_hash() == before
    validate_nonlinear_field_request(program, token)
