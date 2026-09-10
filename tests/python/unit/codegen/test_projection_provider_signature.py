"""Pointwise projection retains its exact state target and optional field reads."""
import pytest

from pops.model import FieldSpace, Operator, Signature, StateSpace


def test_projection_retains_optional_field_input_and_exact_target():
    state = StateSpace("U", ("q",))
    field = FieldSpace("bounds", ("floor",))
    projection = Operator("clamp", "projection", Signature((state, field), state))
    assert projection.signature.inputs == (state, field)
    assert projection.signature.output == state


@pytest.mark.parametrize("invalid", ("field_target", "different_state", "second_state", "field_first"))
def test_projection_cannot_change_its_state_target_or_hide_another_state(invalid):
    state = StateSpace("U", ("q",))
    other = StateSpace("V", ("q",))
    field = FieldSpace("bounds", ("floor",))
    inputs, output = (state, field), state
    if invalid == "field_target":
        output = field
    elif invalid == "different_state":
        output = other
    elif invalid == "second_state":
        inputs = (state, other)
    else:
        inputs = (field, state)
    with pytest.raises(TypeError, match="projection"):
        Operator("invalid", "projection", Signature(inputs, output))
