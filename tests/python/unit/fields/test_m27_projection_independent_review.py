"""Public endpoint authority counter-tests, independent of the M27 trajectory."""
import pops
import pytest
import numpy as np

from pops._ir.quantity import PhysicalSupport
from pops.fields import FieldProblemError
from pops.frames import Cartesian1D
from pops.model import Handle, OwnerPath
from tests.python.unit.fields.test_m27_mixed_public_source import mixed_case


@pytest.mark.parametrize("wrong", ("point", "state", "unknown"))
def test_cell_mean_projection_refuses_foreign_authority_atomically(wrong):
    case, _block, _state, field, problem, program, current, observed, _layout = mixed_case()
    unknown, target = field[problem.unknowns[0]], current.next
    if wrong == "point":
        target = current.n
    elif wrong == "state":
        model = pops.Model("foreign", frame=Cartesian1D())
        state = model.state("c", components=("c",), sampling="cell_average",
            support=PhysicalSupport((("x", "mixed-periodic"),)))
        block = case.block("foreign", model, states=(state,))
        target = program.state(block[state]).next
    else:
        unknown = Handle("c_next", kind="field", owner=OwnerPath.model("other_owner"))
    before = program._next_id, len(program._values)
    with pytest.raises((FieldProblemError, ValueError, TypeError)):
        observed.cell_mean_state(unknown, target=target)
    assert (program._next_id, len(program._values)) == before


@pytest.mark.parametrize("permuted", (False, True))
def test_projection_carries_exact_consumed_original_load_and_selected_unknown(permuted):
    _case, _block, _state, field, problem, _program, current, observed, _layout = mixed_case(permuted=permuted)
    from pops.fields._observation_contract import validate_field_state_cell_mean

    c = next(u for u in problem.unknowns if u.local_id == "c_next")
    projected = observed.cell_mean_state(field[c], target=current.next)
    selected = validate_field_state_cell_mean(projected)
    assert selected.attrs["component"] == (1 if permuted else 0)
    assert projected.point == current.next.point
    assert projected.space == current.next.space
    assert projected.state_ref == current.next.state
    assert projected.attrs["sampling"] == "cell_average"
    assert projected.attrs["measure"] == "cell_volume"
    assert selected.inputs[0].op == "solve_outcome_component"


@pytest.mark.parametrize("permuted", (False, True))
def test_actual_field_lowering_matrices_match_both_original_equations(permuted):
    _case, _block, _state, _field, _problem, program, _current, _observed, _layout = mixed_case(permuted=permuted)
    def literal(data):
        if data["kind"] == "integer":
            return int(data["value"])
        if data["kind"] == "binary64":
            return float.fromhex(data["value"])
        assert data["kind"] == "rational"
        return int(data["numerator"])/int(data["denominator"])
    def expression(data):
        if data[0] == "literal":
            return literal(data[1])
        assert data[0] == "sub"
        return expression(data[1])-expression(data[2])
    coefficients = next(v for v in program._values if v.op == "field_problem_coefficients")
    operator = next(v for v in program._values if v.op == "matrix_free_operator")
    apply = next(v for v in operator.attrs["apply_block"] if v.op == "field_problem_apply")
    diffusion = np.array([expression(x) for x in coefficients.attrs["expressions"]]).reshape(2, 2)
    reaction = np.array([literal(x) for x in apply.attrs["reaction"]]).reshape(2, 2)
    order = (1, 0) if permuted else (0, 1)
    np.testing.assert_array_equal(diffusion, np.array(((0., .01), (-.08**2, 0.)))[np.ix_(order, order)])
    np.testing.assert_array_equal(reaction, np.array(((1., 0.), (-1., 1.)))[np.ix_(order, order)])
    assert apply.attrs["coefficient_admissibility"] == coefficients.attrs["coefficient_admissibility"] == "finite_general"


@pytest.mark.parametrize("kind", ("coordinate", "runtime_parameter", "named_variable"))
def test_closed_coefficient_grammar_refuses_non_state_captures(kind):
    from pops._ir.expr import Var
    from pops._ir.values import RuntimeParamRef
    from pops.analytic import coordinate
    from pops.fields._program_expression import encode_field_expression

    _case, _block, _state, _field, _problem, _program, current, _observed, _layout = mixed_case()
    frame = Cartesian1D()
    value = {"coordinate": lambda: coordinate(frame, frame.axes[0]),
             "runtime_parameter": lambda: RuntimeParamRef("coefficient", 1.),
             "named_variable": lambda: Var("x", "coordinate")}[kind]()
    with pytest.raises((TypeError, NotImplementedError, ValueError)):
        encode_field_expression(value, (current.n,))


def test_detached_empty_dependency_metadata_cannot_hide_coefficient_state_read():
    from pops.codegen.program_emit_field_problem import emit_field_problem_value
    from pops.time.references import canonical_handle

    _case, _block, _state, _field, _problem, program, _current, _observed, _layout = mixed_case()
    coefficients = next(v for v in program._values if v.op == "field_problem_coefficients")
    source = coefficients.inputs[0]
    expressions = list(coefficients.attrs["expressions"])
    expressions[0] = ("input", 0, 0, canonical_handle(source.state_ref).canonical_identity())
    detached = program._new("scalar_field", "field_problem_coefficients", coefficients.inputs,
        {**coefficients.attrs, "expressions": tuple(expressions), "field_dependencies": ()},
        "tampered_detached_metadata", None, point=coefficients.point, inherit_state_ref=False)
    with pytest.raises(ValueError, match="dependency metadata differs"):
        emit_field_problem_value(detached, {}, [], [], target="system")
