"""Independent exact-integral authoring controls; native qualification remains separate."""
import math

import pytest

from pops.analytic import CellBounds, constant, coordinates, input, param
from pops.frames import Cartesian
from pops.lib.initial import Analytic, Gaussian
from pops.runtime._analytic_expression_lowering import lower_analytic_components
from pops.runtime._initial_source_lowering import validate_cell_integral_contract


def evaluate(expression, frame, bounds):
    ((ops, values),) = lower_analytic_components([expression.to_data()], frame_id=frame.canonical_id)
    bounds = [*bounds, math.prod(bounds[2 * a + 1] - bounds[2 * a]
                                 for a in range(len(frame.axes)))]
    stack = []
    for op, value in zip(ops, values):
        if op == "constant": stack.append(value)
        elif op == "input": stack.append(bounds[int(value)])
        elif op in {"erf", "erfc", "exp", "sqrt", "neg"}:
            a = stack.pop()
            stack.append(-a if op == "neg" else getattr(math, op)(a))
        elif op == "where":
            no, yes, condition = stack.pop(), stack.pop(), stack.pop()
            stack.append(yes if condition else no)
        else:
            b, a = stack.pop(), stack.pop()
            if op == "add": result = a + b
            elif op == "sub": result = a - b
            elif op == "mul": result = a * b
            elif op == "div": result = a / b
            elif op == "pow": result = a ** b
            elif op == "ge": result = a >= b
            elif op == "le": result = a <= b
            else: raise AssertionError(op)
            stack.append(result)
    assert len(stack) == 1
    return stack[0]


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_polynomial_integral_is_composable_and_additive_in_every_rank(dim):
    frame = Cartesian(dim)
    cell = CellBounds(frame)
    point = constant(1.0)
    integral = constant(1.0)
    for axis, coordinate in zip(frame.axes, coordinates(frame)):
        point = point * (1 + coordinate ** 2)
        lo, hi = cell.lower(axis), cell.upper(axis)
        integral = integral * ((hi - lo) + (hi ** 3 - lo ** 3) / 3)
    profile = Analytic(frame=frame, components=(point, 2 * point),
                       cell_integrals=(integral, 2 * integral))
    data = profile.initial_source_options()["cell_integrals"]
    validate_cell_integral_contract(data, frame_id=frame.canonical_id, component_count=2)
    full = evaluate(integral, frame, [0, 1] * dim)
    assert full == pytest.approx((4 / 3) ** dim, rel=2e-15)
    left = [0, 0.4] + [0, 1] * (dim - 1)
    right = [0.4, 1] + [0, 1] * (dim - 1)
    assert evaluate(integral, frame, left) + evaluate(integral, frame, right) == pytest.approx(full)


def test_exact_integral_rejects_foreign_bounds_point_coordinates_and_forged_inputs():
    frame = Cartesian(2)
    cell = CellBounds(frame)
    with pytest.raises(ValueError, match="unauthenticated"):
        Analytic(frame=frame, components=(constant(1),), cell_integrals=(input(0, "other"),))
    with pytest.raises(ValueError, match="point coordinates"):
        Analytic(frame=frame, components=(constant(1),), cell_integrals=(coordinates(frame)[0],))
    with pytest.raises(ValueError, match="point expressions"):
        Analytic(frame=frame, components=(cell.measure,))
    with pytest.raises(ValueError, match="one ScalarExpr"):
        Analytic(frame=frame, components=(constant(1),), cell_integrals=())
    with pytest.raises(ValueError, match="does not belong"):
        cell.lower(Cartesian(3).z)


@pytest.mark.parametrize("sign", [-1, 1])
def test_python_gaussian_preset_preserves_nonzero_far_tail_integrals(sign):
    frame = Cartesian(1)
    profile = Gaussian(frame=frame, center={frame.x: 0.0}).as_analytic()
    lo, hi = (8.0, 8.1) if sign > 0 else (-8.1, -8.0)
    actual = evaluate(profile.cell_integrals[0], frame, [lo, hi])
    expected = math.sqrt(math.pi) / 2 * (math.erfc(8) - math.erfc(8.1))
    assert expected > 0
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)
    assert profile.initial_source_options()["native_route"] == "analytic_expression"


def test_integral_contract_authenticates_frame_measure_and_bound_authority():
    frame = Cartesian(1)
    profile = Analytic(frame=frame, components=(constant(1),),
                       cell_integrals=(CellBounds(frame).measure,))
    data = profile.initial_source_options()["cell_integrals"]
    with pytest.raises(ValueError, match="frame differs"):
        validate_cell_integral_contract(data, frame_id=Cartesian(2).canonical_id, component_count=1)
    with pytest.raises(ValueError, match="declared exactness"):
        validate_cell_integral_contract({**data, "measure": "point_value"},
                                        frame_id=frame.canonical_id, component_count=1)


def test_cell_integral_parameters_are_recaptured_from_exact_handle_authorities():
    from tests.python.unit.initial.test_initial_authoring import _case
    from pops.initial import InitialCondition
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    case, frame, _, state, point_parameter = _case()
    # A parameter used only by the integral still participates in the captured authority set.
    integral_parameter = case.param(RuntimeParam("integral-only-coefficient", default=2.0))
    cell = CellBounds(frame)
    profile = Analytic(frame=frame, components=(param(point_parameter),),
                       cell_integrals=(param(integral_parameter) * cell.measure,))
    initial = InitialCondition(state=state, value=profile, projection=ConservativeCellAverage())
    resolved = initial.resolve_references(case.resolve)
    source = resolved.source(case.owner_path).options.to_data()
    expression = source["cell_integrals"]["components"][0]
    reference = expression["root"]["arguments"][0]["reference"]
    assert "ownership_phase" not in reference
    assert reference["param_kind"] == "runtime"
    validate_cell_integral_contract(source["cell_integrals"], frame_id=frame.canonical_id,
                                    component_count=1)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("domain_name", ["transport-domain", "renamed-physical-support"])
def test_parameterized_gaussian_mixtures_lower_as_generic_integrals(dim, domain_name):
    import pops
    from pops.domain import CartesianDomain
    from pops.model import BindSchema
    from pops.params import RuntimeParam
    frame = CartesianDomain(domain_name, (0.0,) * dim, (1.0,) * dim).frame()
    first = Gaussian(frame=frame, center={axis: 0.25 for axis in frame.axes},
                     inverse_width=20.0).as_analytic()
    second = Gaussian(frame=frame, center={axis: 0.75 for axis in frame.axes},
                      inverse_width=80.0).as_analytic()
    case = pops.Case("mixture-owner")
    weight = case.param(RuntimeParam("weight", default=1.0))
    profile = Analytic(frame=frame,
                       components=(param(weight) * first.components[0] + second.components[0],),
                       cell_integrals=(param(weight) * first.cell_integrals[0] + second.cell_integrals[0],))
    resolved = profile.resolve_references(case.resolve)
    schema = BindSchema.from_problem(case)
    compile_values = schema.resolve_compile()
    lowered = []
    for value in (0.25, 2.0):
        binding = schema.resolve_bind({weight: value}, compile_values=compile_values)
        lowered.append(lower_analytic_components(
            [resolved.cell_integrals[0].to_data()], frame_id=frame.canonical_id, bindings=binding)[0])
    assert lowered[0][0] == lowered[1][0]
    assert lowered[0][1] != lowered[1][1]
    assert "erfc" in lowered[0][0] and "parameter" not in lowered[0][0]
    validate_cell_integral_contract(resolved.initial_source_options()["cell_integrals"],
                                    frame_id=frame.canonical_id, component_count=1)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("target", ["system", "amr_system"])
@pytest.mark.parametrize("mixture", [False, True])
def test_public_native_qualification_case_resolves_exact_source(dim, target, mixture):
    from tests.python.integration.runtime.test_exact_analytic_initial import _case
    resolved, _ = _case(target, 16, dim, mixture)
    (binding,) = resolved.initial_condition_plan.bindings
    source = binding.source.options.to_data()
    assert source["native_route"] == "analytic_expression"
    validate_cell_integral_contract(source["cell_integrals"], frame_id=source["frame_id"],
                                    component_count=1)
