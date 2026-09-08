"""Public general field solves preserve physical tuple and exact state input authority."""
from __future__ import annotations

import pytest

import pops
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import laplacian
from pops.model import Handle, OwnerPath
from pops.solvers import CG
from pops.time import FailRun


def field_case(*, joint=False, duplicate_input=False, component_transforms=False):
    case = pops.Case("general fields")
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    frame = Rectangle("field domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    first_model, second_model = pops.Model("first", frame=frame), pops.Model("second", frame=frame)
    components = ("rho", "a", "mx") if component_transforms else ("rho", "a")
    first_state = first_model.state("U", components=components)
    second_state = second_model.state("U", components=components)
    if component_transforms:
        rho, coefficient, momentum = first_state
        first_model.local_transform("momentum_update", (rho, coefficient, momentum + 1))
        first_model.local_transform("density_update", (2 * rho, coefficient, momentum))
        first_model.local_transform("coefficient_update", (rho, 2 * coefficient, momentum))
    selected_numerics = []
    for model, state in ((first_model, first_state), (second_model, second_state)):
        flux = model.flux("static flux", frame=frame, state=state,
            components={axis: tuple(0 * value for value in state) for axis in frame.axes},
            waves={axis: tuple(0 * value for value in state) for axis in frame.axes})
        rate = model.rate("static", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        selected_numerics.append(numerics)
    first_block, second_block = case.block("first", first_model), case.block("second", second_model)
    for block, numerics in zip((first_block, second_block), selected_numerics, strict=True):
        case.numerics(numerics, block=block)
    first, second = first_state[0], second_state[0]
    coefficient = first_state[1]
    unknowns = tuple(Handle(name, kind="field", owner=OwnerPath.model("physical fields"))
                     for name in (("phi1", "phi2") if joint else ("phi",)))
    if joint:
        one, two = unknowns
        equations = (-laplacian(one) + Reaction(one, 2) - Reaction(two, 2) == first,
                     -laplacian(two) + Reaction(two, 2) - Reaction(one, 2) == second)
        condition = bcs.Neumann(0)
    else:
        equations = (-DivCoeffGrad(unknowns[0], coefficient) == first + second,)
        condition = bcs.Periodic()
    problem = FieldProblem("electric", unknowns=unknowns, equations=equations,
        boundaries=tuple(FieldBoundary(unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), condition))
                         for unknown in unknowns), gauge=SharedMeanGauge(unknowns))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CG(max_iter=4000, rel_tol=1e-11, abs_tol=1e-12)))
    program = pops.Program("field stage")
    first_time, second_time = program.state(first_block[first_state]), program.state(second_block[second_state])
    values = {first_block[first_state]: first_time.n,
              second_block[second_state]: first_time.n if duplicate_input else second_time.n}
    point = program.stage("solve", c=0)
    return case, field, problem, program, values, point


@pytest.mark.parametrize("joint", (False, True))
def test_public_field_solve_builds_exact_native_equation_and_independent_storage(joint):
    case, field, problem, program, values, point = field_case(joint=joint)
    result = program.solve(field, values=values, at=point).consume(action=FailRun())
    observation = field.observe(result)
    value = observation[field[problem.unknowns[0]]]
    assert value.point == point
    assert value.block is value.state_ref is None
    load = next(node for node in program._values if node.op == "field_problem_load")
    coefficients = next(node for node in program._values if node.op == "field_problem_coefficients")
    assert load.block is coefficients.block is None
    assert load.state_ref is coefficients.state_ref is None
    assert len(load.attrs["field_dependencies"]) == 2
    assert len(coefficients.attrs["field_dependencies"]) == (0 if joint else 1)
    solve = next(node for node in program._values if node.op == "solve_linear")
    from pops.time._graph.base import CanonicalData
    from pops.time._program.serialization import _json_ready
    assert _json_ready(solve.attrs["solve_request"]["physical_problem"]["field_handle"]) == CanonicalData(case.resolve(field).canonical_identity()).to_data()
    assert solve.attrs["ncomp"] == (2 if joint else 1)
    for handle in values:
        current = program.state(handle)
        program.commit(current.next, program.value("unchanged_" + current.n.name, 1 * current.n, at=current.next.point))
    assert program.validate()


def test_public_field_binding_refuses_same_shaped_foreign_state_input():
    _case, field, _problem, program, values, point = field_case(duplicate_input=True)
    with pytest.raises(ValueError, match="exact state owner"):
        program.solve(field, values=values, at=point)
    assert not any(node.op == "field_problem_load" for node in program._values)


def test_observation_rejects_unconsumed_and_foreign_unknowns():
    _case, field, problem, program, values, point = field_case(joint=True)
    outcome = program.solve(field, values=values, at=point)
    with pytest.raises(TypeError, match="consumed"):
        field.observe(outcome)
    observed = field.observe(outcome.consume(action=FailRun()))
    foreign = Handle(problem.unknowns[0].name, kind="field", owner=OwnerPath.model("foreign"))
    with pytest.raises(ValueError, match="foreign field unknown"):
        observed[foreign]


def test_equal_timestamp_fresh_stage_has_distinct_equation_context():
    _case, field, _problem, program, values, point = field_case()
    program.solve(field, values=values, at=point).consume(action=FailRun())
    other = program.stage("solve", c=0)
    program.solve(field, values=values, at=other).consume(action=FailRun())
    solves = [node for node in program._values if node.op == "solve_linear"]
    assert solves[0].attrs["solve_request"]["equation_identity"] != solves[1].attrs["solve_request"]["equation_identity"]


@pytest.mark.parametrize("joint", (False, True))
def test_public_case_resolves_and_emits_the_actual_field_native_path(joint):
    from pops.codegen import Production
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.time import FixedDt
    case, field, problem, program, values, point = field_case(joint=joint)
    solved = program.solve(field, values=values, at=point).consume(action=FailRun())
    phi = field.observe(solved, field[problem.unknowns[0]])
    gradient = field.observe(solved).gradient(field[problem.unknowns[0]], dimension=2)
    assert gradient.point == point and gradient.state_ref is gradient.block is None
    program.store_history("gradient", gradient)
    program.record_scalar("phi sum", program.sum_component(phi, 0))
    for handle in values:
        current = program.state(handle)
        program.commit(current.next, program.value("unchanged_" + current.n.name, 1 * current.n, at=current.next.point))
    program.step_strategy(FixedDt(0.1))
    case.program(program)
    frame = case._block_registry.spec("first")["model"]._frame
    grid = CartesianGrid(frame=frame, cells=(16, 16),
        periodic=None if joint else PeriodicAxes(frame.axes))
    plan = pops.resolve(pops.validate(case), layout=Uniform(grid), backend=Production())
    assert set(plan.program_field_plans) == {"electric"}
    code = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "apply_general_field<pops::kNativeDimension" in code
    assert "prepare_general_field_coefficients" in code
    assert "general_field_operator.hpp" in code
    assert "ctx.gradient(*field_gradient_" in code
    assert "field gradient physical boundary refused collectively" in code
    import re
    assert re.search(r"prepare_mesh_boundary_session\(\*session_field_input_A\d+_\d+, "
                     r"ctx_owner->prepared_execution_lane\(\)\)", code)
