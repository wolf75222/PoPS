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


def field_case(*, joint=False, duplicate_input=False, component_transforms=False, publication_fields=False):
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
    if publication_fields:
        for name in ("observed_phi", "observed_gx", "observed_gy", "observed_static"):
            first_model.aux(name)
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


def _many_field_problem(count, *, reaction=None, gauge=None, cross=False):
    from pops.fields import ConstantModeGauge
    unknowns = tuple(Handle("phi%d" % i, kind="field", owner=OwnerPath.model("arbitrary fields"))
                     for i in range(count))
    if reaction is None:
        reaction = tuple(tuple(int(i == j) for j in range(count)) for i in range(count))
    equations = []
    for i, unknown in enumerate(unknowns):
        lhs = -DivCoeffGrad(unknown, 2 + i)
        if cross:
            for j, other in enumerate(unknowns):
                if i != j:
                    lhs = lhs - DivCoeffGrad(other, 0.125)
        for j, other in enumerate(unknowns):
            lhs = lhs + Reaction(other, reaction[i][j])
        equations.append(lhs == 0)
    constraint = None if gauge is None else ConstantModeGauge(unknowns, gauge)
    return FieldProblem("many", unknowns=unknowns, equations=tuple(equations),
        boundaries=tuple(FieldBoundary(u, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
                         for u in unknowns), gauge=constraint)


@pytest.mark.parametrize("count", (3, 4, 7))
@pytest.mark.parametrize("cross", (False, True))
def test_arbitrary_field_count_and_cross_diffusion_bind_exact_relations(count, cross):
    from pops.fields._program_problem import _physical_coefficients, validate_field_apply
    problem = _many_field_problem(count, cross=cross)
    diffusion, reaction = _physical_coefficients(problem)
    assert len(diffusion) == (count * count if cross else count)
    assert len(reaction) == count * count
    case = pops.Case("matrix field")
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(), solver=CG(max_iter=200)))
    program = pops.Program("matrix solve")
    result = program.solve(field, values={}, at=program.stage("solve", c=0)).consume(action=FailRun())
    observation = field.observe(result)
    assert len(observation.unknowns) == count
    coefficient = next(v for v in program._values if v.op == "field_problem_coefficients")
    assert coefficient.attrs["ncomp"] == len(diffusion)
    operator = next(v for v in program._values if v.op == "matrix_free_operator")
    apply = next(v for v in operator.attrs["apply_block"] if v.op == "field_problem_apply")
    validate_field_apply(apply)
    from types import SimpleNamespace
    forged = SimpleNamespace(op=coefficient.op, vtype=coefficient.vtype,
        attrs={**coefficient.attrs, "ncomp": coefficient.attrs["ncomp"] + 1})
    with pytest.raises(ValueError, match="coefficient matrix"):
        validate_field_apply(SimpleNamespace(op=apply.op, attrs=apply.attrs,
                            inputs=(*apply.inputs[:2], forged)))


def test_arbitrary_nonshared_and_independent_kernel_modes_are_proved_exactly():
    from pops.fields._program_problem import _physical_coefficients, _resolved_linear_properties
    def validate(problem):
        return _resolved_linear_properties(problem, _physical_coefficients(problem)[1])
    # Kernel (1,2,0), with a positive reaction on its complement.
    problem = _many_field_problem(3, reaction=((4,-2,0),(-2,1,0),(0,0,3)), gauge=((1,2,0),))
    validate(problem)
    # Two independent constant modes, not one shared mean.
    problem = _many_field_problem(3, reaction=((0,0,0),(0,0,0),(0,0,3)), gauge=((1,0,0),(0,1,0)))
    validate(problem)
    with pytest.raises(ValueError, match="complete declared kernel complement"):
        validate(_many_field_problem(3, reaction=((0,0,0),(0,0,0),(0,0,3)), gauge=((1,0,0),)))
    with pytest.raises(ValueError, match="not in the physical reaction kernel"):
        validate(_many_field_problem(3, gauge=((1,1,1),)))
    with pytest.raises(ValueError, match="linearly independent"):
        _many_field_problem(3, gauge=((1,2,0),(2,4,0)))


def test_nonorthogonal_mode_mean_constraints_lower_to_gram_coordinates():
    from pops.fields import ConstantModeGauge
    from pops.fields._constant_mode_nullspace import _author, _emit
    problem = _many_field_problem(3)
    gauge = ConstantModeGauge(problem.unknowns, ((1,0,0),(1,1,0)), (2,5))
    contracts = _author({"components": 3}, gauge, {}, "test")
    from pops.identity.scalar import scalar_literal
    from fractions import Fraction
    assert contracts.gauge["coordinates"] == (scalar_literal(Fraction(-1)), scalar_literal(Fraction(3)))
    emission = _emit(None, [], contracts, "test-mode-plan", None)
    assert "authored constant field mode" in emission.plan
    assert "ctx.alloc_scalar_field(3, 0)" in emission.plan


def test_three_field_cross_matrix_and_nonshared_modes_emit_actual_native_storage():
    from pops.codegen import Production
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.time import FixedDt
    case, _old_field, old_problem, program, values, point = field_case()
    program.solve(_old_field, values=values, at=point).consume(action=FailRun())
    relation = _many_field_problem(3, cross=True, reaction=((4,-2,0),(-2,1,0),(0,0,3)), gauge=((1,2,0),))
    problem = FieldProblem("matrix electric", unknowns=relation.unknowns,
        equations=tuple(eq.lhs == old_problem.equations[0].rhs for eq in relation.equations),
        boundaries=relation.boundaries, gauge=relation.gauge)
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(), solver=CG(max_iter=200)))
    solved = program.solve(field, values=values, at=point).consume(action=FailRun())
    observed = field.observe(solved, field[problem.unknowns[2]])
    program.record_scalar("third mean", program.sum_component(observed, 0))
    for handle in values:
        current = program.state(handle)
        program.commit(current.next, program.value("unchanged_" + current.n.name, 1 * current.n, at=current.next.point))
    program.step_strategy(FixedDt(0.1))
    case.program(program)
    frame = case._block_registry.spec("first")["model"]._frame
    grid = CartesianGrid(frame=frame, cells=(16,16), periodic=PeriodicAxes(frame.axes))
    plan = pops.resolve(pops.validate(case), layout=Uniform(grid), backend=Production())
    code = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "apply_general_field<pops::kNativeDimension, 3, 9>" in code
    assert "prepare_general_field_coefficients<pops::kNativeDimension, 3, 9>" in code
    assert "ctx.alloc_scalar_field(9, 1)" in code
    assert "authored constant field mode" in code
    assert "field_coeff_boundary_" in code


@pytest.mark.parametrize("reaction", (((2,1,0),(0,3,1),(0,0,4)), ((-2,0,0),(0,3,0),(0,0,4))))
def test_general_reaction_is_physical_and_krylov_provider_checks_compatibility(reaction):
    from pops.solvers import GMRES
    problem = _many_field_problem(3, reaction=reaction)
    case = pops.Case("general reaction")
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(), solver=GMRES(max_iter=200)))
    program = pops.Program("general reaction")
    program.solve(field, values={}, at=program.stage("solve", c=0)).consume(action=FailRun())
    other = pops.Program("incompatible CG")
    with pytest.raises(ValueError, match="positive|symmetric"):
        other.solve(field, solver=CG(max_iter=200), values={}, at=other.stage("solve", c=0))
