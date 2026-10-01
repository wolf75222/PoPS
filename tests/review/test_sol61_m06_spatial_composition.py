"""Public source witnesses: accumulation exists; original field stage bridge does not."""
from fractions import Fraction

import pops
import pytest
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.handle_expr import ValueExpr
from pops._ir.quantity import PhysicalSupport
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.fields import (CellCenteredNonlinearCoupled, FieldBoundary,
                         FieldDiscretization, FieldProblem, FieldProblemError, bcs)
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, grad, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.numerics import Diffusion, DiscretizationPlan
from pops.solvers import Newton
from pops.time import (DerivativeStrategy, FailRun, FixedDt, ImplicitStage,
                       SolveUnknown)


def base(*, explicit_support=True):
    frame = CartesianDomain("heat-box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("heat", frame=frame)
    kwargs = dict(sampling="cell_average", support=PhysicalSupport(
        (("x", "heat-box"), ("y", "heat-box")))) if explicit_support else {}
    energy = model.state("H", components=("energy",), **kwargs)
    case = pops.Case("heat-stage")
    block = case.block("material", model, states=(energy,))
    program = pops.Program("heat-step")
    current = program.state(block[energy])
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 6), periodic=PeriodicAxes(frame.axes)))
    return model, energy, case, block, program, current, layout


def field_stage(*, dynamic=False, explicit_support=True):
    model, energy, case, block, program, current, layout = base(explicit_support=explicit_support)
    temperature = Handle("temperature", kind="field", owner=OwnerPath.model("thermal-relation"))
    value = ValueExpr(temperature)
    tau = program.dt if dynamic else .01
    # Exactly H(T+) - tau*div(grad(T+)) = H_n; no chain-rule surrogate.
    problem = FieldProblem("energy-relation", unknowns=(temperature,), equations=(
        Reaction(temperature, 1 + value) - DivCoeffGrad(temperature, tau) == energy[0],),
        boundaries=(FieldBoundary(temperature, bcs.BoundaryCondition(
            bcs.AllPhysicalBoundaries(), bcs.Periodic())),))
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=1e-6), boundaries=(), solver=Newton(tolerance=1e-11)))
    request = field.bind_program_inputs(program=program, values={block[energy]: current.n},
        at=current.next.point, solver=field.default_program_solver())
    observation = field.observe(program.solve(request, solver=field.default_program_solver()).consume(action=FailRun()))
    return case, program, current, layout, field, field[temperature], observation


def test_original_field_literal_energy_residual_and_temperature_observation_emit():
    case, program, current, layout, field, temperature, observation = field_stage(explicit_support=False)
    observed = observation[temperature]
    program.store_history("temperature-observation", observed, depth=1)
    # This is an observation control, NOT an energy update by the field solution.
    program.commit(current.next, program.value("unchanged-energy", 1*current.n, at=current.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    node = next(v for v in resolved.time._values if v.op == "solve_spatial_field")
    assert node.inputs[2].op == "state"
    assert node.inputs[2].point == current.n.point
    assert "candidate(index, 0)" in code and "capture0(index, 0)" in code
    assert "original_field_residual_recheck_failed" in code
    assert "temperature-observation" in code
    assert observed.block is None and observed.space is None


def test_original_field_cannot_capture_the_actual_program_duration():
    with pytest.raises(FieldProblemError, match="diffusion requires exact State captures"):
        field_stage(dynamic=True)


def test_nonlinear_observation_has_no_cell_mean_state_projection_even_with_complete_support():
    _, program, current, _, _, temperature, observation = field_stage()
    before = program._next_id, len(program._values)
    with pytest.raises(FieldProblemError, match="projection requires the native cell-centred finite-volume stencil"):
        observation.cell_mean_state(temperature, target=current.next)
    assert (program._next_id, len(program._values)) == before
    assert current.next.space.support is not None
    assert current.next.space.sampling == "cell_average"
    assert not program._commits


def test_linear_field_projection_is_already_a_positive_public_route():
    from tests.python.unit.fields.test_m27_mixed_public_source import mixed_case
    _, _, _, field, problem, program, current, observed, _ = mixed_case()
    unknown = next(u for u in problem.unknowns if u.local_id == "c_next")
    candidate = observed.cell_mean_state(field[unknown], target=current.next)
    program.commit(current.next, candidate)
    assert candidate.state_ref == current.next.state
    assert candidate.space == current.next.space


@pytest.mark.parametrize("scale", (Fraction(1), Fraction(1, 2)))
def test_existing_implicit_stage_preserves_accumulated_energy_and_actual_dt(scale):
    frame = CartesianDomain("heat-box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("heat", frame=frame)
    energy = model.state("H", components=("energy",))
    h = energy[0]
    temperature = model.primitive("T", (sqrt(1 + 4*h) - 1) / 2)
    flux = model.diffusive_flux("conduction", state=energy, value=grad(temperature))
    rate = model.rate("heat", equation=ddt(energy) == div(flux))
    accumulation = model.local_transform("H_of_T", (h + h*h,))
    case = pops.Case("heat-stage")
    block = case.block("material", model, states=(energy,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    program = pops.Program("heat-step")
    current = program.state(block[energy])
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 6), periodic=PeriodicAxes(frame.axes)))
    seed = program.value("temperature-seed", 1*current.n, at=current.next.point)
    stage = ImplicitStage(rate, current.n, scale*program.dt, accumulation=accumulation)
    request = stage.request(unknown=SolveUnknown("T", seed), seed=seed,
        derivative=DerivativeStrategy("finite_difference"))
    solved = program.solve(request, solver=Newton(tolerance=1e-11)).consume(action=FailRun())[0]
    program.store_history("temperature-candidate", solved, depth=1)
    new_energy = program.transform(solved, transform=accumulation)
    program.commit(current.next, program.value("new-energy", new_energy, at=current.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    node = next(v for v in resolved.time._values if v.op == "solve_spatial_nonlinear")
    from pops.time._program.spatial_solve import spatial_rate_weight
    rhs = next(v for v in node.attrs["residual_block"] if v.op == "diffusive_rhs")
    assert spatial_rate_weight(node, rhs) == {1: scale}
    assert rhs.inputs[0].op == "local_transform"
    assert rhs.inputs[0].inputs[0] is node.attrs["iterate"]
    assert node.attrs["solve_request"]["residual_interpretation"] == "Q(q)-U_n-tau*R(Q(q))"
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "PreparedSpatialResidual<" in code and "PreparedDiffusion<" in code
    assert "ctx.commit_many(" in code and "temperature-candidate" in code
