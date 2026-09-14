"""One prepared implicit residual retains the full ordered state width."""
from fractions import Fraction

import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import RateExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import (DerivativeStrategy, FailRun, FixedDt, ImplicitStage,
                       ImplicitDiffusionStage, SolveUnknown, SolveRequestError)
from pops.solvers import Newton


def make_source_stage(*, nonlinear=False, width_mismatch=False, transport=False, weighted=False, zero=False, partition=False):
    frame = Rectangle("vector-domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("vector-source", frame=frame)
    state = model.state("U", components=("u", "v"))
    u, v = state
    source = model.source("algebraic_source", on=state, value=(-u + v, u - 2 * v))
    if transport:
        from pops.numerics import reconstruction, riemann, variables
        from pops.numerics.spatial import FiniteVolume
        flux = model.flux("transport", frame=frame, state=state,
                           components={axis: (u, v) for axis in frame.axes},
                           waves={axis: (1 + 0 * u, 1 + 0 * v) for axis in frame.axes})
        rate = model.rate("rate", equation=ddt(state) == -div(flux) + source)
        discretization = FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                      reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    else:
        rate = model.rate("rate", equation=ddt(state) == (RateExpr(()) if zero else
                          (Fraction(2, 3) * source - source + source if weighted else source)))
        discretization = StateStorage()
    if partition:
        complement = rate.select(rate.occurrences[0], rate.occurrences[2])
        rate = rate.select(rate.occurrences[1])
    accumulation = model.local_transform("coordinate_to_state", (u + u * u, v + v * v)) if nonlinear else None
    case = pops.Case("vector-implicit")
    block = case.block("vector-block", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, discretization)
    if partition:
        numerics.rates.add(complement, StateStorage())
    case.numerics(numerics, block=block)
    program = pops.Program("vector-stage")
    temporal = program.state(block[state])
    seed = program.value("seed", 1.0 * temporal.n, at=temporal.next.point)
    if width_mismatch:
        from pops.model import StateSpace
        seed = program._replace_value(seed, space=StateSpace("wrong-width", ("only",)))
    stage = ImplicitStage(rate, temporal.n, program.dt, accumulation=accumulation)
    request = stage.request(unknown=SolveUnknown("coordinates", seed), seed=seed,
                            derivative=DerivativeStrategy("finite_difference"))
    solved = program.solve(request, solver=Newton(tolerance=1e-12)).consume(action=FailRun())[0]
    conserved = solved if accumulation is None else program.transform(solved, transform=accumulation)
    candidate = program.value("candidate", conserved, at=temporal.next.point)
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))
    return case, layout


def test_legacy_diffusion_stage_is_the_same_generic_authority():
    from pops.time.implicit_diffusion import ImplicitDiffusionStage as legacy
    assert legacy is ImplicitDiffusionStage is ImplicitStage


@pytest.mark.parametrize("nonlinear", [False, True], ids=["identity", "nonidentity-Q"])
def test_two_component_source_only_stage_resolves_and_emits_existing_prepared_operations(nonlinear):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    case, layout = make_source_stage(nonlinear=nonlinear)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    node = next(value for value in resolved.time._values if value.op == "solve_spatial_nonlinear")
    request = node.attrs["solve_request"]
    assert len(request["unknowns"]) == 1
    assert request["unknowns"][0]["components"] == 2
    assert request["physical_problem"]["kind"] == "evolved_implicit_stage"
    operations = request["source_mapping"]["residual_operations"]
    assert "source" in operations and "diffusive_rhs" not in operations
    assert set(operations) <= {"state", "source", "local_transform", "linear_combine"}
    if nonlinear:
        source = next(value for value in node.attrs["residual_block"] if value.op == "source")
        assert source.inputs[0].op == "local_transform"
        assert source.inputs[0].inputs[0] is node.attrs["iterate"]
    emitter, _ = lower_and_validate(resolved.blocks[0].model)
    code = emit_cpp_program(resolved.time, model=emitter)
    assert "PreparedSpatialResidual<pops::kNativeDimension>" in code
    assert "PreparedDiffusion<" not in code
    assert "SolveConsumption::kAccept" in code
    assert "ctx.commit_many(" in code


def test_seed_width_mismatch_is_refused_before_residual_recording():
    with pytest.raises(SolveRequestError, match="unknown_type_mismatch"):
        make_source_stage(width_mismatch=True)


def test_transport_requires_a_separate_prepared_exchange_contract():
    with pytest.raises(SolveRequestError, match="unsupported_residual_operation"):
        make_source_stage(transport=True)


@pytest.mark.parametrize("zero", [False, True], ids=["signed-repeated-sources", "zero-balance"])
def test_source_occurrence_coefficients_and_zero_identity_reach_prepared_ir(zero):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    case, layout = make_source_stage(weighted=not zero, zero=zero)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    node = next(value for value in resolved.time._values if value.op == "solve_spatial_nonlinear")
    rate = next(value for value in node.attrs["residual_block"]
                if value.op == "linear_combine" and "physical_balance" in value.attrs)
    view = rate.attrs["physical_balance"]
    expected = [] if zero else [Fraction(2, 3), -1, 1]
    assert [row.coefficient for row in view.occurrences] == expected
    assert len(rate.inputs) == (1 if zero else 3)
    if not zero:
        assert all(value.op == "source" for value in rate.inputs)
    else:
        assert rate.inputs[0].op == "state"
    assert rate.attrs["coeffs"] == tuple(pops.time.values._Coeff({0: c}).to_polynomial()
                                         for c in ([0] if zero else expected))
    emitter, _ = lower_and_validate(resolved.blocks[0].model)
    assert "PreparedSpatialResidual<" in emit_cpp_program(resolved.time, model=emitter)


def test_selected_source_partition_keeps_qualified_balance_identity_and_signed_weight():
    case, layout = make_source_stage(weighted=True, partition=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    node = next(value for value in resolved.time._values if value.op == "solve_spatial_nonlinear")
    rate = next(value for value in node.attrs["residual_block"]
                if value.op == "linear_combine" and "physical_balance" in value.attrs)
    view = rate.attrs["physical_balance"]
    assert view.ordinals == (1,)
    assert [row.coefficient for row in view.occurrences] == [-1]
    assert rate.inputs[0].attrs["operator_handle"].owner_path == view.balance.handle.owner_path
