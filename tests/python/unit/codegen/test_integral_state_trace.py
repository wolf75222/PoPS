"""Public integral declarations retain exact accepted FV evaluations without a model recipe."""

import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt


def _case(*, extra_unaccepted=False):
    frame = Rectangle("trace_domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("u", components=("density",))
    flux = model.flux("flux", frame=frame, state=state,
                      components={axis: tuple(state) for axis in frame.axes},
                      waves={axis: (1.,) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("trace_case")
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                   reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(plan, block=block)
    program = pops.Program("trace_ssprk2")
    temporal = program.state(block[state])
    first = rate(temporal.n)
    stage = program.stage("predictor", c=1)
    predictor = program.value("predictor", temporal.n + program.dt * first, at=stage)
    second = rate(predictor)
    accepted = program.value("accepted", .5 * temporal.n + .5 * predictor
                             + .5 * program.dt * second, at=temporal.next.point)
    program.commit(temporal.next, accepted)
    quantity = program.integral_state("q", initial=.7)
    program.accept_external_trace(quantity, rate=first, axis=0, side=1, component=0)
    program.accept_external_trace(quantity, rate=second, axis=0, side=1, component=0)
    if extra_unaccepted:
        program.accept_external_trace(quantity, rate=rate(temporal.n), axis=1,
                                      side=1, component=0)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, quantity


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_two_stage_trace_has_two_exact_evaluations_and_one_persistent_identity(target):
    case, layout, program, quantity = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
                              model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                              target=target)
    deliveries = [line for line in source.splitlines() if "ctx.consume_external_trace(" in line]
    assert len(deliveries) == 2
    assert "/evaluation:1" in deliveries[0]
    assert "/evaluation:3" in deliveries[1]
    assert deliveries[0] != deliveries[1]
    assert all(quantity.identity in line for line in deliveries)
    assert source.count("ctx.stage_exchange_batch(") == 2
    assert source.index("ctx.declare_integral_state(") < source.index("ctx.install(")
    if target == "system":
        assert source.index(deliveries[0]) < source.index("ctx.commit_many(")
    else:
        # AMR stages all active level/subcycle faces, then consumes once after hierarchy sync.
        assert source.count("if (ctx.level() == 0) {") >= 1
        assert source.index("ctx.advance_hierarchy(dt") < source.rindex(".post_synchronization(dt)")
    assert program._serialize()["version"] == 5
    assert resolved.time._integral_transfers == program._integral_transfers


def test_amr_accepted_trace_intersects_finest_owner_coverage_with_physical_activity():
    case, layout, _, _ = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
                              model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                              target="amr_system")
    # Both accepted evaluations retain their own dt and check coverage before
    # staging a physical face. The EB mask alone is null on an ordinary mesh.
    assert source.count("ctx.pointwise_active_mask(") == 2
    assert source.count("ctx.pointwise_exchange_coverage_mask(") == 2
    assert source.count("coverage_values(cell,0) < 0.5") == 2
    assert source.count("active_values(cell,0) < 0.5") == 2
    assert source.count("(pops::Real(1) / pops::Real(2)) * dt), 1, axis, side, component") == 2


def test_unaccepted_rate_cannot_supply_a_persistent_integral():
    case, layout, _, _ = _case(extra_unaccepted=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="accepted, exact conservative"):
        emit_cpp_program(resolved.time,
                         model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_exact_integral_handle_rejects_foreign_program_and_duplicate_selector():
    _, _, program, quantity = _case()
    first = next(value for value in program._values if value.op == "rhs")
    with pytest.raises(ValueError, match="already has a persistent-state consumer"):
        program.accept_external_trace(quantity, rate=first, axis=0, side=1, component=0)
    other = program.integral_state("other", initial=0.)
    with pytest.raises(ValueError, match="already has a persistent-state consumer"):
        program.accept_external_trace(other, rate=first, axis=0, side=1, component=0)
    with pytest.raises(ValueError, match="finite and nonzero"):
        program.accept_external_trace(other, rate=first, axis=1, side=1,
                                      component=0, scale=0.)
    foreign = pops.Program(program.name)
    foreign.integral_state("q", initial=.7)
    with pytest.raises(ValueError, match="owned by this Program"):
        foreign.accept_external_trace(quantity, rate=first, axis=0, side=1, component=0)
