"""Independent source review: exact rate-evaluation ownership for persistent traces."""
import json
import re

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


def _ssprk_case(*, diagnostic=False):
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("u", components=("density",))
    flux = model.flux("flux", frame=frame, state=state,
                      components={axis: tuple(state) for axis in frame.axes},
                      waves={axis: (1.,) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("integral_stages")
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                   reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(plan, block=block)
    program = pops.Program("ssprk")
    temporal = program.state(block[state])
    first = rate(temporal.n)
    stage = program.stage("predictor", c=1)
    predictor = program.value("predictor", temporal.n + program.dt * first, at=stage)
    last = rate(predictor)
    accepted = program.value("accepted", .5 * temporal.n + .5 * predictor
                             + .5 * program.dt * last, at=temporal.next.point)
    program.commit(temporal.next, accepted)
    integral = program.integral_state("q", initial=.7)
    for selected in (first, last):
        program.accept_external_trace(integral, rate=selected, axis=0, side=1, component=0)
    if diagnostic:
        program.accept_external_trace(integral, rate=rate(temporal.n), axis=1, side=1,
                                      component=0)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    # Periodicity is immaterial to source ownership; native delivery must reject absent exterior.
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, integral


def test_ssprk_same_physical_occurrence_has_two_exact_evaluation_selectors():
    case, layout, _, integral = _ssprk_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
                             model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert json.dumps(integral.identity) in source
    assert source.index("ctx.declare_integral_state(") < source.index("ctx.install([=](double dt)")
    calls = re.findall(r"ctx\.consume_external_trace\([^;]+;", source)
    assert len(calls) == 2
    selectors = [re.search(r"TraceSelection\{(.*)\}, pops::Real", call).group(1) for call in calls]
    values = [json.loads("[" + selector + "]") for selector in selectors]
    assert values[0][:5] == values[1][:5]
    assert values[0][5] != values[1][5]
    assert all("/evaluation:" in row[5] for row in values)
    for row in values:
        # The selected author evaluation must also be the producer's exact context.
        assert source.count(json.dumps(row[5])) >= 2


def test_unaccepted_diagnostic_rate_cannot_supply_an_integral():
    case, layout, _, _ = _ssprk_case(diagnostic=True)
    with pytest.raises(ValueError, match="accepted, exact conservative"):
        resolved = pops.resolve(pops.validate(case), layout=layout)
        emit_cpp_program(resolved.time,
                         model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_integral_handle_cannot_be_reused_by_a_homonymous_program():
    _, _, first, integral = _ssprk_case()
    other = pops.Program(first.name)
    other.integral_state("q", initial=.7)
    with pytest.raises(ValueError, match="owned by this Program"):
        other.accept_external_trace(integral, rate=None, axis=0, side=1, component=0)


def test_detachment_preserves_integral_identity_and_transfer_rate_ids():
    from pops.time._program.detach import detach_compiled_program

    case, layout, _, integral = _ssprk_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    detached = detach_compiled_program(resolved.time)
    assert detached._serialize(include_provenance=False) == resolved.time._serialize(
        include_provenance=False)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    assert emit_cpp_program(detached, model_graph=graph) == emit_cpp_program(
        resolved.time, model_graph=graph)
    assert integral.identity in emit_cpp_program(detached, model_graph=graph)


def test_amr_delivery_is_in_post_synchronization_not_each_level_step():
    case, layout, _, _ = _ssprk_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time, target="amr_system",
                             model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    deliveries = [match.start() for match in re.finditer(r"ctx\.consume_external_trace\(", source)]
    assert len(deliveries) == 2
    guard = source.rindex("if (ctx.level() == 0) {", 0, deliveries[0])
    begin_phase = source.rindex("ctx.set_stage_time(1, 1);", 0, guard)
    assert begin_phase < guard < deliveries[0] < deliveries[1]
    advance = source.index("ctx.advance_hierarchy(dt, _advance_level);")
    assert source.index(".post_synchronization(dt);", advance) > advance
