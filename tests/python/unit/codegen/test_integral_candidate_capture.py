"""Typed global reads enter the common Program IR, never a uniform circuit field."""
import pytest
import re
import pops
from pops._ir.quantity import PhysicalDimension
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt

_DIMENSIONLESS = PhysicalDimension()


def build_feedback(*, gamma=.3, initial=.7, units=_DIMENSIONLESS, cells=8,
                   periodic=True, proposed_dt=.01, selected_axis=0, initial_condition=False,
                   physical_global=False, components=("mass",), repeat_capture=False):
    frame = Rectangle("feedback", lower=(0.,0.), upper=(1.,1.)).frame(Cartesian2D())
    model = pops.Model("feedback_physics", frame=frame)
    state = model.state("density", components=components)
    x,y = frame.axes
    flux = model.flux("transport", frame=frame, state=state,
        components={x:tuple(state[c] for c in range(len(components))),
                    y:tuple(0*state[c] for c in range(len(components)))},
        waves={x:tuple(1. for _ in components),y:tuple(0. for _ in components)})
    if physical_global:
        circuit = model.global_quantity("circuit_quantity", units=units)
        reaction = model.source("reaction", on=state,
            value=tuple(-gamma*circuit*state[c] for c in range(len(components))))
    else:
        reaction = model.source("reaction", on=state,
            value=tuple(-gamma*state[c] for c in range(len(components))))
    balance = model.rate("balance", equation=ddt(state)==-div(flux)+reaction)
    reaction_rate = balance.select(reaction)
    transport = balance.select(flux)
    case = pops.Case("global_feedback")
    block = case.block("fluid",model)
    plan = DiscretizationPlan()
    plan.rates.add(transport,FiniteVolume(flux=flux,variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.User(state=state,
            body=lambda left,right,fl,fr,speed:.5*(fl+fr)-.5*speed*(right-left),
            stability=lambda left,right,fl,fr,speed:speed)))
    plan.rates.add(reaction_rate,StateStorage())
    if not periodic:
        from pops.boundary import TransportBoundarySet
        from pops.boundary.transport import Outflow
        plan.boundaries.add(TransportBoundarySet({
            frame.boundaries.x_min:Outflow(state=block[state]),
            frame.boundaries.x_max:Outflow(state=block[state]),
        },periodic=PeriodicAxes((y,))))
    case.numerics(plan,block=block)
    program = pops.Program("feedback_lie")
    temporal = program.state(block[state])
    quantity = program.integral_state("q",initial=initial,units=units)
    q = program.integral_value(quantity,at=temporal.n.point,scope="candidate")
    source = (program.evaluate_source(reaction_rate,temporal.n,
        global_inputs={block[circuit]:q}) if physical_global else reaction_rate(temporal.n))
    reacted = program.value("reaction_candidate",
        tuple(temporal.n[c]+program.dt*(source[c] if physical_global else q*source[c])
              for c in range(len(components))),at=temporal.n.point)
    if repeat_capture:
        reacted = program.value("reaction_candidate_again",
            tuple(reacted[c]+program.dt*q*source[c] for c in range(len(components))),
            at=temporal.n.point)
    rate = transport(reacted)
    accepted = program.value("accepted",reacted+program.dt*rate,at=temporal.next.point)
    program.commit(temporal.next, accepted)
    program.accept_external_trace(quantity,rate=rate,axis=selected_axis,side=1,component=0,scale=-1.)
    program.step_strategy(FixedDt(proposed_dt))
    case.program(program)
    if initial_condition:
        from pops.initial import InitialCondition
        from pops.lib.initial import BindArray
        from pops.projection import ConservativeCellAverage
        case.initials.add(InitialCondition(state=block[state],value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame,cells=(cells,1),
                                  periodic=PeriodicAxes(frame.axes if periodic else (y,))))
    return case,layout,program,quantity,temporal


@pytest.mark.parametrize("target",("system","amr_system"))
@pytest.mark.parametrize("periodic",(True,False))
def test_typed_integral_capture_is_a_true_scalar_pod_in_native_cell_kernel(target,periodic):
    case,layout,program,quantity,_ = build_feedback(periodic=periodic)
    resolved = pops.resolve(pops.validate(case),layout=layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),target=target)
    assert quantity.identity.startswith("pops.integral.v2/")
    assert source.count("ctx.capture_integral_candidate(")==1
    assert source.count("ctx.integral_candidate_value(")==1
    assert source.count("ctx.consume_external_trace(")==1
    checked = source.index("ctx.integral_candidate_value(")
    assert source.index("pops::for_each_cell",checked)>checked
    assert "integral_capture_1A" not in source
    assert "global_value_" in source and "ctx.set_stage_time(0, 1)" in source
    assert not re.search(r"const pops::Real \w+ = integral_capture_\d+;", source)
    # One exact host read feeds both the numerical equation and its affine face proof.
    pod = re.search(r"const pops::Real (global_value_\d+_\d+) = ctx.integral_candidate_value", source)
    assert pod is not None
    assert source.count("= %s;" % pod.group(1)) >= 2
    assert resolved.time._integral_units==program._integral_units
    assert "integral_units_v2" in program._serialize()


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_one_capability_has_distinct_checked_pod_bindings_for_repeated_equation_uses(target):
    case, layout, *_ = build_feedback(repeat_capture=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target=target)
    captures = re.findall(
        r"const pops::Real (global_value_\d+_\d+) = ctx.integral_candidate_value\((integral_capture_\d+),",
        source)
    assert len(captures) == 2
    assert len({alias for alias, _ in captures}) == 2
    assert len({capability for _, capability in captures}) == 1
    assert not re.search(r"const pops::Real \w+ = integral_capture_\d+;", source)
    for alias, _ in captures:
        assert source.count("= %s;" % alias) >= 2


def test_typed_capture_metadata_mutation_is_refused_by_emitter():
    case,layout,program,_,_ = build_feedback()
    capture = next(value for value in program._values if value.op=="integral_candidate")
    program._replace_value(capture,attrs={**capture.attrs,"units":capture.attrs["units"]+" "})
    resolved = pops.resolve(pops.validate(case),layout=layout)
    with pytest.raises(ValueError,match="metadata changed"):
        emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_units_scope_owner_point_and_direct_body_are_explicit():
    _,_,program,quantity,temporal = build_feedback()
    with pytest.raises(ValueError,match="candidate scope"):
        program.integral_value(quantity,at=temporal.n.point,scope="accepted")
    legacy = program.integral_state("legacy",initial=1.)
    with pytest.raises(ValueError,match="physical units"):
        program.integral_value(legacy,at=temporal.n.point,scope="candidate")
    with pytest.raises(ValueError,match="exact IntegralState"):
        pops.Program("foreign").integral_value(quantity,at=temporal.n.point,scope="candidate")
    different = program.integral_value(quantity,at=temporal.next.point,scope="candidate")
    with pytest.raises(ValueError,match="same exact point"):
        program.value("badpoint",(temporal.n[0]*different,),at=temporal.n.point)
    with pytest.raises(TypeError,match="direct Equation/FieldProblem"):
        different.to_cpp()


def test_units_and_source_body_are_in_semantic_identity_but_legacy_has_no_new_metadata():
    first = build_feedback()[2]
    assert first._ir_hash()!=build_feedback(gamma=.31)[2]._ir_hash()
    assert first._ir_hash()!=build_feedback(units=PhysicalDimension((("charge",1),)))[2]._ir_hash()
    program = pops.Program("legacy")
    quantity = program.integral_state("q",initial=.7)
    assert quantity.identity.startswith("pops.integral.v1/")
    assert "integral_units_v2" not in program._serialize()
