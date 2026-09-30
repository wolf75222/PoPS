"""Source reception of original bodies through the real native interval API."""
from fractions import Fraction
import pops
import pytest
from pops.analytic import coordinate, sin, time
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes, PeriodicAxes
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, StateStorage, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt, MovingFieldProjection
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


def declared_case(*, components=("density",), velocity=.7, with_source=False,
                  selected_reconstruction=None, cells=16, output_mode=None,
                  face_weights=(Fraction(1,2),Fraction(1,2)),source_measure_weights=(1,0),proposed_dt=.001):
    frame=CartesianDomain("domain",(0.,),(1.,)).frame(Cartesian1D())
    model=pops.Model("arbitrary_transport",frame=frame)
    state=model.state("q",components=components)
    flux=model.flux("authored_flux",frame=frame,state=state,
        components={axis:tuple(velocity*entry for entry in state) for axis in frame.axes},
        waves={axis:(velocity,)*len(components) for axis in frame.axes})
    source=model.source("actual_source",on=state,value=tuple(.05*entry for entry in state)) if with_source else None
    balance=model.rate("original_equation",equation=ddt(state)==(-div(flux) if source is None else -div(flux)+source))
    physical=balance if source is None else balance.select(flux)
    source_rate=None if source is None else balance.select(source)
    case=pops.Case("moving_interval"); block=case.block("fluid",model)
    plan=DiscretizationPlan()
    plan.rates.add(physical,FiniteVolume(flux=flux,variables=variables.Conservative(state),
        reconstruction=selected_reconstruction or reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    if source_rate is not None: plan.rates.add(source_rate,StateStorage())
    case.numerics(plan,block=block)
    program=pops.Program("original_reynolds")
    temporal=program.state(block[state]); x=coordinate(frame,frame.axes[0])
    evolution=GeometryEvolution((x+.08*sin(6.283185307179586*x)*time(program.clock),))
    geometry=program.geometry_state(temporal.n,evolution=evolution)
    candidate=program.reynolds_update(geometry,physical_rate=physical(temporal.n),
        source_rate=None if source_rate is None else source_rate(temporal.n),
        projection=MovingFieldProjection(face_weights,source_measure_weights),
        geometry_tolerance=1e-13,at=temporal.next.point)
    program.commit(temporal.next,candidate); program.step_strategy(FixedDt(proposed_dt)); case.program(program)
    layout=MovingControlVolumes(Uniform(CartesianGrid(frame=frame,cells=(cells,),
        periodic=PeriodicAxes(frame.axes))),evolution=evolution)
    if output_mode is not None:
        from pops.output import ConsumerGraph, NPZ, ScientificOutput
        case.consumers(ConsumerGraph.from_consumers((ScientificOutput(
            format=NPZ(output_mode),schedule=pops.time.every(1,clock=program.clock),
            fields=(block[state],),target="accepted-moving"),)))
    return case,layout


@pytest.mark.parametrize("components,velocity",[(('a',),.7),(('a','b','c'),-.3),(('c','a','b'),.7)])
@pytest.mark.parametrize("with_source",[False,True])
def test_original_bodies_emit_one_owned_interval_without_static_rhs(components,velocity,with_source):
    case,layout=declared_case(components=components,velocity=velocity,with_source=with_source)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source=emit_cpp_program(resolved.time,model_graph=graph)
    assert "project_moving_interval<%d>"%len(components) in source
    assert "ctx.prepare_moving_interval_update(" in source
    assert "ctx.commit_moving_interval(" in source
    assert "ctx.commit_many(" not in source
    assert "mesh_speed*u[c]" in source
    assert "face.swept_volume/time.duration" in source
    assert "ctx.rhs_into(" not in source
    assert "_resource=" not in source
    assert "result.physical_amount[c]=time.duration*flux.checked_density()" in source
    from pops.time._program.detach import detach_compiled_program
    assert emit_cpp_program(detach_compiled_program(resolved.time),model_graph=graph)==source


def test_unrealized_reconstruction_refuses_as_a_provider_limit():
    case,layout=declared_case(selected_reconstruction=reconstruction.WENO5())
    with pytest.raises(NotImplementedError,match="moving stencil metric provider"):
        pops.resolve(pops.validate(case),layout=layout)


def test_moving_scientific_consumer_resolves_its_exact_state_and_layout():
    from pops.output import ParallelMode
    case,layout=declared_case(output_mode=ParallelMode.SERIAL)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert len(resolved.consumer_graph.nodes)==1
    assert resolved.consumer_graph.nodes[0].quantities[0].layout_id==resolved.layout_plan.layouts[0].handle.qualified_id


@pytest.mark.parametrize("weights",[(1,0),(0,1),(2,-1),(Fraction(1,3),Fraction(2,3))])
def test_noncentered_density_policy_requires_its_own_numerical_face_provider(weights):
    # Authoring keeps the general policy. This selected realization refuses it
    # during resolve, before any compiler/runtime materialization.
    case,layout=declared_case(face_weights=weights)
    validated=pops.validate(case)
    with pytest.raises(NotImplementedError,match="centered Rusanov.*face_weights=.*prepared numerical-face provider"):
        pops.resolve(validated,layout=layout)


def test_source_measure_quadratures_remain_general_and_operant():
    sources=[]
    for weights in ((1,0),(0,1),(Fraction(1,2),Fraction(1,2)),(2,-1)):
        case,layout=declared_case(with_source=True,source_measure_weights=weights,
                                 face_weights=(.5,.5))
        resolved=pops.resolve(pops.validate(case),layout=layout)
        source=emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
        measure=next(line for line in source.splitlines() if "const auto measure=" in line)
        assert "cell.previous_measure" in measure and "cell.measure" in measure
        assert "time.duration*measure*" in source
        sources.append(measure)
    assert len(set(sources))==4,"distinct authored quadratures must change the real source projection"
