"""Source authority and public product-space BGK; no installed execution claim."""
from fractions import Fraction
import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.m19_bgk_case import fixed_moment_counterexample

from tests.python.support.m19_bgk_case import build

def emit(resolved):
    return emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))

@pytest.mark.parametrize('reverse',(False,True))
def test_bgk_aux_counterexample_now_uses_exact_device_provider(reverse):
    source=emit(fixed_moment_counterexample(reverse=reverse))
    assert source.count('ctx.prepare_provider_values(')==1
    assert 'ctx.template provider_values_view<1>(' in source
    assert 'const pops::Real velocity_coordinate = providers(index, 0);' in source
    preparation=source.index('ctx.prepare_provider_values(')
    loop=source.index('for (int li = 0;',preparation)
    kernel=source.index('pops::for_each_cell(',loop)
    assert preparation<loop<kernel
    assert source.rfind('ctx.set_stage_time(0, 1);',0,preparation)>=0
    assert 'std::as_const(' in source

@pytest.mark.parametrize('nx,nv,reverse',((32,32,False),(64,64,True)))
def test_public_bgk_reduces_lifts_all_moments_at_each_tableau_stage(tmp_path,nx,nv,reverse):
    resolved=build(tmp_path,nx=nx,nv=nv,reverse=reverse,scope='renamed arbitrary collision owner')
    assert len(resolved.layout_plan.mappings)==6
    reductions=[row.requirement.physical_map.reductions[0] for row in resolved.layout_plan.mappings
                if row.requirement.physical_map.reductions]
    dv=Fraction(16,nv);v=tuple(-8+(j+Fraction(1,2))*dv for j in range(nv))
    assert {row.weights for row in reductions}=={tuple(dv*x**k for x in v) for k in range(3)}
    values=resolved.time._values
    assert sum(row.op=='layout_map_import' for row in values)==12
    rates=[row for row in values if row.op=='coupled_rate']
    assert len(rates)==2 and all(len(row.inputs)==4 for row in rates)
    assert all(len({state.point for state in row.inputs})==1 for row in rates)
    source=emit(resolved)
    assert source.count('ctx.prepare_provider_values(')==2
    positions=[source.index('ctx.prepare_provider_values('),source.rindex('ctx.prepare_provider_values(')]
    for c,pos in zip((0,1),positions,strict=True):
        assert source.rfind('ctx.set_stage_time(%d, 1);'%c,0,pos)>=0
    before=resolved.time._ir_hash()
    assert emit(resolved)==source and resolved.time._ir_hash()==before
    (tmp_path/'bgk-program.cpp').write_text(source)

def independent_exchange(*,provide=True,kind='aux'):
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.fields import AnalyticAux, AuxiliaryBoundary
    from pops.analytic import coordinate
    from pops.math import Var
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    frame=Rectangle('independent exchange',(0,0),(1,1)).frame(Cartesian2D())
    m=pops.Model('unrelated reversible exchange',frame=frame)
    a=m.species('first',state=('amount',));b=m.species('second',state=('amount',))
    p=pops.Program('exchange at a fractional evaluation point')
    if provide:
        aux=m.module.aux_field('gate','cell_scalar',frame=frame.canonical_id,unit='1')
        m.module.aux_provider(AnalyticAux(m.module.aux_handle(aux),
            coordinate(frame,frame.axes[1]),frame=frame,
            boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    q=Var('gate',kind)*(b[0]-a[0])
    op=m.coupled_rate('exchange',inputs=(a,b),outputs={a:(q,),b:(-q,)})
    case=pops.Case('distinct shared provider composition')
    ab=case.block('z_first',m,states=(a,));bb=case.block('a_second',m,states=(b,))
    sa,sb=p.state(ab[a]),p.state(bb[b]);point=p.stage('fractional callback',c=Fraction(2,3))
    va=p.value('first at stage',sa.n,at=point);vb=p.value('second at stage',sb.n,at=point)
    rates=op(va,vb)
    for state,block,value in ((sa,ab,va),(sb,bb,vb)):
        p.commit(state.next,p.value('next '+block.local_id,value+p.dt*rates[block],at=state.next.point))
    from pops.time import FixedDt
    p.step_strategy(FixedDt(.01));case.program(p)
    return pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(frame=frame,cells=(3,5),periodic=PeriodicAxes(frame.axes))))

def test_independent_exchange_with_shared_component_names_and_fractional_stage_aux():
    source=emit(independent_exchange())
    assert source.count('pops::for_each_cell(')==1
    pos=source.index('ctx.prepare_provider_values(')
    assert source.rfind('ctx.set_stage_time(2, 3);',0,pos)>=0
    assert 'pops_input_0_component_0' in source and 'pops_input_1_component_0' in source
    assert 'const pops::Real gate = providers(index, 0);' in source

def test_missing_provider_refuses_before_cpp_emission():
    with pytest.raises(ValueError,match="symbolic dependency 'gate'"):
        emit(independent_exchange(provide=False))

def test_primitive_recipe_remains_deferred():
    with pytest.raises(NotImplementedError,match='prim/aux vars are deferred'):
        emit(independent_exchange(kind='prim'))

def test_wrong_graph_cannot_reroute_the_coupled_operator():
    resolved=fixed_moment_counterexample();unrelated=independent_exchange()
    with pytest.raises((KeyError,ValueError),match='route for block'):
        emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(unrelated.blocks))
