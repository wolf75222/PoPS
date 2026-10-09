"""Public three-state provider data, two explicit stages; no Native execution."""
from fractions import Fraction
import numpy as np
import pytest
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.support.coupled_field_data_case import build,FRACTIONS,NAMES
from tests.python.support.coupled_provider_data_oracle import data,step

@pytest.mark.parametrize('cells,reverse,permuted',(((7,3),False,False),((3,7),True,True)))
def test_coupled_data_uses_exact_catalyst_and_two_stage_preparations(tmp_path,cells,reverse,permuted):
    resolved=build(cells=cells,reverse=reverse,permuted=permuted)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source=emit_cpp_program(resolved.time,model_graph=graph)
    rates=[node for node in resolved.time._values if node.op=='coupled_rate']
    assert len(rates)==2 and all(len(node.inputs)==3 for node in rates)
    assert all(node.inputs[0].block.local_id==NAMES[2 if permuted else 0] for node in rates)
    positions=[source.index('ctx.prepare_provider_values('),source.rindex('ctx.prepare_provider_values(')]
    for point,position in zip(FRACTIONS,positions,strict=True):
        marker='ctx.set_stage_time(%d, %d);'%(point.numerator,point.denominator)
        assert source.rfind(marker,0,position)>=0
        assert position<source.index('pops::for_each_cell(',position)
    assert source.count('const pops::Real nonlinear_gain = providers(index, 0);')==2
    assert source.count('ctx.template provider_values_view<1>(')==2
    assert source.count('pops::for_each_cell(')==2
    assert 'std::as_const(' in source
    before=resolved.time._ir_hash()
    assert emit_cpp_program(resolved.time,model_graph=graph)==source and resolved.time._ir_hash()==before
    (tmp_path/'coupled-provider-data.cpp').write_text(source)

@pytest.mark.parametrize('cells',((7,3),(3,7)))
def test_independent_reference_conserves_exchange_and_exact_catalyst(cells):
    states,gain=data(cells);before=tuple(a.copy() for a in states)
    for _ in range(2):states=step(states,gain)
    np.testing.assert_allclose(states[0][0]+states[1][0],before[0][0]+before[1][0],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(states[2],before[2])
    assert np.max(abs(states[0]-before[0]))>1e-2
    np.testing.assert_allclose(states[0][1]-before[0][1],2*(1/64)*gain,rtol=0,atol=1e-14)

def test_time_dependent_analytic_aux_keeps_one_exact_declared_clock():
    # Dynamic providers retain the declared Clock; the original witness stays static.
    from pops.fields import AnalyticAux
    from pops.analytic import time
    from pops.time import Program
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.model import Module
    frame=Rectangle('time-contract',(0,0),(1,1)).frame(Cartesian2D())
    module=Module('time-contract owner',frame=frame);aux=module.aux_field('time_data')
    clock=Program('time-contract').clock
    producer=AnalyticAux(module.aux_handle(aux),time(clock),frame=frame)
    assert producer.expression.time_clocks()==(clock,)
    assert producer.options()['temporal_contract']=='analytic-aux-time@1'
