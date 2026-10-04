"""Physical stage analytic auxiliaries; Source lowering, no PoPS Native."""
from fractions import Fraction
import pytest
from pops.analytic import CellBounds, coordinate, time
from pops.codegen._analytic_aux import emit_analytic_aux_launcher
from pops.codegen._analytic_expression_lowering import lower_analytic_components
from pops.codegen._compile_emit import _emit_auxiliary_route_registration
from pops.codegen.cpp_strings import cpp_string_literal
from pops.codegen.module_lowering import _module_to_model
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.fields import AnalyticAux
from pops.frames import Cartesian2D
from pops.model import Module
from pops.time import Program
from tests.python.support.coupled_field_data_case import build


def producer(expression=None):
    frame=Rectangle('actual geometric and temporal coefficient',(0,0),(1,1)).frame(Cartesian2D())
    program=Program('actual auxiliary physical stage')
    module=Module('actual coefficient owner')
    target=module.aux_handle(module.aux_field('time_data',frame=frame.canonical_id))
    bounds=CellBounds(frame)
    value=bounds.lower(frame.axes[0])+bounds.measure+time(program.clock)
    result=AnalyticAux(target,value if expression is None else expression,frame=frame)
    module.aux_provider(result)
    return frame,program,module,result


def test_time_and_geometry_input_slots_are_separate_and_clock_authenticated():
    frame,program,module,result=producer()
    ((opcodes,literals),)=lower_analytic_components((result.expression.to_data(),),
        frame_id=frame.canonical_id,time_clock_id=program.clock.qualified_id,time_input_slot=5)
    assert [value for op,value in zip(opcodes,literals) if op=='input']==[0,4,5]
    with pytest.raises(ValueError,match='another logical Clock'):
        lower_analytic_components((result.expression.to_data(),),frame_id=frame.canonical_id,
            time_clock_id=Program('foreign time').clock.qualified_id,time_input_slot=5)
    with pytest.raises(TypeError,match='time input slot'):
        lower_analytic_components((result.expression.to_data(),),frame_id=frame.canonical_id,
            time_clock_id=program.clock.qualified_id,time_input_slot=True)
    source=emit_analytic_aux_launcher('actual_identity',result)
    assert 'geometry.face_coordinate(0, index[0] + 0)' in source
    assert 'geometry.spacing(0) * geometry.spacing(1)' in source
    assert source.count('= physical_stage_time;')==1
    assert 'context.point.require_physical_time('+cpp_string_literal(program.clock.qualified_id)+')' in source
    assert source.index('require_physical_time(')<source.index('Kokkos::parallel_reduce')
    assert 'pops.analytic-aux-time@1.' in source
    assert result.options()['temporal_contract']=='analytic-aux-time@1'


@pytest.mark.parametrize('reverse,permuted',((False,False),(True,True)))
@pytest.mark.parametrize('ssprk2',(False,True))
def test_public_coupled_time_provider_uses_every_consumed_stage(reverse,permuted,ssprk2,tmp_path):
    resolved=build(cells=(3,7),reverse=reverse,permuted=permuted,
        time_dependent=True,ssprk2=ssprk2)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source=emit_cpp_program(resolved.time,model_graph=graph)
    fractions=(Fraction(0),Fraction(1)) if ssprk2 else (Fraction(1,4),Fraction(3,4))
    positions=[source.index('ctx.prepare_provider_values('),source.rindex('ctx.prepare_provider_values(')]
    for point,position in zip(fractions,positions,strict=True):
        assert source.rfind('ctx.set_stage_time(%d, %d);'%(point.numerator,point.denominator),0,position)>=0
        assert position<source.index('pops::for_each_cell(',position)
    assert source.count('std::as_const(')>0
    from pops.codegen._compiler_lowering import require_compiler_lowering
    model=graph.model_for_block(resolved.blocks[0].name)
    launcher=require_compiler_lowering(model).native_loader_source(name='ActualTemporalModel')
    assert 'context.point.require_physical_time('+cpp_string_literal(resolved.time.clock.qualified_id)+')' in launcher
    (tmp_path/'program.cpp').write_text(source);(tmp_path/'model.cpp').write_text(launcher)


def test_unrelated_clock_combination_is_not_assumed_to_share_one_physical_stage():
    frame,program,module,result=producer()
    with pytest.raises(ValueError,match='one exact consuming logical Clock'):
        AnalyticAux(result.target,time(program.clock)+time(Program('unrelated').clock),frame=frame)
