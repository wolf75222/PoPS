"""Real public AMR resolve/emission of transformed principal stages and row widths."""
import re
import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_amr import _flux_expression_budgets
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.principal_amr_case import principal_amr_case


@pytest.mark.parametrize('widths,reverse,levels,stages,singular', (
    ((1,1),False,1,False,False), ((1,1),False,1,True,False),
    ((1,1),True,2,True,False), ((2,3),False,2,False,False),
    ((2,3),True,2,True,False), ((1,1),False,2,True,True)))
def test_complete_joint_amr_emission_keeps_exact_stages_and_flux_projections(
        widths,reverse,levels,stages,singular):
    case,layout,*_ = principal_amr_case(widths,reverse=reverse,levels=levels,
                                       stages=stages,singular=singular)
    plan = pops.resolve(pops.validate(case),layout=layout)
    assert plan.resolved_hierarchy.plan.level_count == levels
    source = emit_cpp_program(plan.time,model=ProgramModelGraph.from_resolved_blocks(plan.blocks),
                              target='amr_system')
    evaluations = 2 if stages else 1
    assert len(re.findall(r'\w+_evaluate\(ctx,\d+,\w+_inputs',source)) == evaluations
    assert source.count('ctx.attach_principal_flux(') == len(widths)*evaluations
    assert _flux_expression_budgets(plan.time) == ((evaluations,1),)*len(widths)
    assert 'prepare_principal_inputs(inputs,blocks,node,sources)' in source
    assert 'ctx.set_stage_time(0,1)' in source
    if stages:
        assert 'ctx.set_stage_time(1,1)' in source
    first = 0
    for width in widths:
        assert source.count('.component_faces(%d,%d)' % (first,width)) == evaluations
        first += width
    assert 'params.get(0)' in source and 'params.get(2)' in source
