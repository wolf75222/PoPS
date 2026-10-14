"""Source gates for the N07 native hierarchy receiver; no surrogate engine."""
import pytest
import pops
from examples.api_v02.hierarchy_flux_v14 import build_hierarchy_flux_v14
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize('mode', ('success','constant','stale_stage','last_flux','double_h','old_volume','parent_reject'))
def test_two_barrier_hierarchy_case_resolves_without_legacy_amr(mode):
    spec=build_hierarchy_flux_v14(mode=mode)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert not plan.blocks
    values=spec.case._time._values
    assert not any(value.op.startswith('amr_') for value in values)
    assert any(value.op=='continuation_publish' for value in values)
    assert 'increment_identity_v02' in str(spec.case._time._serialize())
    assert any(value.op=='bundle_validate' for value in values)
    if mode=='success':
        cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
        assert 'Continuation' in cpp and 'ctx.continuation_initialize' in cpp
