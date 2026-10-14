"""Source-only integration; native event and rollback reception are separate."""
import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_lowerability import all_ops
from pops.time._program.detach import detach_compiled_program
from pops.time import ComputedDt
from examples.api_v02.localized_event_v14 import build_localized_event_v14


@pytest.mark.parametrize('mode',('success','stale_dense'))
def test_localized_event_full_graph_computed_duration_and_complete_candidate(mode):
    spec=build_localized_event_v14(mode=mode);p=spec.case._time
    assert not p._block_indices() and isinstance(p._step_strategy,ComputedDt)
    assert pops.validate(spec.case)
    assert detach_compiled_program(p)._ir_hash()==p._ir_hash()
    operations=list(all_ops(p))
    assert any(v.op=='structured_iterate' for v in operations)
    reached=next(v for v in p._values if v.op=='reached_duration')
    publication=next(v for v in p._values if v.op=='continuation_publish')
    assert reached.id<publication.id
    cpp=emit_cpp_program(p)
    assert 'ctx.reached_duration(' in cpp and 'ctx.continuation_publish(' in cpp
    assert cpp.index('ctx.reached_duration(')<cpp.index('ctx.continuation_publish(')
