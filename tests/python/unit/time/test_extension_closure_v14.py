"""Source composition/serialization only; numerical reception is installed native."""
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.time._program.detach import detach_compiled_program
from examples.api_v02.extension_closure_v14 import build_extension_closure_v14
from examples.api_v02.distributed_gc_v14 import build_distributed_gc


@pytest.mark.parametrize('permutation,names',[
    ((0,1,2,3),('q','H')),((2,0,3,1),('temperature','inventory'))])
def test_equation_extension_is_closed_zero_block_program(permutation,names):
    spec=build_extension_closure_v14(permutation=permutation,names=names)
    p=spec.case._time
    assert spec.case.validate() and not p._block_indices() and not spec.initial_values
    assert detach_compiled_program(p)._ir_hash()==p._ir_hash()
    cpp=emit_cpp_program(p)
    assert 'continuation_publish(' in cpp and 'for (std::int64_t' in cpp


@pytest.mark.parametrize('ranks',(2,3))
def test_gc_frontiers_are_compiled_outside_local_body(ranks):
    spec=build_distributed_gc(ranks=ranks);p=spec.case._time
    assert spec.case.validate()
    assert detach_compiled_program(p)._ir_hash()==p._ir_hash()
    cpp=emit_cpp_program(p)
    assert cpp.count('::collect_garbage<pops::kNativeDimension>')>=5
    assert '.configure(65536,2)' in cpp
