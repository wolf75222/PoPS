"""Authoring/serialization/code generation only; no native reception claim."""
import pytest

from examples.api_v02.coarsening_moments_v14 import build_coarsening_moments_v14
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_lowerability import all_ops
from pops.time._program.detach import detach_compiled_program


@pytest.mark.parametrize('mode', ('mean_only', 'enriched'))
def test_coarsening_checks_are_consumed_in_real_closed_program(mode):
    spec = build_coarsening_moments_v14(mode=mode)
    program = spec.case._time
    assert spec.case.validate() and not program._block_indices() and not spec.initial_values
    detached = detach_compiled_program(program)
    assert detached._ir_hash() == program._ir_hash()
    operations = list(all_ops(program))
    assert any(v.op == 'bundle_remap_validate' for v in operations)
    assert any(v.op == 'continuation_publish' for v in operations)
    cpp = emit_cpp_program(detached)
    assert 'moment_loss' in cpp and 'history_second' in cpp
    assert 'codec.validate(' in cpp and 'continuation_publish(' in cpp
    assert 'explicit_finite_volume_weights' in str(program._serialize())
