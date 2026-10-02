"""Explicit joint bound migration, retaining the original physical matrices."""
import numpy as np
import pops
from pops.model.hash_data import canonical_hash_data
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.principal_primitive_case import primitive_case

def test_joint_wave_bound_is_explicit_and_preserves_full_matrix_lambda():
    case,layout,_,matrices,_=primitive_case((3,4),reverse=True,map_reverse=True)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    module=resolved.blocks[0].model.module
    expected={axis:(float(np.max(np.sum(abs(matrix),axis=1))),)*7
              for axis,matrix in zip(('x','y'),matrices)}
    assert canonical_hash_data(module._eigenvalues)==canonical_hash_data(expected)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    assert graph is not None
