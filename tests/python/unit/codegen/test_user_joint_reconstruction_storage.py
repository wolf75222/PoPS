"""Storage and nesting must cover reads requested by another conservative row."""
import pytest

from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.module_codegen import _emit_bricks
from tests.python.unit.codegen.test_user_joint_reconstruction import joint_case


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("mixed,cross_offset,expected", ((False, 1, 3), (True, 3, 4)))
def test_every_sampled_row_allocates_the_complete_group_halo(reverse, mixed, cross_offset, expected):
    resolved = joint_case(reverse=reverse, mixed=mixed, cross_offset=cross_offset)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    for block in resolved.blocks:
        native = graph.model_for_block(block.name)._m
        assert native._program_state_ghost_depth == expected, block.name
        body = _emit_bricks(native)[1]
        assert "program_state_ghost_depth = %d;" % expected in body
        assert block.numerics.primary_spatial().runtime_spatial().ghost_depth == expected
        requirement = block.numerics.amr_stencil_requirement(owner=block.numerics.rates[0].rate.owner_path, dimension=2)
        assert requirement.minimum_buffer == (expected, expected)
