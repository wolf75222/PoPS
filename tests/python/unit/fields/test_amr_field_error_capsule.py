"""A fine Field remains a typed opaque value across hierarchy recovery."""
import pops
import pytest

from examples.api_v02.continuation_capture_amr import build_continuation_capture_amr
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_source_map import source_map


@pytest.mark.parametrize("illegal_read", (False, True))
def test_fine_field_error_keeps_full_support_and_authored_access_choice(illegal_read):
    spec = build_continuation_capture_amr(field_payload=True, illegal_read=illegal_read)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(plan.time,
        model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target="amr_system")
    assert "HierarchyBarrierKind::nonlinear_solve" in source
    assert "ctx.invocation_level()" in source
    assert "ctx.raise_hierarchy_typed_failure(" in source
    assert "ctx.hierarchy_failure_value(" in source
    assert "ctx.own_structural_field(" in source
    assert "ctx.require_program_field_access(" in source
    assert "recovered_storage" in source
    assert source_map(source)["complete"]
