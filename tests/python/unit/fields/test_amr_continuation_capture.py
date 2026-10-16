"""Actual nonlinear solve suspension inside an authored assessed child scope."""
import pops
import pytest

from examples.api_v02.continuation_capture_amr import build_continuation_capture_amr
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_source_map import source_map


@pytest.mark.parametrize("recover", (False, True))
def test_complete_child_scope_promotes_only_after_hierarchy_vote(recover):
    spec = build_continuation_capture_amr(recover=recover)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(plan.time,
        model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target="amr_system")
    assert "HierarchyBarrierKind::nonlinear_solve" in source
    assert "child_scope.emplace(ctx.continuation_authority().begin_scope())" in source
    assert "ctx.prepare_hierarchy_scope_promotion(" in source
    vote = source.index("ctx.validate_hierarchy_scope_promotions(")
    commit = source.index("ctx.commit_hierarchy_scope_promotions(")
    assert vote < source.index("accepted hierarchy child promotion failed", vote) < commit
    assert "child_scope.reset();" in source
    assert ".promote(std::move(*" not in source
    assert source_map(source)["complete"]
