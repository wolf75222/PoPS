import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_lowerability import all_ops
from pops.codegen.program_models import ProgramModelGraph
from examples.api_v02.method_relaxed_composite import build_t10_composite


@pytest.mark.parametrize("ratio", (1, 2))
def test_composite_affine_publication_lowers_one_parent_region(ratio):
    spec = build_t10_composite(temporal_ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    values = list(all_ops(plan.time))
    assert sum(v.op == "accept_increment" for v in values) == 1
    assert sum(v.op == "composite_pair" for v in values) == 3
    assert not plan.time._post_sync_commits
    cpp = emit_cpp_program(plan.time, target="amr_system",
                           model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    for token in ("ctx.begin_affine_finalization", "ctx.prepare_affine_finalization",
                  "ctx.accept_increment", "ctx.finish_affine_finalization", "ctx.composite_pair"):
        assert token in cpp
    assert "_level_programs->front().post_synchronization(dt)" in cpp
    assert "PyObject" not in cpp


def test_finalization_operations_cannot_escape_the_parent_scope():
    program = pops.Program("invalid_affine_scope")
    with pytest.raises(ValueError, match="after_synchronization"):
        program.accept_increment(scale=program.float64(1), duration=program.float64(.1))
    with pytest.raises(ValueError, match="after_synchronization"):
        program.attempt_start(None)


def test_duplicate_and_conditional_affine_publications_fail_during_authoring():
    program = pops.Program("duplicate_affine")
    def duplicate(builder):
        scale, duration = builder.float64(1.), builder.float64(.1)
        builder.accept_increment(scale=scale, duration=duration)
        builder.accept_increment(scale=scale, duration=duration)
    with pytest.raises(ValueError, match="only once"):
        program.after_synchronization(duplicate)
    other = pops.Program("conditional_affine")
    def conditional(builder):
        scale, duration = builder.float64(1.), builder.float64(.1)
        builder.branch(scale > 0, lambda b: b.accept_increment(scale=scale, duration=duration),
                       lambda b: duration)
    with pytest.raises(ValueError, match="direct after_synchronization"):
        other.after_synchronization(conditional)
