"""AMR finite algebra source contracts; native reception lives in a separate file."""
import pops
import pytest

from examples.api_v02.finite_amr import build_finite_amr
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_amr_finite_scope import continuation_scope_capture, resume_scope
from pops.time._program.detach import detach_compiled_program


@pytest.mark.parametrize("distributed,ratio", ((False, 1), (False, 2), (True, 1), (True, 2)))
def test_full_amr_program_emits_finite_algebra_and_native_invocation_scope(distributed, ratio):
    spec = build_finite_amr(distributed=distributed, ranks=3, ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(plan.time, target="amr_system",
                             model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "ctx.finite_value_scope()" in source
    assert "ctx.advance_hierarchy(dt, _advance_level)" in source
    assert ("ctx.distributed_registry()" if distributed else "ctx.continuation_new_domain(") in source
    assert "finite_total" in source and "30" in source
    assert plan.time._ir_hash() == detach_compiled_program(plan.time)._ir_hash()


def test_barrier_retains_and_reactivates_the_exact_finite_invocation_lease():
    environment = {("amr_finite_scope",): True}
    capture = continuation_scope_capture("=, &u0", environment)
    assert capture == "=, &u0, _finite_value_lease = ctx.current_finite_value_lease()"
    lines = []
    resume_scope(lines, environment)
    assert lines == ["auto _finite_value_scope = ctx.resume_finite_value_scope(_finite_value_lease);"]
    assert continuation_scope_capture("=", {}) == "="
    lines = []
    resume_scope(lines, {})
    assert lines == []


def test_post_reflux_rejection_has_a_separate_finite_scope():
    spec = build_finite_amr(reject_parent=True)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(plan.time, target="amr_system",
                             model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert source.count("ctx.finite_value_scope()") == 2
    assert "finite_parent_reject" in source
