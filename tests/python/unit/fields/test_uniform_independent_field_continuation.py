"""Closed field algebra keeps callback capture ownership and exact leaves."""
import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen._orchestration_compile import build_program_model_graph
from examples.api_v02.continuation_independent_uniform import build_independent_uniform


@pytest.mark.parametrize("reject", (False,True))
def test_uniform_closed_field_observations_inside_branches_emit(reject):
    spec = build_independent_uniform(reject=reject)
    plan = pops.resolve(pops.validate(spec.case),layout=spec.layout)
    source = emit_cpp_program(plan.time,model=build_program_model_graph(plan),target="system")
    assert source.count("configure_continuation_field_provider(") == 2
    assert source.count("seal_continuation_field_observation(") == 9
    assert "continuation_publish" in source
    assert "field equation kernel" in source


def test_nested_operator_cannot_escape_or_capture_a_sibling_region():
    p = pops.Program("nested_apply_authority")
    escaped = []
    hidden = []
    def first(b):
        escaped.append(b.matrix_free_operator("hidden"))
        hidden.append(b.scalar_field("hidden_coefficient"))
        return b.int64(0)
    p.branch(p.boolean(True), first, lambda b:b.int64(0))
    before = p._ir_hash(),p._next_id
    with pytest.raises(ValueError,match="region"):
        p.set_apply(escaped[0],lambda b,out,x:x)
    assert (p._ir_hash(),p._next_id) == before
    caught=[]
    def second(b):
        operator = b.matrix_free_operator("local")
        with pytest.raises(ValueError,match="region"):
            b.set_apply(operator,lambda c,out,x:c.laplacian(out,hidden[0]))
        caught.append(True)
        b.set_apply(operator,lambda c,out,x:2*x)
        return b.int64(0)
    p.branch(p.boolean(True),second,lambda b:b.int64(0))
    assert caught == [True]
