"""Source contracts; generation is distinct from native reception."""
import pytest
from pops.continuations import DecisionAssess, _origin_checked
from pops.graph_functions import GraphFunction, graph
from pops.time import Program
from pops.time.variant_types import Bool, Failure, Float64, Int64, Product, Result, Variant, type_of
from pops.time._program.detach import detach_compiled_program
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_lowerability import all_ops
from examples.api_v02.decision_assess_v14 import build_decision_assess_v14


@pytest.mark.parametrize('mode',('success','stop','parent_stop','budget'))
def test_decision_algebra_has_native_scopes_transform_and_closed_failure(mode):
    spec=build_decision_assess_v14(mode=mode)
    p=spec.case._time
    assert spec.case.validate() and not p._block_indices()
    assert detach_compiled_program(p)._ir_hash()==p._ir_hash()
    names=[v.op for v in all_ops(p)]
    assert 'continuation_transform' in names and 'continuation_attest' in names
    assert 'structured_iterate' in names
    cpp=emit_cpp_program(p)
    assert '.supersede(' in cpp and '.begin_scope()' in cpp and '.accept(' in cpp
    assert 'decision_stop' in cpp and 'decision_reject' in cpp and 'decision_budget' in cpp


def test_decision_descriptor_refuses_incomplete_sum_and_transform_table():
    @graph(inputs=Product(base=Product(),proposal=Product(),candidate=Product()),
           output=Product(accepted=Bool,checks=Product()))
    def legacy(p,args):
        return p.product(accepted=p.boolean(True),checks=p.product())
    with pytest.raises(TypeError,match='exactly accept/reject/transform/stop'):
        DecisionAssess(legacy,{})
    decision=Variant(accept=Product(),reject=Product(controller_update=Product(),reason=Int64),
        transform=Variant(rescale=Float64),stop=Int64)
    @graph(inputs=legacy.inputs,output=decision)
    def complete(p,args):
        return p.variant(decision,'stop',p.int64(7))
    with pytest.raises(ValueError,match='exactly every Transform'):
        DecisionAssess(complete,{})


@pytest.mark.parametrize('origin',('trial_failure','transform_failure','assessment_failure'))
def test_nested_error_keeps_its_payload_and_origin_after_graph_deserialization(origin):
    errors=Failure(decision_reject=Product(code=Int64))
    @graph(inputs=Product(),output=Int64)
    def nested(p,args):
        p.branch(p.boolean(True),lambda q:q.require_success(q.variant(Result(Int64,errors),'err',
            q.variant(errors,'decision_reject',q.product(code=q.int64(9))))),lambda q:q.int64(0))
        return p.int64(7)
    restored=GraphFunction.from_data(nested.to_data(),inputs=nested.inputs,output=nested.output)
    for function in (nested,restored):
        p=Program('origin_check')
        result=p.capture(lambda q:_origin_checked(q,lambda b:function(b),origin))
        outer=dict(dict(type_of(result).signature['members'])['err']['members'])
        assert set(outer)=={origin,'runtime'}
        inner=dict(outer[origin]['members'])
        assert set(inner)=={'decision_reject','runtime'}
        assert dict(inner['decision_reject']['members'])['code']['vtype']=='int64'


@pytest.mark.parametrize('mode',('nested_success','nested_reject'))
def test_closed_trial_composes_two_authenticated_child_advances(mode):
    spec=build_decision_assess_v14(mode=mode);p=spec.case._time
    assert spec.case.validate() and not p._block_indices()
    operations=list(all_ops(p))
    assert sum(v.op=='continuation_consume_increment' for v in operations)>=4
    assert any(v.op=='graph_argument' for v in operations) is False
    cpp=emit_cpp_program(p)
    assert '.consume_increment(' in cpp and 'trial_failure' in cpp
