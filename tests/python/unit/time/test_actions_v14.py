"""Action is a library of closed graphs with an unavoidable native final contract."""
from dataclasses import replace
import json
import pops
import pytest
from pops.actions import Action,ActionFailure
from pops.codegen.program_codegen import emit_cpp_program
from pops.graph_functions import graph,GraphFunction
from pops.time.variant_types import Bool,Product
from examples.api_v02.action_checks_v14 import (
    Data,Request,Budget,Target,Vec,EvidenceType,Termination,
    build_action_checks_v14,make_nilpotent_action,
)


def _arguments(p):
    return dict(data=p.product(matrix=p.product(a=p.float64(0),b=p.float64(1),c=p.float64(0),d=p.float64(0)),
        vector=p.product(x=p.float64(1),y=p.float64(3)),h=p.float64(.2)),
        accuracy=p.product(atol=p.float64(1e-12),rtol=p.float64(0),scale=p.float64(1),request=p.product()),
        budget=p.product(terms=p.int64(2)))


@pytest.mark.parametrize('mode',('exact','fabricated','budget','domain'))
def test_action_equations_control_and_provenance_survive_native_source_emission(mode):
    spec=build_action_checks_v14(mode);plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    source=emit_cpp_program(plan.time)
    assert not plan.blocks and 'fold count outside declared capacity' in source
    for code in ActionFailure:assert str(int(code)) in source
    data=plan.time.to_graph().to_data()
    wire=json.dumps(data)
    assert 'pops.action.v14' in wire and 'action_contract' in wire
    assert 'error_contract' in wire and 'claimed_error' in wire
    # The action is expanded into ordinary graph/region operations.
    assert all(value.op!='action' for value in plan.time._values)


def test_action_strict_graphs_and_roundtrip_target_authentication():
    with pytest.raises(TypeError,match='closed GraphFunction'):
        Action(lambda p,a:None,lambda p,a:None,lambda p,a:None)
    action=make_nilpotent_action()
    data=json.loads(json.dumps(action.to_data()))
    restored=Action(*(GraphFunction.from_data(data[name]) for name in ('target','body','error_contract')))
    assert restored.to_data()==action.to_data()
    assert make_nilpotent_action(dishonest=True).body.identity!=action.body.identity


@pytest.mark.parametrize('bad_part',('body','error_contract'))
def test_action_rejects_unchecked_bool_contract_or_body_atomically(bad_part):
    p=pops.Program('action_bad_contract');arguments=_arguments(p);action=make_nilpotent_action()
    inputs=Product(target=Target,data=Data,accuracy=Request,budget=Budget) if bad_part=='body' else Product(
        target=Target,data=Data,values=Vec,evidence=EvidenceType,termination=Termination,accuracy=Request,budget=Budget)
    @graph(inputs=inputs,output=Bool)
    def bad(q,args):return q.boolean(True)
    before=p._ir_hash()
    with pytest.raises(TypeError,match='typed Result'):
        replace(action,**{bad_part:bad}).evaluate(p,**arguments)
    assert p._ir_hash()==before


def test_native_target_formula_binding_changes_real_program_identity():
    first=build_action_checks_v14(coefficient=1);second=build_action_checks_v14(coefficient=2)
    plans=[pops.resolve(pops.validate(s.case),layout=s.layout) for s in (first,second)]
    assert plans[0].time._ir_hash()!=plans[1].time._ir_hash()
    assert emit_cpp_program(plans[0].time)!=emit_cpp_program(plans[1].time)
