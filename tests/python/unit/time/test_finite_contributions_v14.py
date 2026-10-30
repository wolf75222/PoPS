"""Finite ledger syntax/type/graph checks; runtime evidence lives in native tests."""
import pytest
import pops
from pops.contributions import Contribution, Delivery, Increment, LinearMap, FiniteInterval
from pops.graph_functions import graph
from pops.time.variant_types import Product,type_of
from pops.codegen.typed_delivery import delivered_frontiers


def fixture():
    p=pops.Program('ledger')
    d=p.range_set(0,capacity=8,ownership='replicated')
    q=p.map(d,lambda g,i:g.float64(g.index_value(i))+1)
    @graph(inputs=Product(value=type_of(q)),output=type_of(q))
    def identity(g,args): return args['value']
    action=LinearMap.differential(identity,at=p.product(value=q),wrt='value')
    window=FiniteInterval(p.float64(0),p.float64(1),'time')
    c=Contribution('c',q,window)
    return p,q,c,action


def test_empty_array_application_checks_exact_domains_and_retains_window():
    p,q,c,action=fixture()
    inc=Increment((c,),(Delivery(c,q,action),))
    out=inc.apply(p,q,name='update')
    assert inc.require_application(out) is out
    assert any(v.op=='finite_same_domain' for v in p._values)
    packet=c.pack(p)
    restored=Contribution.from_history('previous',packet,clock='time')
    assert restored.value.op=='product_get' and restored.interval.begin.op=='product_get'


def test_replacement_rejects_old_result_and_duplicate_delivery():
    p,q,c,action=fixture()
    inc=Increment((c,),(Delivery(c,q,action),)); old=inc.apply(p,q,name='old')
    new=Contribution('corrected',p.map(p.domain(q),lambda g,i:g.at(q,i)*2),c.interval)
    updated=inc.replace(c,new)
    with pytest.raises(ValueError,match='exact increment version'):
        updated.require_application(old)
    updated.require_application(updated.apply(p,q,name='new'))
    with pytest.raises(ValueError,match='duplicate delivery'):
        Increment((c,),(Delivery(c,q,action),Delivery(c,q,action)))


def test_action_is_generated_differential_not_user_supplied_linearity_claim():
    p,q,c,action=fixture()
    with pytest.raises(TypeError):
        LinearMap(action.function,action.at,'value',action=action.function)
    with pytest.raises(TypeError,match='LinearMap'):
        Delivery(c,q,action.function)


@pytest.mark.parametrize('count',(0,1,2))
def test_map_internal_use_is_delivered_element_by_element(count):
    # IR19: a map delivers its body element by element, as one element-wise
    # evaluation whatever its extent; an empty extent covers its empty term (D5).
    from pops.codegen.typed_delivery import EACH
    p=pops.Program('owned_map'); d=p.range_set(count,capacity=8,ownership='replicated'); owned=[]
    def body(g,i):
        value=g.float64(g.index_value(i)); owned.append(value); return value
    out=p.map(d,body)
    frontiers=delivered_frontiers(out,lambda node:node is owned[0])
    assert [[ref.invocation for ref in frontier] for frontier in frontiers]==[[((out.id,EACH,'map'),)]]


def test_domain_diagnostic_cannot_consume_array_inventory():
    p,q,_,_=fixture()
    diagnostic=p.float64(p.size(p.domain(q)))
    assert delivered_frontiers(diagnostic,lambda value:value is q)==(frozenset(),)


@pytest.mark.parametrize('mode',('success','wrong_domain','negative_history','parent_reject'))
def test_complete_bdf2_bundle_resolves_without_physical_carrier(mode):
    from examples.api_v02.finite_bdf2_v14 import build_finite_bdf2_v14
    spec=build_finite_bdf2_v14(mode=mode)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert not plan.blocks and not spec.initial_values
