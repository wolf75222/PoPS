"""Durable C12 composition regressions discovered by independent source probes.
No native execution is claimed by these tests.
"""

# Origin: independent test_ad_conversion.py
import pops
from pops.graph_functions import graph,GraphFunction
from pops.graph_derivatives import jvp
from pops.time.variant_types import Float64,Int64,Bool,Product,type_of
from pops.continuations import ContinuationSchema,Candidate
from pops.contributions import Contribution,Delivery,Increment,FiniteInterval,LinearMap
from pops.ledger_provenance import InventoryBinding,validate_program_contracts
from pops.inventory_decode import checked_inventory_decode
from test_inventory_decode_v15 import problem


def test_jvp_conversion_keeps_retained_value_defect_and_ledger_on_one_primal():
    contribution_type=Product(value=Float64,interval=Product(begin=Float64,end=Float64))
    output=Product(value=Float64,defect=Float64,ledger=Product(J=contribution_type,conversion_residual=contribution_type))
    @graph(inputs=Product(q=Float64,time=Float64),output=output)
    def convert(g,args):
        g,functions,bindings=problem(program=g,base_value=args['q'])
        recipient=bindings['recipient'];window=FiniteInterval(args['time'],args['time']+1,'physical')
        @graph(inputs=Product(value=type_of(recipient)),output=type_of(recipient))
        def identity(h,a):return a['value']
        action=LinearMap.differential(identity,at=g.product(value=recipient),wrt='value')
        flux=Contribution('J',g.float64(2),window)
        ledger=Increment((flux,),(Delivery(flux,recipient,action),))
        bindings['application']=ledger.apply(g,recipient,name='target')
        result=g.require_success(checked_inventory_decode(g,**bindings,**functions))
        retained=result['value'];defect=result['defect']
        residual=Contribution('conversion_residual',defect,window,kind='defect')
        final=Increment((flux,residual),(Delivery(flux,recipient,action),Delivery(residual,recipient,action)))
        packed=final.pack(g,bindings=(InventoryBinding(recipient,('q',),conversion=retained),))
        return g.product(value=retained,defect=defect,ledger=packed)
    derivative=jvp(convert)
    restored=GraphFunction.from_data(derivative.to_data())
    p=pops.Program('AD_conversion_ledger')
    initial=p.product(q=p.float64(1),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    out=restored(p,primal=p.product(q=old['q'],time=old['time']),
                tangent=p.product(q=p.float64(1),time=p.float64(0)))['value']
    candidate=Candidate(p,base=base,proposal=p.product(h=p.float64(1)),
        continuation=p.product(q=out['value'],time=old['time']+1,step=old['step']+1),
        increment=out['ledger'],observations=p.product(defect=out['defect']),checks=p.product())
    validate_program_contracts(p)
    assert candidate.attrs['increment_proof_v15']['status']=='bound_ledger'


# Origin: independent test_conversion_invocations.py
import pytest
import pops
from pops.graph_functions import graph
from pops.time.variant_types import Product,type_of
from pops.continuations import ContinuationSchema,Candidate
from pops.contributions import Contribution,Delivery,Increment,FiniteInterval,LinearMap
from pops.ledger_provenance import InventoryBinding,validate_program_contracts
from pops.inventory_decode import checked_inventory_decode
from pops.codegen.typed_delivery import delivered_frontiers
from test_inventory_decode_v15 import problem


@pytest.mark.parametrize('stale',(False,True))
def test_prior_loop_conversion_does_not_certify_current_ledger(stale):
    p=pops.Program('conversion_epoch')
    initial=p.product(q=p.float64(1),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    p,functions,bindings=problem(program=p,base_value=old['q'])
    recipient=bindings['recipient'];window=FiniteInterval(old['time'],old['time']+1,'physical')
    @graph(inputs=Product(value=type_of(recipient)),output=type_of(recipient))
    def identity(g,a):return a['value']
    action=LinearMap.differential(identity,at=p.product(value=recipient),wrt='value')
    def update(g,amount):
        flux=Contribution('J',amount,window)
        ledger=Increment((flux,),(Delivery(flux,recipient,action),))
        args=dict(bindings,application=ledger.apply(g,recipient,name='application'))
        checked=g.require_success(checked_inventory_decode(g,**args,**functions))
        value=checked['value'];defect=checked['defect']
        residual=Contribution('conversion_residual',defect,window,kind='defect')
        final=Increment((flux,residual),(Delivery(flux,recipient,action),Delivery(residual,recipient,action)))
        packet=final.pack(g,bindings=(InventoryBinding(recipient,('q',),conversion=value),))
        return g.product(value=value,fresh=value,ledger=packet)
    seed=update(p,p.float64(0))
    def body(g,carry,index):
        new=update(g,g.float64(index+1))
        return g.product(value=carry['fresh'] if stale else new['value'],fresh=new['fresh'],ledger=new['ledger'])
    result=p.fold(seed,2,body,max_steps=2)
    is_conversion=lambda v:v.op=='provenance_bind' and v.attrs.get('contract_kind')=='inventory_conversion'
    is_ledger=lambda v:v.op=='provenance_bind' and v.attrs.get('contract_kind')=='inventory_ledger'
    conversion=next(iter(delivered_frontiers(result,is_conversion,path=('value',))[0]))
    ledger=next(iter(delivered_frontiers(result,is_ledger,path=('ledger',))[0]))
    assert bool(conversion.invocation!=ledger.invocation) is stale
    def freeze():
        return Candidate(p,base=base,proposal=p.product(h=p.float64(1)),
            continuation=p.product(q=result['value'],time=old['time']+1,step=old['step']+1),
            increment=result['ledger'],observations=p.product(),checks=p.product())
    if stale:
        with pytest.raises(ValueError,match='different native loop invocations'):freeze()
    else:
        freeze();validate_program_contracts(p)


# Origin: independent test_current_schema_tamper.py
import pytest
from pops.graph_functions import graph,GraphFunction
from pops.time.variant_types import Float64,Product,type_of
from pops.time._program.structured_control import value_signature
from pops.time._program.versioned_values import validate_continuations
from test_current_dependencies_v15 import fixture,freeze
from pops.ledger_provenance import validate_program_contracts


@pytest.mark.parametrize('surface',('read','validator'))
def test_schema_cannot_drop_current_equations_but_retain_identity(surface):
    p,base,old,requirement=fixture()
    schema=dict(base.attrs['schema'],current_dependencies=())
    signature=dict(value_signature(base),schema=schema)
    forged=p._replace_value(base,attrs=dict(base.attrs,schema=schema,signature=signature))
    with pytest.raises(ValueError,match='declarations differ from their compiled identity'):
        if surface=='read':p.continuation_read(forged)
        else:validate_continuations(p)


def test_live_capability_type_with_current_equations_loads_and_calls():
    p,base,old,requirement=fixture()
    @graph(inputs=Product(base=type_of(base)),output=Float64)
    def read(g,args):return g.continuation_read(args['base'])['q']
    restored=GraphFunction.from_data(read.to_data(),inputs=read.inputs,output=read.output)
    q=restored(p,base=base)+1
    freeze(p,base,old,q,requirement.evaluate(p,current=p.product(q=q)))
    validate_program_contracts(p)


# Origin: independent test_adversarial_reload.py
import copy
import json
import pytest
from pops.graph_functions import graph, GraphFunction
from pops.graph_derivatives import jvp
from pops.time import Program
from pops.time.variant_types import Float64, Product
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_source_map import source_map
from test_provenance_codegen_v15 import _updated
from typed_program_support import typed_state


def entry(mapping, name):
    return next(pair for pair in mapping['$map'] if pair[0] == name)


@pytest.mark.parametrize('wrong', ('old', 'double_new'))
def test_loader_rejects_rewritten_input_and_computed_reference(wrong):
    @graph(inputs=Product(q=Float64), output=Product(foreign=Float64, updated=Float64))
    def equation(p, a):
        q=a['q']; other=q if wrong=='old' else 2*(q+1)
        return p.product(foreign=other, updated=_updated(p,q))
    snapshot=json.loads(json.dumps(equation.to_data()))
    packet=next(n for n in reversed(snapshot['nodes']) if n['op']=='product')
    wrong_ref=packet['inputs'][0]
    bind=next(n for n in snapshot['nodes'] if n['op']=='provenance_bind')
    assert wrong_ref < bind['id']
    bind['inputs']=[wrong_ref]
    contract=entry(bind['attrs'],'contract')[1]
    entry(contract,'computed')[1]={'$ref':wrong_ref}
    with pytest.raises(ValueError,match='arithmetic/control'):
        GraphFunction.from_data(snapshot)


def test_jvp_transports_primal_contract_but_tangent_does_not_inherit_it():
    @graph(inputs=Product(q=Float64), output=Float64)
    def equation(p,a):return _updated(p,a['q'])
    derivative=GraphFunction.from_data(json.loads(json.dumps(jvp(equation).to_data())))
    p=Program('independent_tangent');state=typed_state(p,'state',state_name='U');state.n
    answer=derivative(p,primal=p.product(q=p.float64(2)),tangent=p.product(q=p.float64(1)))
    from pops.inventory_decode import _projection
    tangent=_projection(answer['tangent'])
    def ancestors(value,seen=None):
        seen=set() if seen is None else seen
        if value.id in seen:return
        seen.add(value.id);yield value
        for child in value.inputs:yield from ancestors(child,seen)
    assert not any(n.op=='provenance_bind' for n in ancestors(tangent))
    assert any(n.op=='provenance_bind' for n in p._issued_values.values())
    p.record_scalar('tangent',tangent)
    p.commit(state.next,p.value('accepted',state.n,at=state.next.point))
    cpp=emit_cpp_program(p);mapping=source_map(cpp)
    assert mapping['complete'] and not mapping['unlocated']
    assert 'ctx.record_scalar("tangent"' in cpp




def test_context_trace_preserves_exact_same_base_path():
    from pops.ledger_provenance import _context_rows,_same_context_rows
    p=pops.Program('context_same_base_path')
    initial=p.product(q=p.float64(1),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    envelope=p.product(continuation=p.product(q=old['q']),increment=p.product())
    left=_context_rows(envelope,path=('continuation','q'),replacements={},choices=frozenset())
    right=_context_rows(envelope,path=('continuation','q'),replacements={},choices=frozenset())
    assert len(left)==1 and len(left[0][1])==1
    reference=next(iter(left[0][1]))
    assert reference.value_id==old.id and reference.projection==('q',)
    assert _same_context_rows(left,right,frozenset())
