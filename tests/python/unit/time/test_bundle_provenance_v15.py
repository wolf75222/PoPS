"""Source-only composition of mandatory provenance through exact Bundle transports."""
import json
import pytest
import pops
from pops.bundles import BundleSchema,Remap,remap
from pops.continuations import ContinuationSchema,Candidate
from pops.contributions import Contribution,Delivery,Increment,FiniteInterval,LinearMap
from pops.ledger_provenance import InventoryBinding,validate_program_contracts
from pops.current_dependencies import CurrentDependency
from pops.graph_functions import graph,GraphFunction
from pops.time.variant_types import Float64,Product,Result,Failure,type_of
from pops.codegen.typed_delivery import delivered_frontiers


def raw(p,d,q,r,z,history,source=None):
    return p.product(instance=d,fields=p.product(q=q,r=r,z=z),geometry=p.product(),
        maps=p.product(source=d if source is None else source),histories=p.product(q=history))


def setup(*,dependency=False):
    p=pops.Program('bundle_provenance')
    d=p.range_set(2,capacity=4,ownership='replicated')
    q=p.map(d,lambda g,i:g.float64(g.index_value(i))+1)
    r=p.map(d,lambda g,i:g.float64(g.index_value(i))+4)
    z=p.map(d,lambda g,i:2*g.at(q,i))
    proto=raw(p,d,q,r,z,q)
    bs=BundleSchema(proto,domains={'current':('instance',),'source':('maps','source')},
        bindings={('fields','q'):'current',('fields','r'):'current',('fields','z'):'current',('histories','q'):'current'})
    requirement=None
    if dependency:
        @graph(inputs=Product(q=type_of(q)),output=type_of(q))
        def twice(g,a):return g.map(g.domain(a['q']),lambda b,i:2*b.at(a['q'],i))
        requirement=CurrentDependency(twice,target=('bundle','fields','z'),inputs={'q':('bundle','fields','q')})
    initial=p.product(bundle=bs.bind(p,proto),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',),
        current_dependencies=() if requirement is None else (requirement,))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    return p,bs,schema,base,old,requirement


def ledger(p,q,time,amount):
    @graph(inputs=Product(value=type_of(q)),output=type_of(q))
    def identity(g,a):return a['value']
    action=LinearMap.differential(identity,at=p.product(value=q),wrt='value')
    delta=p.map(p.domain(q),lambda g,i:g.float64(amount))
    c=Contribution('J',delta,FiniteInterval(time,time+1,'clock'))
    return Increment((c,),(Delivery(c,q,action),))


def freeze(p,bs,base,old,q,packet,*,r=None,z=None,history=None):
    b=old['bundle'];r=b['fields']['r'] if r is None else r;z=b['fields']['z'] if z is None else z
    history=b['histories']['q'] if history is None else history
    bundle=bs.bind(p,raw(p,b['instance'],q,r,z,history,b['maps']['source']))
    after=p.product(bundle=bundle,time=old['time']+1,step=old['step']+1)
    return Candidate(p,base=base,proposal=p.product(h=p.float64(1)),continuation=after,
                     increment=packet,observations=p.product(),checks=p.product())


@pytest.mark.parametrize('mode',('old','new','stale','hidden_history','wrong_coordinate','modified'))
def test_exact_bundle_coordinate_requires_its_current_application(mode):
    p,bs,schema,base,old,_=setup();q=old['bundle']['fields']['q'];t=old['time']
    a=ledger(p,q,t,1);b=ledger(p,q,t,2);qa=a.apply(p,q,name='old');qb=b.apply(p,q,name='new')
    retained=qa if mode in ('old','stale','hidden_history') else qb
    target=('bundle','fields','r') if mode=='wrong_coordinate' else ('bundle','fields','q')
    if mode=='modified':retained=p.map(p.domain(qb),lambda g,i:2*g.at(qb,i))
    packet=(a if mode=='old' else b).pack(p,bindings=(InventoryBinding(q,target),))
    if mode in ('old','new'):
        freeze(p,bs,base,old,retained,packet);validate_program_contracts(p)
    else:
        with pytest.raises(ValueError,match='retained inventory|exact candidate base recipient'):
            freeze(p,bs,base,old,retained,packet,history=qb if mode=='hidden_history' else None)


@pytest.mark.parametrize('mode',('current','stale','wrong_coordinate'))
def test_current_dependency_is_checked_through_bundle(mode):
    p,bs,schema,base,old,requirement=setup(dependency=True);b=old['bundle'];oq=b['fields']['q']
    q=p.map(p.domain(oq),lambda g,i:g.at(oq,i)+1)
    arg=oq if mode=='stale' else b['fields']['r'] if mode=='wrong_coordinate' else q
    current=p.product(bundle=p.product(fields=p.product(q=arg)))
    z=requirement.evaluate(p,current=current)
    if mode=='current':freeze(p,bs,base,old,q,p.product(),z=z);validate_program_contracts(p)
    else:
        with pytest.raises(ValueError,match='current dependency'):
            freeze(p,bs,base,old,q,p.product(),z=z)


def test_bundle_application_survives_closed_graph_json_reload_and_detach():
    from pops.time._program.detach import detach_compiled_program
    p,bs,schema,base,old,_=setup();q=old['bundle']['fields']['q']
    prototype=ledger(p,q,old['time'],2).pack(p,bindings=(InventoryBinding(q,('bundle','fields','q')),))
    @graph(inputs=Product(old=schema.read_type),output=Product(bundle=bs.value_type,packet=type_of(prototype)))
    def update(g,a):
        before=a['old'];b=before['bundle'];q=b['fields']['q'];inc=ledger(g,q,before['time'],2)
        value=inc.apply(g,q,name='new')
        bundle=bs.bind(g,raw(g,b['instance'],value,b['fields']['r'],b['fields']['z'],q,b['maps']['source']))
        packet=inc.pack(g,bindings=(InventoryBinding(q,('bundle','fields','q')),))
        return g.product(bundle=bundle,packet=packet)
    loaded=GraphFunction.from_data(json.loads(json.dumps(update.to_data())),inputs=update.inputs,output=update.output)
    result=loaded(p,old=old)
    Candidate(p,base=base,proposal=p.product(h=p.float64(1)),
        continuation=p.product(bundle=result['bundle'],time=old['time']+1,step=old['step']+1),
        increment=result['packet'],observations=p.product(),checks=p.product())
    validate_program_contracts(p)
    detached=detach_compiled_program(p);validate_program_contracts(detached)
    assert detached._ir_hash()==p._ir_hash()


def test_remap_delivers_only_candidate_coordinate_not_old_guard_or_other_path():
    p=pops.Program('remap_frontier');d=p.range_set(2,capacity=4,ownership='replicated')
    oldq=p.float64(111);proto=raw(p,d,oldq,p.float64(7),p.float64(9),oldq)
    bs=BundleSchema(proto,domains={'current':('instance',),'source':('maps','source')},bindings={})
    old=bs.bind(p,proto)
    @graph(inputs=Product(bundle=bs.value_type,data=Product()),output=Result(bs.value_type,Failure()))
    def transfer(g,a):
        b=a['bundle'];newd=g.range_set(g.size(b['instance']),family=b['instance'])
        candidate=bs.bind(g,raw(g,newd,g.float64(222),g.float64(7),g.float64(9),b['fields']['q'],b['instance']))
        return g.variant(Result(bs.value_type,Failure()),'ok',candidate)
    loaded=GraphFunction.from_data(json.loads(json.dumps(transfer.to_data())),inputs=transfer.inputs,output=transfer.output)
    answer=p.require_success(remap(p,Remap(loaded,source_instance=('maps','source'),histories={'q':'transfer'}),old,data=p.product()))
    nodes={v.id:v for v in p._issued_values.values()}
    def marker(v):return v.op=='finite_real' and v.attrs.get('value') in (111.,222.)
    fields=delivered_frontiers(answer,marker,path=('fields','q'))
    histories=delivered_frontiers(answer,marker,path=('histories','q'))
    seen=lambda rows:{nodes[ref.value_id].attrs['value'] for row in rows for ref in row}
    assert seen(fields)=={222.}
    assert seen(histories)=={111.}
    assert seen(delivered_frontiers(answer,marker,path=('maps','source')))==set()


@pytest.mark.parametrize('changed',(False,True))
def test_rebound_bundle_recipient_projection_keeps_exact_base_coordinate(changed):
    p,bs,schema,base,old,_=setup();b=old['bundle'];q=b['fields']['q']
    current=p.map(p.domain(q),lambda g,i:g.at(q,i)+1) if changed else q
    rebound=bs.bind(p,raw(p,b['instance'],current,b['fields']['r'],b['fields']['z'],q,b['maps']['source']))
    recipient=rebound['fields']['q'];inc=ledger(p,recipient,old['time'],2)
    value=inc.apply(p,recipient,name='new')
    packed=inc.pack(p,bindings=(InventoryBinding(recipient,('bundle','fields','q')),))
    if changed:
        with pytest.raises(ValueError,match='exact candidate base recipient'):
            freeze(p,bs,base,old,value,packed)
    else:freeze(p,bs,base,old,value,packed);validate_program_contracts(p)


@pytest.mark.parametrize('certified',(False,True))
def test_distributed_reduction_is_plain_data_unless_it_strips_a_certified_packet(certified):
    from pops.time._program.distributed_arrays import Distributed
    from pops.time.variant_types import Int64
    p=pops.Program('uncertified_reduction')
    initial=p.product(q=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    @graph(inputs=Product(value=Float64),output=Float64)
    def identity(g,a):return a['value']
    c=Contribution('J',p.float64(1),FiniteInterval(old['time'],old['time']+1,'clock'))
    inc=Increment((c,),(Delivery(c,old['q'],LinearMap.differential(identity,at=p.product(value=old['q']),wrt='value')),))
    packet=inc.pack(p)
    # The numerical sum is legitimate user data. Taking its input out of a
    # certified ledger, however, must not launder that ledger into a plain sum.
    scalar=packet['J']['value'] if certified else p.float64(1)
    dd=Distributed(p,storage_bytes=65536,storage_versions=32)
    domain=dd.domain((0,1),capacity=1)
    array=p.require_success(dd.local_map(domain,
        lambda g,i:g.variant(Result(Float64,Int64),'ok',scalar),byte_capacity=4096))
    total=dd.reduce(array,p.float64(0),lambda g,a,b:a+b,
        order='global_gid_ascending',byte_capacity=4096)
    def candidate():
        return Candidate(p,base=base,proposal=p.product(h=p.float64(1)),
            continuation=p.product(q=old['q'],time=old['time']+1,step=old['step']+1),
            increment=p.product(sum=total),observations=p.product(),checks=p.product())
    if certified:
        with pytest.raises(ValueError,match='ledger packet|delivery rule'):
            candidate()
    else:candidate();validate_program_contracts(p)


@pytest.mark.parametrize('mode',('intact','uncertified_arithmetic','declared_stale','child_stale'))
def test_consumed_child_proof_does_not_silently_certify_parent_arithmetic(mode):
    from pops.time.variant_types import Bool
    p,bs,schema,base,old,_=setup();q=old['bundle']['fields']['q']
    first=ledger(p,q,old['time'],1);second=ledger(p,q,old['time'],2)
    qa=first.apply(p,q,name='child_application')
    packet=(second if mode=='child_stale' else first).pack(p,
        bindings=(InventoryBinding(q,('bundle','fields','q')),))
    if mode=='child_stale':
        with pytest.raises(ValueError,match='different ledger'):
            freeze(p,bs,base,old,qa,packet)
        return
    child=freeze(p,bs,base,old,qa,packet)
    assert child.attrs['increment_proof_v15']['status']=='bound_ledger'
    @graph(inputs=Product(base=schema.read_type,proposal=Product(h=Float64),
        candidate=type_of(p.continuation_candidate_value(child))),
        output=Product(accepted=Bool,checks=Product()))
    def assess(g,a):return g.product(accepted=g.boolean(True),checks=g.product())
    accepted=p.continuation_accept(child,p.continuation_attest(child,assess))
    consumed=p.continuation_consume_increment(accepted)
    parent_packet=consumed
    if mode=='uncertified_arithmetic':
        changed=p.map(p.domain(consumed['J']['value']),lambda g,i:2*g.at(consumed['J']['value'],i))
        parent_packet=p.product(sum=changed)
    if mode=='declared_stale':
        # A new declaration cannot borrow the child proof for a different update.
        parent_packet=second.pack(p,bindings=(InventoryBinding(q,('bundle','fields','q')),))
        with pytest.raises(ValueError,match='different ledger'):
            freeze(p,bs,base,old,qa,parent_packet)
    else:
        parent=freeze(p,bs,base,old,qa,parent_packet)
        assert parent.attrs['increment_proof_v15']['status']=='uncertified_user_product'
        assert parent.attrs['increment_proof_v15']['recipient_obligations']==0
        validate_program_contracts(p)


def test_bundle_inventory_native_profile_elaborates_complete_real_program():
    from examples.api_v02.bundle_inventory_v15 import build_bundle_inventory_v15
    from pops.codegen.program_codegen import emit_cpp_program
    spec=build_bundle_inventory_v15()
    assert not spec.initial_values and not spec.case._time._block_indices()
    assert spec.case.validate()
    cpp=emit_cpp_program(spec.case._time)
    assert 'continuation_publish(' in cpp and 'new_instance == old_instance' in cpp



def test_plain_product_skips_inventory_delivery_but_keeps_current_equations(monkeypatch):
    import pops.codegen.typed_delivery as delivery
    p,bs,schema,base,old,_=setup()
    def forbidden(*args,**kwargs):raise AssertionError('uncertified product invoked delivery analysis')
    with monkeypatch.context() as m:
        m.setattr(delivery,'delivered_frontiers',forbidden)
        candidate=freeze(p,bs,base,old,old['bundle']['fields']['q'],p.product(delta=p.float64(1)))
    assert candidate.attrs['increment_proof_v15']['status']=='uncertified_user_product'
    p,bs,schema,base,old,requirement=setup(dependency=True)
    q=p.map(p.domain(old['bundle']['fields']['q']),lambda g,i:g.at(old['bundle']['fields']['q'],i)+1)
    stale=requirement.evaluate(p,current=p.product(bundle=p.product(fields=p.product(q=old['bundle']['fields']['q']))))
    with pytest.raises(ValueError,match='current dependency'):
        freeze(p,bs,base,old,q,p.product(delta=p.float64(1)),z=stale)
