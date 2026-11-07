from collections.abc import Mapping
import gc
import weakref
from pops.ledger_provenance import _packet_reference_scanner, _refs

import pytest
import pops
from pops.graph_functions import graph, GraphFunction
from pops.time.variant_types import Float64, Product, type_of
from pops.continuations import ContinuationSchema, Candidate
from pops.contributions import Contribution, Delivery, Increment
from pops.finite_contributions import FiniteInterval, LinearMap
from pops.ledger_provenance import bind_contract, validate_contract_operation, validate_program_contracts
from pops.time._program.structured_control import value_signature


def setup():
    p=pops.Program('ledger_test')
    initial=p.product(q=p.float64(0),history=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    proposal=p.product(h=p.float64(1))
    return p,base,old,proposal


def ledger(p,q,time,delta):
    @graph(inputs=Product(value=type_of(q)),output=type_of(q))
    def identity(g,a):return a['value']
    action=LinearMap.differential(identity,at=p.product(value=q),wrt='value')
    c=Contribution('J',p.float64(delta),FiniteInterval(time,time+1,'clock'))
    return Increment((c,),(Delivery(c,q,action),))


def freeze(p,base,old,proposal,q,increment,*,history=None,observations=None):
    after=p.product(q=q,history=old['history'] if history is None else history,
                    time=old['time']+proposal['h'],step=old['step']+1)
    return Candidate(p,base=base,proposal=proposal,continuation=after,increment=increment,
                     observations=p.product() if observations is None else observations,checks=p.product())


@pytest.mark.parametrize('mode',('old','new','mismatch','history','observations','zero','double'))
def test_terminal_recipient_not_global_reachability(mode):
    p,base,old,proposal=setup();q=old['q']
    first=ledger(p,q,old['time'],1);second=first.replace(first.contributions[0],
        Contribution('J',p.float64(2),first.contributions[0].interval))
    a=first.apply(p,q,name='old');b=second.apply(p,q,name='new')
    retain=a if mode in ('old','mismatch','history','observations') else b
    if mode=='zero':retain=a+0*b
    if mode=='double':retain=2*b
    packed=(first if mode=='old' else second).pack(p)
    kwargs=dict(history=b if mode=='history' else None,observations=p.product(new=b) if mode=='observations' else None)
    if mode in ('old','new'):
        freeze(p,base,old,proposal,retain,packed,**kwargs)
    else:
        with pytest.raises(ValueError,match='retained inventory'):
            freeze(p,base,old,proposal,retain,packed,**kwargs)


@pytest.mark.parametrize('crossed',(False,True))
def test_correlated_branch_pair(crossed):
    p,base,old,proposal=setup();q=old['q']
    first=ledger(p,q,old['time'],1);second=first.replace(first.contributions[0],
        Contribution('J',p.float64(2),first.contributions[0].interval))
    a=first.apply(p,q,name='old');b=second.apply(p,q,name='new')
    one,two=first.pack(p),second.pack(p)
    condition=old['step']>0
    pair=p.branch(condition,lambda g:g.product(value=a,ledger=two if crossed else one),
                  lambda g:g.product(value=b,ledger=one if crossed else two))
    if crossed:
        with pytest.raises(ValueError,match='different ledger'):
            freeze(p,base,old,proposal,pair['value'],pair['ledger'])
    else:freeze(p,base,old,proposal,pair['value'],pair['ledger'])


@pytest.mark.parametrize('multiply',(False,True))
def test_copied_contract_and_changed_computed_ref_are_rejected(multiply):
    p,base,old,proposal=setup();q=old['q'];inc=ledger(p,q,old['time'],2)
    new=inc.apply(p,q,name='new');foreign=2*new if multiply else q+1
    metadata=dict(new.attrs['contract'],computed=foreign)
    with pytest.raises(ValueError,match='arithmetic/control'):
        bind_contract(p,foreign,'inventory_application',metadata)


def test_missing_contract_rejected():
    p,base,old,proposal=setup();q=old['q'];inc=ledger(p,q,old['time'],2)
    a=inc.apply(p,q,name='new')
    with pytest.raises(TypeError,match='complete mandatory'):
        validate_contract_operation(a.inputs,{},value_signature(a))


def test_graph_roundtrip_rebinds_each_recipient():
    p,base,old,proposal=setup()
    @graph(inputs=Product(q=Float64,time=Float64),output=Product(value=Float64,ledger=Product(J=Product(value=Float64,interval=Product(begin=Float64,end=Float64)))))
    def update(g,a):
        q=a['q'];inc=ledger(g,q,a['time'],2)
        return g.product(value=inc.apply(g,q,name='new'),ledger=inc.pack(g))
    loaded=GraphFunction.from_data(update.to_data())
    first=loaded(p,q=old['q'],time=old['time'])
    other=loaded(p,q=old['history'],time=old['time'])
    freeze(p,base,old,proposal,first['value'],first['ledger'])
    with pytest.raises(ValueError,match='retained inventory'):
        freeze(p,base,old,proposal,first['value'],other['ledger'])


def test_raw_product_is_explicitly_uncertified():
    p,base,old,proposal=setup()
    candidate=freeze(p,base,old,proposal,old['q']+1,p.product(J=p.float64(2)))
    assert candidate.attrs['increment_proof_v15']['status']=='uncertified_user_product'

@pytest.mark.parametrize('mode',('exact','modified','partial','renamed'))
def test_packet_projection_reconstruction_keeps_mandatory_claim(mode):
    p,base,old,proposal=setup();q=old['q'];inc=ledger(p,q,old['time'],2)
    a=inc.apply(p,q,name='new');packed=inc.pack(p)
    if mode=='exact':packet=p.product(J=packed['J'])
    elif mode=='modified':packet=p.product(J=p.product(value=packed['J']['value']+1,interval=packed['J']['interval']))
    elif mode=='partial':packet=p.product(value=packed['J']['value'])
    else:packet=p.product(renamed=packed['J'])
    if mode=='exact':freeze(p,base,old,proposal,a,packet)
    else:
        with pytest.raises(ValueError,match='ledger packet'):
            freeze(p,base,old,proposal,a,packet)


class CountedMapping(Mapping):
    def __init__(self, values):
        self.data = dict(values)
        self.reads = 0
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, key):
        return self.data[key]
    def values(self):
        self.reads += 1
        return self.data.values()


@pytest.mark.parametrize('present', (False, True))
def test_shared_metadata_dag_is_scanned_once_without_excluding_any_key(present):
    p = pops.Program('packet_reference_dag')
    value = p.float64(1)
    leaf = CountedMapping({'unexpected_key': value if present else 1})
    nodes = [leaf]
    metadata = leaf
    for _ in range(32):
        metadata = CountedMapping({'signature': metadata, 'opaque_future_metadata': metadata})
        nodes.append(metadata)
    scan = _packet_reference_scanner()
    actual = scan(metadata)
    assert tuple(id(item) for item in actual) == ((id(value),) if present else ())
    assert all(node.reads == 1 for node in nodes)
    assert scan(metadata) is actual
    assert all(node.reads == 1 for node in nodes)


def test_ordered_global_refs_keep_multiplicity_while_prefilter_keeps_first_occurrences():
    p = pops.Program('packet_reference_order')
    a, b = p.float64(1), p.float64(2)
    shared = {'nested': [b, a, b]}
    metadata = [a, shared, shared, b]
    assert tuple(id(x) for x in _refs(metadata)) == tuple(map(id, (a,b,a,b,b,a,b,b)))
    assert tuple(id(x) for x in _packet_reference_scanner()(metadata)) == (id(a), id(b))


def test_scanner_retains_sources_only_until_local_query_is_released():
    p = pops.Program('packet_reference_lifetime')
    value = p.float64(1)
    source = CountedMapping({'ref': value})
    retained = weakref.ref(source)
    scan = _packet_reference_scanner()
    assert scan(source)[0] is value
    del source
    gc.collect()
    assert retained() is not None
    del scan
    gc.collect()
    assert retained() is None


def test_new_query_rechecks_changed_public_metadata():
    p = pops.Program('packet_reference_fresh_query')
    a, b = p.float64(1), p.float64(2)
    metadata = {'ref': a}
    assert _packet_reference_scanner()(metadata)[0] is a
    metadata['ref'] = b
    assert _packet_reference_scanner()(metadata)[0] is b


def test_cycles_fail_closed_and_failed_query_does_not_poison_later_scan():
    metadata = []
    metadata.append(metadata)
    scan = _packet_reference_scanner()
    with pytest.raises(ValueError, match='cyclic metadata'):
        scan(metadata)
    metadata.clear()
    assert scan(metadata) == ()


def test_arbitrary_shared_metadata_reference_still_reaches_delivery_guard(monkeypatch):
    from pops.ledger_provenance import _validate_packet_delivery
    import pops.codegen.typed_delivery as delivery
    p, base, old, proposal = setup()
    packet = ledger(p, old['q'], old['time'], 1).pack(p)
    value = p.float64(0)
    shared = {'future_metadata': (packet, packet)}
    value = p._replace_value(value, attrs=dict(value.attrs, unfamiliar=(shared, shared)))
    envelope = p.product(increment=value)
    class ReachedExactDelivery(Exception):
        pass
    def reached(*args, **kwargs):
        raise ReachedExactDelivery
    monkeypatch.setattr(delivery, 'delivered_frontiers', reached)
    with pytest.raises(ReachedExactDelivery):
        _validate_packet_delivery(envelope, {})


def test_same_ssa_id_in_different_programs_does_not_collapse_authorities():
    first, second = pops.Program('first_owner'), pops.Program('second_owner')
    a, b = first.float64(1), second.float64(1)
    assert a.id == b.id and a.prog is not b.prog
    actual = _packet_reference_scanner()({'refs': (a, b, a)})
    assert len(actual) == 2 and actual[0] is a and actual[1] is b
