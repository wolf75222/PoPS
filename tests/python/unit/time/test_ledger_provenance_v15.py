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
