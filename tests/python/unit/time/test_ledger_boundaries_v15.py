"""Closed ledger obligations remain exact across native loop executions and reload."""
import pytest
from test_ledger_provenance_v15 import setup,ledger,freeze
from pops.contributions import Contribution
from pops.time._program.versioned_values import validate_schema_data


@pytest.mark.parametrize("count",(0,1,2))
def test_carried_complete_application_and_ledger_are_same_invocation(count):
    p,base,old,proposal=setup();q=old['q'];time=old['time']
    initial=ledger(p,q,time,1)
    pair=p.product(value=initial.apply(p,q,name='initial'),ledger=initial.pack(p))
    def body(g,carry,index):
        inc=ledger(g,q,time,2)
        return g.product(value=inc.apply(g,q,name='current'),ledger=inc.pack(g))
    answer=p.fold(pair,count,body,max_steps=2)
    freeze(p,base,old,proposal,answer['value'],answer['ledger'])


def test_static_same_node_does_not_certify_previous_loop_invocation():
    p,base,old,proposal=setup();q=old['q'];time=old['time']
    initial=ledger(p,q,time,1)
    pair=p.product(value=initial.apply(p,q,name='initial'),ledger=initial.pack(p))
    def body(g,carry,index):
        inc=ledger(g,q,time,2)
        return g.product(value=carry['value'],ledger=inc.pack(g))
    answer=p.fold(pair,2,body,max_steps=2)
    with pytest.raises(ValueError,match='different native loop invocations|different ledger'):
        freeze(p,base,old,proposal,answer['value'],answer['ledger'])


def test_changed_second_interval_cannot_reuse_application_proof():
    from pops.ledger_provenance import bind_contract
    p,base,old,proposal=setup();q=old['q']
    one=ledger(p,q,old['time'],1)
    two=Contribution('K',p.float64(2),one.contributions[0].interval)
    from pops.contributions import Increment,Delivery
    both=Increment((*one.contributions,two),(*one.deliveries,Delivery(two,q,one.deliveries[0].linear_map)))
    applied=both.apply(p,q,name='two')
    contract=dict(applied.attrs['contract']);data=dict(contract['ledger'])
    data['contributions']=(data['contributions'][0],dict(data['contributions'][1],end=p.float64(-1)))
    contract['ledger']=data
    with pytest.raises(ValueError,match='exact declared interval'):
        bind_contract(p,applied.inputs[0],'inventory_application',contract)


@pytest.mark.parametrize('field',('current_dependencies','scientific','roles'))
def test_changed_schema_declaration_cannot_keep_identity(field):
    p,base,old,proposal=setup()
    from pops.time._program.structured_control import value_signature
    data=dict(value_signature(base)['schema'])
    data[field]=({'unexpected':True},)
    with pytest.raises(ValueError,match='compiled identity'):
        validate_schema_data(data)


@pytest.mark.parametrize('mode',('valid','crossed','duplicate'))
def test_two_nonlinear_recipients_keep_their_own_exact_defect(mode):
    from test_inventory_decode_v15 import problem
    from pops.inventory_decode import checked_inventory_decode
    from pops.contributions import Contribution,Delivery,Increment,FiniteInterval,LinearMap
    from pops.graph_functions import graph
    from pops.time.variant_types import Product,type_of
    from pops.ledger_provenance import InventoryBinding
    p,base,old,proposal=setup()
    entries=[]
    for name in ('q','history'):
        _,functions,args=problem(program=p,base_value=old[name])
        recipient=args['recipient']
        @graph(inputs=Product(value=type_of(recipient)),output=type_of(recipient))
        def identity(g,a):return a['value']
        action=LinearMap.differential(identity,at=p.product(value=recipient),wrt='value')
        entries.append((name,functions,args,action))
    window=FiniteInterval(old['time'],old['time']+1,'physical')
    shared=Contribution('J',p.float64(2),window)
    original=Increment((shared,),tuple(Delivery(shared,args['recipient'],action) for _,f,args,action in entries))
    results=[]
    for name,f,args,action in entries:
        args['application']=original.apply(p,args['recipient'],name='application_'+name)
        results.append(p.require_success(checked_inventory_decode(p,**args,**f)))
    defects=[Contribution('R_'+name,results[i]['defect'],window,kind='defect')
             for i,(name,f,args,action) in enumerate(entries)]
    deliveries=list(original.deliveries)
    deliveries.extend(Delivery(defects[i],entries[1-i if mode=='crossed' else i][2]['recipient'],entries[i][3])
                      for i in range(2))
    if mode=='duplicate':
        defects.append(Contribution('R_duplicate',results[0]['defect'],window,kind='defect'))
        deliveries.append(Delivery(defects[-1],entries[0][2]['recipient'],entries[0][3]))
    final=Increment((shared,*defects),tuple(deliveries))
    packet=final.pack(p,bindings=tuple(InventoryBinding(args['recipient'],(name,),conversion=results[i]['value'])
                                     for i,(name,f,args,action) in enumerate(entries)))
    def construct():return freeze(p,base,old,proposal,results[0]['value'],packet,history=results[1]['value'])
    if mode=='valid':construct()
    else:
        with pytest.raises(ValueError,match='conversion defect'):
            construct()


def test_context_trace_distinguishes_exact_current_base_paths():
    from pops.ledger_provenance import _context_rows,_same_context_rows
    p,base,old,proposal=setup();q=old['q'];other=old['history']
    inc=ledger(p,q,old['time'],2);packet=inc.pack(p)
    envelope=p.product(increment=packet)
    left=_context_rows(envelope,path=('increment',),replacements={packet.id:q},choices=frozenset())
    right=_context_rows(envelope,path=('increment',),replacements={packet.id:other},choices=frozenset())
    assert left and right and all(refs for _,refs in (*left,*right))
    assert not _same_context_rows(left,right,frozenset())


def test_opaque_iteration_schema_scan_does_not_grant_structured_transport():
    import pops
    from pops.time._program.structured_control import value_signature
    from pops.time._program.versioned_values import validate_continuations
    p=pops.Program('opaque_schema_scan')
    result=p.iterate(p.float64(0),lambda g,x:x,lambda g,x:g.boolean(True),max_steps=0)
    assert result.vtype=='iteration_result'
    validate_continuations(p)
    with pytest.raises(TypeError,match='structured control cannot carry'):
        value_signature(result)


@pytest.mark.parametrize('key',('signature','state_signature','success_signature','error_signature'))
def test_actual_structural_metadata_cannot_hide_a_stale_capability_schema(key):
    import pops
    from pops.time._program.structured_control import value_signature
    from pops.time._program.versioned_values import validate_continuations
    _,base,_,_=setup()
    capability=dict(value_signature(base));schema=dict(capability['schema'])
    schema['roles']=({'changed':True},)
    capability['schema']=schema
    p=pops.Program('stale_schema_metadata')
    value=p.float64(0)
    p._replace_value(value,attrs=dict(value.attrs,**{key:{'vtype':'product','members':(('token',capability),)}}))
    with pytest.raises(ValueError,match='compiled identity'):
        validate_continuations(p)


def test_native_capability_without_its_structural_signature_is_rejected():
    from pops.time._program.versioned_values import validate_continuations
    p,base,_,_=setup()
    attrs=dict(base.attrs);del attrs['signature']
    p._replace_value(base,attrs=attrs)
    with pytest.raises(KeyError,match='signature'):
        validate_continuations(p)
