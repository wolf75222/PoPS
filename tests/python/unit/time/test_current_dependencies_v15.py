"""Schema-enforced retained current equations; tests invoke the actual freeze."""
import json
import pytest
from pops.time import Program
from pops.time.variant_types import Float64, Int64, Product, type_of
from pops.graph_functions import graph, GraphFunction
from pops.continuations import Candidate, ContinuationSchema
from pops.current_dependencies import CurrentDependency
from pops.ledger_provenance import validate_program_contracts, bind_contract


@graph(inputs=Product(q=Float64),output=Float64)
def twice(p,a):return 2*a['q']


def fixture():
    p=Program('current_dependency')
    initial=p.product(q=p.float64(1),z=p.float64(2),history=p.float64(1),time=p.float64(0),step=p.int64(0))
    requirement=CurrentDependency(twice,target=('z',),inputs={'q':('q',)})
    schema=ContinuationSchema(initial,time=('time',),step=('step',),current_dependencies=(requirement,))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    return p,base,old,requirement


def freeze(p,base,old,q,z,observations=None):
    return Candidate(p,base=base,proposal=p.product(h=p.float64(1)),
        continuation=p.product(q=q,z=z,history=old['history'],time=old['time']+1,step=old['step']+1),
        increment=p.product(),observations=p.product() if observations is None else observations,checks=p.product())


@pytest.mark.parametrize('mode',('recomputed','stale','omitted','old_stage'))
def test_schema_dependency_is_mandatory_but_old_stages_remain_legal(mode):
    p,base,old,requirement=fixture();oldq=old['q']
    q=oldq+1 if mode!='old_stage' else oldq+(oldq*2-oldq)
    current=p.product(q=oldq if mode=='stale' else q)
    z=requirement.evaluate(p,current=current) if mode!='omitted' else q*2
    if mode in ('stale','omitted'):
        with pytest.raises(ValueError,match='current dependency'):
            freeze(p,base,old,q,z)
    else:
        freeze(p,base,old,q,z,observations=p.product(previous=oldq*2))
        validate_program_contracts(p)


@pytest.mark.parametrize('crossed',(False,True))
def test_dependency_follows_correlated_current_record_branches(crossed):
    p,base,old,requirement=fixture();oldq=old['q'];a=oldq+1;b=oldq+2
    za=requirement.evaluate(p,current=p.product(q=a))
    zb=requirement.evaluate(p,current=p.product(q=b))
    pair=p.branch(old['step']>0,lambda g:g.product(q=a,z=zb if crossed else za),
                  lambda g:g.product(q=b,z=za if crossed else zb))
    if crossed:
        with pytest.raises(ValueError,match='current dependency'):
            freeze(p,base,old,pair['q'],pair['z'])
    else:freeze(p,base,old,pair['q'],pair['z'])


def test_copying_dependency_metadata_and_rewriting_computed_is_not_a_proof():
    p,base,old,requirement=fixture();q=old['q']+1
    z=requirement.evaluate(p,current=p.product(q=q));wrong=z.inputs[0]+1
    metadata=dict(z.attrs['contract'],computed=wrong)
    with pytest.raises(ValueError,match='arithmetic/control'):
        bind_contract(p,wrong,'current_dependency',metadata)


def test_same_type_previous_paths_are_not_interchangeable():
    p,base,old,requirement=fixture();q=old['q']
    z=requirement.evaluate(p,current=p.product(q=old['history']))
    with pytest.raises(ValueError,match='current dependency'):
        freeze(p,base,old,q,z)


@pytest.mark.parametrize('count',(0,1,2))
def test_current_pair_remains_bound_through_loop_carry(count):
    p,base,old,requirement=fixture();q=old['q']
    pair=p.product(q=q,z=requirement.evaluate(p,current=p.product(q=q)))
    def body(g,carry,index):
        q=carry['q']+1
        return g.product(q=q,z=requirement.evaluate(g,current=g.product(q=q)))
    answer=p.fold(pair,count,body,max_steps=2)
    freeze(p,base,old,answer['q'],answer['z'])


def test_closed_requirement_reloads_and_rebinds_on_two_calls():
    p,base,old,requirement=fixture()
    @graph(inputs=Product(q=Float64),output=Product(q=Float64,z=Float64))
    def update(g,a):
        q=a['q']+1
        return g.product(q=q,z=requirement.evaluate(g,current=g.product(q=q)))
    loaded=GraphFunction.from_data(json.loads(json.dumps(update.to_data())))
    one=loaded(p,q=old['q']);two=loaded(p,q=old['history'])
    freeze(p,base,old,one['q'],one['z'])
    with pytest.raises(ValueError,match='current dependency'):
        freeze(p,base,old,one['q'],two['z'])


def test_two_named_arguments_can_share_one_exact_current_path_after_json_reload():
    @graph(inputs=Product(left=Float64,right=Float64),output=Float64)
    def sum_two(g,a):return a['left']+a['right']
    loaded=GraphFunction.from_data(json.loads(json.dumps(sum_two.to_data())))
    requirement=CurrentDependency(loaded,target=('z',),inputs={'left':('q',),'right':('q',)})
    p=Program('aliased_current_arguments');initial=p.product(q=p.float64(1),z=p.float64(2),history=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',),current_dependencies=(requirement,))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base);q=old['q']+1
    freeze(p,base,old,q,requirement.evaluate(p,current=p.product(q=q)))


@pytest.mark.parametrize('crossed',(False,True))
def test_variant_injections_with_identical_payload_are_not_interchangeable(crossed):
    from pops.time.variant_types import Variant
    tags=Variant(left=Float64,right=Float64)
    @graph(inputs=Product(q=tags),output=Float64)
    def value(g,a):return g.match(a['q'],left=lambda h,x:2*x,right=lambda h,x:3*x)
    requirement=CurrentDependency(value,target=('z',),inputs={'q':('q',)})
    p=Program('variant_current');payload=p.float64(1)
    a=p.variant(tags,'left',payload);b=p.variant(tags,'right',payload)
    initial=p.product(q=a,z=p.float64(2),history=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',),current_dependencies=(requirement,))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    z=requirement.evaluate(p,current=p.product(q=a))
    if crossed:
        with pytest.raises(ValueError,match='current dependency'):
            freeze(p,base,old,b,z)
    else:freeze(p,base,old,a,z)


def test_loop_carry_cannot_keep_previous_iterate_as_current_dependency():
    p,base,old,requirement=fixture();q=old['q']
    pair=p.product(q=q,z=requirement.evaluate(p,current=p.product(q=q)))
    def body(g,carry,index):
        q=carry['q']
        return g.product(q=q+1,z=requirement.evaluate(g,current=g.product(q=q)))
    answer=p.fold(pair,2,body,max_steps=2)
    with pytest.raises(ValueError,match='current dependency'):
        freeze(p,base,old,answer['q'],answer['z'])


def test_schema_requirement_and_exact_argument_refs_survive_program_detach():
    from pops.time._program.detach import detach_compiled_program
    p,base,old,requirement=fixture();q=old['q']+1
    freeze(p,base,old,q,requirement.evaluate(p,current=p.product(q=q)))
    before=p.to_graph().to_data()
    detached=detach_compiled_program(p)
    assert detached.to_graph().to_data()==before
    assert detached._ir_hash()==p._ir_hash()
    validate_program_contracts(detached)
    bound=next(v for v in detached._issued_values.values() if v.op=='continuation_open')
    from pops.time._program.structured_control import value_signature
    declarations=value_signature(bound)['schema']['current_dependencies']
    assert len(declarations)==1 and tuple(declarations[0]['target'])==('z',)
    marker=next(v for v in detached._issued_values.values() if v.op=='provenance_bind' and v.attrs['contract_kind']=='current_dependency')
    assert marker.attrs['contract']['input_values']['q'].prog is detached
