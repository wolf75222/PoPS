"""Dependent finite remap with real ownership changes in a fixed communicator.

Four cells are permuted and redistributed between endpoint ranks. Fields,
geometry, Index maps and two-level recurrence history move through native imports.
Changing the communicator size on restart is a separate backend capability.
"""
import pops
from pops.bundles import BundleSchema, Remap, remap
from pops.continuations import ContinuationSchema, Advance, Assess, advance_once
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time._program.distributed_arrays import Distributed
from pops.time._program.structured_control import value_signature
from pops.time.variant_types import Bool,Float64,Int64,Product,Variant,Result,Failure,RuntimeFailure,type_of
from .common import CaseSpec
from .models_M01_M14 import _domain

CAP=262144
STORAGE=dict(storage_bytes=16*1024*1024,storage_versions=2048)
MapError=Product(rank=Int64,gid=Int64,reason=Variant(runtime=RuntimeFailure,user=Int64))
Errors=Failure(distributed=MapError)


def _map(p,d,domain,body):
    result=d.local_map(domain,lambda q,i:q.variant(Result(Float64,Int64),'ok',body(q,i)),byte_capacity=CAP)
    from pops.time.variant_types import ValueType
    out=Result(ValueType(dict(value_signature(result)['members'])['ok']),Errors)
    wrapped=p.match(result,ok=lambda q,value:q.variant(out,'ok',value),
        err=lambda q,error:q.variant(out,'err',q.variant(Errors,'distributed',error)))
    return p.require_success(wrapped)


def _sum(p,d,array):
    return d.reduce(array,p.float64(0),lambda g,a,b:a+b,
                    order='global_gid_ascending',byte_capacity=CAP)


def build_bundle_repartition_v14(*, ranks=2, mode='success', steps=3):
    if ranks not in (2,3) or mode not in {'success','stale_history','parent_reject'}:
        raise ValueError('MPI2/MPI3 and a declared receiver mode are required')
    _,layout=_domain('bundle_repartition_context',(4,4))
    case=pops.Case('native_bundle_repartition_'+mode);p=pops.Program('finite_repartition_equations')
    d=Distributed(p,**STORAGE)
    initial_offsets=(0,1,4) if ranks==2 else (0,1,1,4)
    domain=d.domain(initial_offsets,capacity=4)
    q=_map(p,d,domain,lambda g,i:g.float64(d.gid(i)+1))
    volumes=_map(p,d,domain,lambda g,i:g.float64(.25))
    centers=_map(p,d,domain,lambda g,i:(g.float64(d.gid(i))+.5)/4)
    parents=d.requests(domain,domain,lambda g,i:d.index(domain,d.gid(i)))
    raw=p.product(instance=domain,fields=p.product(q=q),geometry=p.product(volume=volumes,center=centers),
                  maps=p.product(source=domain,parent=parents),histories=p.product(previous=q))
    bs=BundleSchema(raw,domains={'current':('instance',),'source':('maps','source')},bindings={
        ('fields','q'):'current',('geometry','volume'):'current',('geometry','center'):'current',
        ('maps','parent'):'current',('maps','parent','[]'):'source',('histories','previous'):'current'})
    bundle=bs.bind(p,raw)
    initial=p.product(bundle=bundle,time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',),byte_capacity=4*1024*1024)
    base=p.continuation_open(initial,schema);proposal=p.product(h=p.requested_dt())
    data_type=Product(step=Int64)
    remap_type=Result(type_of(bundle),Errors)
    @graph(inputs=Product(bundle=type_of(bundle),data=data_type),output=remap_type)
    def repartition(g,args):
        dd=Distributed(g,**STORAGE);old=args['bundle'];old_domain=old['instance']
        split=g.branch(args['data']['step']%2==0,lambda b:b.int64(4),lambda b:b.int64(0))
        offsets=(g.int64(0),split,g.int64(4)) if ranks==2 else (g.int64(0),split,split,g.int64(4))
        new_domain=dd.domain(offsets,family=old_domain)
        requests=dd.requests(new_domain,old_domain,lambda b,i:dd.index(old_domain,(3*dd.gid(i)+1)%4))
        transferred={key:dd.import_values(old['geometry'][key],requests,byte_capacity=CAP)
                     for key in ('volume','center')}
        values=dd.import_values(old['fields']['q'],requests,byte_capacity=CAP)
        history=dd.import_values(old['histories']['previous'],requests,byte_capacity=CAP)
        if mode=='stale_history':
            history=g.branch(args['data']['step']>0,lambda b:old['histories']['previous'],lambda b:history)
        result=bs.bind(g,g.product(instance=new_domain,fields=g.product(q=values),
            geometry=g.product(**transferred),maps=g.product(source=old_domain,parent=requests),
            histories=g.product(previous=history)))
        return g.variant(remap_type,'ok',result)
    mapping=Remap(repartition,source_instance=('maps','source'),histories={'previous':'transfer'})
    envelope=Product(continuation=schema.write_type,increment=Product(change=Float64),
                     observations=Product(previous=Float64),checks=Product())
    trial_type=Result(envelope,Errors)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=trial_type)
    def trial(g,args):
        dd=Distributed(g,**STORAGE);before=args['base'];h=args['proposal']['h']
        moved=g.require_success(remap(g,mapping,before['bundle'],data=g.product(step=before['step'])))
        old=moved['fields']['q'];history=moved['histories']['previous']
        values=_map(g,dd,moved['instance'],lambda b,i:
            (1-2*h)*dd.at_local(old,i)+h*dd.at_local(history,i))
        updated=bs.bind(g,g.product(instance=moved['instance'],fields=g.product(q=values),
            geometry=moved['geometry'],maps=moved['maps'],histories=g.product(previous=old)))
        state=g.product(bundle=updated,time=before['time']+h,step=before['step']+g.int64(1))
        return g.variant(trial_type,'ok',g.product(continuation=state,
            increment=g.product(change=_sum(g,dd,values)-_sum(g,dd,old)),
            observations=g.product(previous=_sum(g,dd,history)),checks=g.product()))
    assessment_type=Product(accepted=Bool,checks=Product(recurrence=Float64,volume=Float64))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),output=assessment_type)
    def assessment(g,args):
        dd=Distributed(g,**STORAGE);before=args['base'];after=args['candidate']['continuation'];h=args['proposal']['h']
        defect=_sum(g,dd,after['bundle']['fields']['q'])-(1-2*h)*_sum(g,dd,before['bundle']['fields']['q'])-h*_sum(g,dd,before['bundle']['histories']['previous'])
        volume=_sum(g,dd,after['bundle']['geometry']['volume'])-1
        tests=(defect*defect<1.e-24,volume*volume<1.e-26,after['time']==before['time']+h)
        accepted=g.boolean(True)
        for test in tests: accepted=g.branch(test,lambda b,accepted=accepted:accepted,lambda b:b.boolean(False))
        return g.product(accepted=accepted,checks=g.product(recurrence=defect,volume=volume))
    approved=p.require_success(advance_once(p,Advance(trial,Assess(assessment)),base=base,proposal=proposal))
    if mode=='parent_reject':
        error=Failure(parent=Product(code=Int64));failure=Result(type_of(approved),error)
        approved=p.require_success(p.branch(p.continuation_read(base)['step']>0,
            lambda b:b.variant(failure,'err',b.variant(error,'parent',b.product(code=b.int64(14)))),
            lambda b:b.variant(failure,'ok',approved)))
    p.continuation_publish(approved);p.step_strategy(FixedDt(.1));case.program(p)
    def check(runtime):
        return {'passed':runtime.macro_step()==steps and abs(runtime.time()-.1*steps)<1.e-14,'frontier_only':True}
    return CaseSpec(case,layout,{}, {},.1*steps,.1,check,
        'Native zero-block Bundle repartition, geometry/Index/history transfer in fixed communicator')
