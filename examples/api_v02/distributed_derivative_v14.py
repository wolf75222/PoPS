"""Closed ArrayDist equations and weighted JVP/VJP in a zero-block native continuation.

The independent receiver forms the complete Jacobian from a small incidence matrix.
No NumPy operation participates in either the equations or their native derivatives.
"""
import pops
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.graph_derivatives import GraphPairings, jvp, vjp, tangent_type
from pops.time import FixedDt
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Bool, Float64, Product, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain

TRANSPORT_BYTES = 4*1024*1024


def build_distributed_derivative_v14(mode='gather', *, ranks=2, empty=False, wrong_domain=False):
    if ranks not in (1,2,3) or mode not in ('gather','scatter','scan'):
        raise ValueError('reception covers MPI1/2/3 and gather/scatter/scan')
    offsets = ((0,3) if ranks==1 else (0,1,3) if ranks==2 else (0,1,1,3))
    if empty: offsets=tuple(0 for _ in offsets)
    _, layout = _domain('distributed_derivative_context', (4,4))
    case=pops.Case('distributed_derivative_v14_'+mode)
    p=pops.Program('distributed_derivative_equations')
    d=Distributed(p,storage_bytes=256*1024*1024,storage_versions=8192)
    domain=d.domain(offsets,capacity=3)
    def map_(q,D,body):return q.map(D,body,byte_capacity=TRANSPORT_BYTES)
    source=map_(p,domain,lambda q,i:q.float64(q.index_value(i)+1))
    zeros=map_(p,domain,lambda q,i:q.float64(0))
    initial=p.product(domain=domain,source=source,value=zeros,tangent=zeros,pullback=zeros,
                      time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);current=p.continuation_read(base)
    domain,source=current['domain'],current['source']
    @graph(inputs=Product(x=type_of(source)),output=type_of(source))
    def equations(q,a):
        x=a['x'];D=q.domain(x)
        if mode=='gather':
            requests=Distributed(q).requests(D,D,lambda r,i:r.index(D,r.index_value(i)%2))
            values=q.gather(x,requests,byte_capacity=TRANSPORT_BYTES)
        elif mode=='scatter':
            requests=Distributed(q).requests(D,D,lambda r,i:r.index(D,r.index_value(i)%2))
            values=q.scatter_reduce(x,requests,D,q.float64(0),lambda r,s,v:s+v,
                order='global_gid_ascending',byte_capacity=TRANSPORT_BYTES)
        else:
            values=q.scan(x,q.float64(0),lambda r,s,v:r.product(state=s+v,value=s+v),
                order='global_gid_ascending',byte_capacity=TRANSPORT_BYTES)['values']
        return map_(q,D,lambda r,i:r.at(values,i)**2)
    ti,to=tangent_type(equations.inputs),tangent_type(equations.output)
    @graph(inputs=Product(value=to),output=to)
    def output_metric(q,a):
        value=a['value']
        return map_(q,q.domain(value),lambda r,i:(r.float64(r.index_value(i))+1)*r.at(value,i))
    @graph(inputs=Product(value=ti),output=ti)
    def input_inverse_metric(q,a):
        value=a['value']['x']
        return q.product(x=map_(q,q.domain(value),lambda r,i:r.at(value,i)/(r.float64(r.index_value(i))+2)))
    direction_domain=d.domain(offsets,family=domain) if wrong_domain else domain
    direction=map_(p,direction_domain,lambda q,i:1/(q.float64(q.index_value(i))+1))
    forward=jvp(equations,transport_byte_capacity=TRANSPORT_BYTES)(p,
        primal=p.product(x=source),tangent=p.product(x=direction))
    pulled=vjp(equations,pairings=GraphPairings(output_metric,input_inverse_metric),
        transport_byte_capacity=TRANSPORT_BYTES)(p,primal=p.product(x=source),
            covector=map_(p,domain,lambda q,i:q.float64(1/3)))
    h=p.requested_dt();proposal=p.product(h=h)
    continued=p.product(domain=domain,source=source,value=forward['value'],tangent=forward['tangent'],
        pullback=pulled['pullback']['x'],time=current['time']+h,step=current['step']+p.int64(1))
    envelope=p.product(continuation=continued,increment=p.product(),observations=p.product(),checks=p.product())
    candidate=p.continuation_freeze(base,proposal,envelope)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=type_of(envelope)),
           output=Product(accepted=Bool,checks=Product()))
    def assess(q,a):
        value=a['candidate']['continuation'];D=value['domain'];good=q.boolean(True)
        for role in ('source','value','tangent','pullback'):
            good=q.branch(good,lambda z,role=role:z.same_domain(D,z.domain(value[role])),lambda z:z.boolean(False))
        return q.product(accepted=good,checks=q.product())
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(.01));case.program(p)
    return CaseSpec(case,layout,{}, {},.01,.01,
        lambda runtime:{'passed':runtime.macro_step()==1,'frontier_only':True},
        'zero-block native distributed '+mode+' and weighted JVP/VJP; empty rank when MPI3')


def build_request_topology_derivative_v14(*, ranks=2, wrapped=False):
    """One compiled graph: requested duration supplies the continuous branch input."""
    offsets = (0,3) if ranks==1 else (0,1,3) if ranks==2 else (0,1,1,3)
    if ranks not in (1,2,3):raise ValueError('reception covers MPI1/2/3')
    _,layout=_domain('request_topology_context',(4,4))
    case=pops.Case('request_topology_v14_'+str(wrapped));p=pops.Program('request_topology_equations')
    d=Distributed(p,storage_bytes=256*1024*1024,storage_versions=8192)
    domain=d.domain(offsets,capacity=3)
    source=p.map(domain,lambda q,i:q.float64(q.index_value(i)+1),byte_capacity=TRANSPORT_BYTES)
    initial=p.product(domain=domain,source=source,input=p.float64(0),value=p.float64(0),
        tangent=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);current=p.continuation_read(base)
    @graph(inputs=Product(x=Float64,values=type_of(current['source'])),output=Float64)
    def function(g,a):
        def selected(q,x):
            D=q.domain(a['values'])
            requests=Distributed(q).requests(D,D,lambda r,i:
                r.index(D,r.branch(x>0,lambda z:z.int64(0),lambda z:z.int64(1))))
            imported=q.gather(a['values'],requests,byte_capacity=TRANSPORT_BYTES)
            return x*x+q.reduce(imported,q.float64(0),lambda r,s,v:s+v,
                order='global_gid_ascending',byte_capacity=TRANSPORT_BYTES)
        if not wrapped:return selected(g,a['x'])
        trace=g.iterate(a['x'],lambda q,x:selected(q,x),lambda q,x:q.boolean(False),max_steps=1)
        return g.match(trace,ok=lambda q,item:item['state'],err=lambda q,error:3*a['x'])
    h=p.requested_dt();x=100*(h-.01)
    result=jvp(function,differentiate_iterations=True,transport_byte_capacity=TRANSPORT_BYTES)(p,
        primal=p.product(x=x,values=current['source']),
        tangent=p.product(x=p.float64(1),values=p.map(current['domain'],lambda q,i:q.float64(0),
            byte_capacity=TRANSPORT_BYTES)))
    proposal=p.product(h=h)
    after=p.product(domain=current['domain'],source=current['source'],input=x,value=result['value'],
        tangent=result['tangent'],time=current['time']+h,step=current['step']+p.int64(1))
    envelope=p.product(continuation=after,increment=p.product(),observations=p.product(),checks=p.product())
    candidate=p.continuation_freeze(base,proposal,envelope)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=type_of(envelope)),
           output=Product(accepted=Bool,checks=Product()))
    def assess(q,a):
        state=a['candidate']['continuation']
        return q.product(accepted=q.branch(q.isfinite(state['value']),
            lambda r:r.isfinite(state['tangent']),lambda r:r.boolean(False)),checks=q.product())
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(.015));case.program(p)
    return CaseSpec(case,layout,{}, {},.015,.015,
        lambda runtime:{'passed':runtime.macro_step()==1,'frontier_only':True},
        'Native Requests branch topology; same graph rejects h=.01 and differentiates h=.015')
