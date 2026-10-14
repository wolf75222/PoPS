"""Zero-block finite-algorithm AD, including recoverable failures and switching surfaces."""
import pops
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.graph_derivatives import jvp
from pops.time import FixedDt
from pops.time.variant_types import Bool,Float64,Product,Failure,Result,type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def build_finite_iteration_ad_v14(mode='iteration'):
    if mode not in {'iteration','budget','surface','typed_error','second'}:
        raise ValueError('unknown finite AD witness')
    _,layout=_domain('finite_ad_context',(4,4))
    case=pops.Case('finite_ad_v14_'+mode);p=pops.Program('finite_algorithm_equations')
    x=.05 if mode=='surface' else (-.4 if mode=='typed_error' else .4)
    initial=p.product(input=p.float64(x),value=p.float64(0),tangent=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);current=p.continuation_read(base)
    @graph(inputs=Product(x=Float64),output=Float64)
    def function(g,a):
        if mode=='typed_error':
            errors=Failure(problem=Product(value=Float64));result=Result(Float64,errors)
            @graph(inputs=Product(x=Float64),output=result)
            def partial(q,args):
                return q.branch(args['x']>0,lambda r:r.variant(result,'ok',args['x']**2),
                    lambda r:r.variant(result,'err',r.variant(errors,'problem',r.product(value=3*args['x']))))
            failed=partial(g,x=a['x'])
            return g.match(failed,ok=lambda q,value:value,
                err=lambda q,error:q.match(error,problem=lambda r,payload:payload['value'],
                    runtime=lambda r,payload:r.float64(-999)))
        if mode=='second':
            return g.branch(a['x']>0,lambda q:a['x']**3,lambda q:a['x']**2)
        result=g.iterate(a['x'],lambda q,value:value*value/4,lambda q,value:value<.05,
                         max_steps=0 if mode in {'budget','surface'} else 8)
        return g.match(result,ok=lambda q,item:item['state'],err=lambda q,error:3*a['x'])
    first=jvp(function,differentiate_iterations=True)
    if mode=='second':
        result=jvp(first,differentiate_iterations=True)(p,
            primal=p.product(primal=p.product(x=current['input']),tangent=p.product(x=p.float64(1))),
            tangent=p.product(primal=p.product(x=p.float64(1)),tangent=p.product(x=p.float64(0))))
        value,tangent=result['value']['tangent'],result['tangent']['tangent']
    else:
        result=first(p,primal=p.product(x=current['input']),tangent=p.product(x=p.float64(1)))
        value,tangent=result['value'],result['tangent']
    h=p.requested_dt();proposal=p.product(h=h)
    after=p.product(input=current['input'],value=value,tangent=tangent,time=current['time']+h,step=current['step']+p.int64(1))
    envelope=p.product(continuation=after,increment=p.product(),observations=p.product(),checks=p.product())
    candidate=p.continuation_freeze(base,proposal,envelope)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=type_of(envelope)),
           output=Product(accepted=Bool,checks=Product()))
    def assess(g,a):
        answer=a['candidate']['continuation']
        finite=g.branch(g.isfinite(answer['value']),lambda q:g.isfinite(answer['tangent']),lambda q:q.boolean(False))
        return g.product(accepted=finite,checks=g.product())
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(.01));case.program(p)
    return CaseSpec(case,layout,{}, {},.01,.01,
        lambda runtime:{'passed':runtime.macro_step()==1,'frontier_only':True},
        'Native complete Continuation of a finite algorithm and its exact directional derivative')
