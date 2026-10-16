"""Four-way assessment as closed Python graphs executed by native PoPS.

The raw trial proposes q+h. Assessment requests a nontrivial correction to
q+h/2, reassesses its residual, rejects the first attempt with an exact controller
update, and accepts the second. A stop is never converted into a retry.
"""
from __future__ import annotations

import pops
from pops.continuations import ContinuationSchema, Advance, DecisionAssess, advance, advance_once
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Float64, Int64, Product, Variant, Result, Failure, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def build_decision_assess_v14(*, mode='success', steps=3):
    if mode not in {'success','stop','parent_stop','budget','nested_success','nested_reject'}:
        raise ValueError('unknown Decision receiver mode')
    _,layout=_domain('decision_native_context',(4,4))
    case=pops.Case('native_decision_assess_'+mode);p=pops.Program('decision_equations')
    domain=p.range_set(2,capacity=2,ownership='replicated')
    q=p.map(domain,lambda g,i:1.+g.float64(g.index_value(i)))
    initial=p.product(q=q,previous=q,time=p.float64(0),step=p.int64(0),attempt=p.int64(0),corrections=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema)
    proposal=p.product(h=p.requested_dt(),attempt=p.int64(0),stop=p.boolean(False))
    controller=Product(attempt=Int64)
    reason=Product(code=Int64,time=Float64,observed=Float64)
    envelope=Product(continuation=schema.write_type,increment=Product(amount=Float64),
        observations=Product(),checks=Product(corrected=Int64))
    trial_type=Result(envelope,Failure())

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=trial_type)
    def trial(g,args):
        before=args['base'];request=args['proposal'];h=request['h'];old=before['q']
        new=g.map(g.domain(old),lambda b,i:b.at(old,i)+h)
        after=g.product(q=new,previous=old,time=before['time']+h,step=before['step']+g.int64(1),
            attempt=request['attempt'],corrections=before['corrections'])
        return g.variant(trial_type,'ok',g.product(continuation=after,
            increment=g.product(amount=2*h),observations=g.product(),checks=g.product(corrected=g.int64(0))))

    decision_type=Variant(accept=Product(residual=Float64),
        reject=Product(controller_update=controller,reason=reason),
        transform=Variant(normalize=Product(scale=Float64)),stop=reason)

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),output=decision_type)
    def decision(g,args):
        before=args['base'];request=args['proposal'];candidate=args['candidate'];after=candidate['continuation']
        delta=g.map(g.domain(after['q']),lambda b,i:
            b.at(after['q'],i)-b.at(before['q'],i)-request['h']*.5)
        residual=g.reduce(delta,g.float64(0),lambda b,a,x:a+x*x,order='ascending')
        observed=g.reduce(after['q'],g.float64(0),lambda b,a,x:a+x,order='ascending')
        def stop(b):
            return b.variant(decision_type,'stop',b.product(code=b.int64(91),time=after['time'],observed=observed))
        def assessed(b):
            def admissible(c):
                return c.branch(request['attempt']==0,
                    lambda d:d.variant(decision_type,'reject',d.product(
                        controller_update=d.product(attempt=d.int64(1)),
                        reason=d.product(code=d.int64(41),time=after['time'],observed=observed))),
                    lambda d:d.variant(decision_type,'accept',d.product(residual=residual)))
            return b.branch(residual<1.e-26,admissible,stop)
        def ready(b):
            requested=request['stop']
            if mode=='stop':
                requested=b.branch(before['step']>0,lambda c:c.boolean(True),lambda c:requested)
            return b.branch(requested,stop,assessed)
        return g.branch(candidate['checks']['corrected']==0,
            lambda b:b.variant(decision_type,'transform',b.variant(
                Variant(normalize=Product(scale=Float64)),'normalize',b.product(scale=b.float64(.5)))),ready)

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope,
                          payload=Product(scale=Float64)),output=trial_type)
    def normalize(g,args):
        before=args['base'];candidate=args['candidate'];raw=candidate['continuation'];scale=args['payload']['scale']
        new=g.map(g.domain(raw['q']),lambda b,i:
            b.at(before['q'],i)+scale*(b.at(raw['q'],i)-b.at(before['q'],i)))
        after=g.product(q=new,previous=raw['previous'],time=raw['time'],step=raw['step'],
            attempt=raw['attempt'],corrections=before['corrections']+g.int64(1))
        return g.variant(trial_type,'ok',g.product(continuation=after,
            increment=g.product(amount=scale*candidate['increment']['amount']),observations=candidate['observations'],
            checks=g.product(corrected=g.int64(1))))

    retried=Result(Product(proposal=type_of(proposal),controller=controller),Failure())
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),controller=controller,reason=reason),output=retried)
    def retry(g,args):
        # Consume the actual Reject reason and controller update. A wrong reason
        # cannot silently pass the scientific assessment on the next attempt.
        attempt=g.branch(args['reason']['code']==41,lambda b:args['controller']['attempt'],lambda b:b.int64(0))
        request=g.product(h=args['proposal']['h'],attempt=attempt,stop=args['proposal']['stop'])
        return g.variant(retried,'ok',g.product(proposal=request,controller=g.product(attempt=attempt)))

    method=Advance(trial,DecisionAssess(decision,{'normalize':normalize}),retry)
    nested=mode.startswith('nested_')
    if nested:
        # These are the complete typed errors of this supplied child method;
        # no payload is replaced with a scalar status at the parent boundary.
        child_errors=Failure(continuation_budget=Product(value=Int64),
            trial_failure=Failure(),assessment_failure=Failure(),transform_failure=Failure(),
            decision_budget=Product(limit=Int64,used=Int64),decision_stop=reason,
            decision_reject=Product(controller_update=controller,reason=reason))
        nested_result=Result(envelope,child_errors)
        child_method=method
        @graph(inputs=Product(base=type_of(base),proposal=type_of(proposal)),output=nested_result)
        def nested_trial(g,args):
            authority=args['base'];before=g.continuation_read(authority);h=args['proposal']['h']
            first_request=g.product(h=h*.5,attempt=g.int64(0),stop=g.boolean(False))
            first=g.require_success(advance(g,child_method,base=authority,proposal=first_request,
                controller=g.product(attempt=g.int64(0)),max_attempts=g.int64(2),max_transforms=g.int64(1)))
            first_base=g.continuation_child_base(first)
            remaining=(before['time']+h)-g.continuation_read(first_base)['time']
            second_request=g.product(h=remaining,attempt=g.int64(0),stop=g.boolean(False))
            second=g.require_success(advance(g,child_method,base=first_base,proposal=second_request,
                controller=g.product(attempt=g.int64(0)),max_attempts=g.int64(2),max_transforms=g.int64(1)))
            amount=g.continuation_consume_increment(first)['amount']+g.continuation_consume_increment(second)['amount']
            after=g.continuation_candidate_value(g.continuation_accepted_candidate(second))['continuation']
            if mode=='nested_reject':
                def reject_child(b):
                    third_base=b.continuation_child_base(second)
                    request=b.product(h=b.float64(0),attempt=b.int64(0),stop=b.boolean(False))
                    b.require_success(advance_once(b,Advance(child_method.trial,child_method.assessment),
                        base=third_base,proposal=request,max_transforms=b.int64(1)))
                    return b.int64(0)
                g.branch(before['step']>0,reject_child,lambda b:b.int64(0))
            continuation=g.product(q=after['q'],previous=before['q'],time=after['time'],
                step=before['step']+g.int64(1),attempt=after['attempt'],corrections=after['corrections'])
            return g.variant(nested_result,'ok',g.product(continuation=continuation,
                increment=g.product(amount=amount),observations=g.product(),checks=g.product(corrected=g.int64(1))))
        # The parent decision sees the complete candidate, checks its own residual
        # and does not inherit either child's acceptance ticket.
        method=Advance(nested_trial,method.assessment,retry)
        proposal=p.product(h=p.requested_dt(),attempt=p.int64(1),stop=p.boolean(False))
    current=p.continuation_read(base)
    budget=(p.branch(current['step']>0,lambda g:g.int64(0),lambda g:g.int64(1))
            if mode=='budget' else p.int64(1))
    def execute(g):
        accepted=g.require_success(advance(g,method,base=base,proposal=proposal,
            controller=g.product(attempt=g.int64(0)),max_attempts=g.int64(2),max_transforms=budget))
        if mode=='parent_stop':
            def stop_after_child(b):
                child=b.continuation_child_base(accepted)
                request=b.product(h=b.float64(0),attempt=b.int64(1),stop=b.boolean(True))
                b.require_success(advance_once(b,method,base=child,proposal=request,max_transforms=b.int64(1)))
                return accepted
            return g.branch(current['step']>0,stop_after_child,lambda b:accepted)
        return accepted
    accepted=p.require_success(p.continuation_capture(execute))
    p.continuation_publish(accepted);p.step_strategy(FixedDt(.1));case.program(p)
    return CaseSpec(case,layout,{}, {}, .1*steps,.1,
        lambda runtime:{'passed':runtime.macro_step()==steps,'frontier_only':True},
        'Native Decision algebra: corrected trial, original residual, Reject/retry, typed Stop and scoped rollback')
