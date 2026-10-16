"""Deferred native Continuation counterexamples: genuine PoPS equations and capabilities.

This Uniform/fixed-layout witness stores a field, an Int64 recurrence, Variant, finite Array,
Domain and aliased Index. It is not AMR or evolving child-field-frontier qualification.
"""
from __future__ import annotations
import numpy as np
import pops
from pops.continuations import ContinuationSchema, Candidate, Assess, assess, Advance, advance_once, advance
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Product, Variant, Float64, Int64, Bool, Result, Failure, type_of
from pops.time.expressions import ProgramScalar
from examples.api_v02.methods_T01_T12 import _storage
from examples.api_v02.models_M01_M14 import _initial
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics


def build_continuation(*, mode='success', steps=3):
    allowed={'success','assessment_reject','stale_transform','wrong_candidate','parent_reject',
             'duplicate_increment','retry','loop_identity'}
    if mode not in allowed: raise ValueError('unknown continuation receiver')
    if type(steps) is not int or steps < 1: raise ValueError('positive integral step count required')
    parts=_storage('Continuation_native',('u',),(0.,))
    case,layout,model,block,state,p,q,data=parts
    domain=p.range_set(p.int64(2),capacity=2,ownership='replicated')
    samples=p.map(domain,lambda g,i:g.index_value(i))
    index=p.index(domain,p.int64(0))
    mode_type=Variant(cold=Int64,warm=Int64)
    initial=p.product(state=q.n,time=p.float64(0.),step=p.int64(0),count=p.int64(0),
                      mode=p.variant(mode_type,'cold',p.int64(0)),domain=domain,index=index,samples=samples)
    schema=ContinuationSchema(initial,fields=((q.next,('state',)),),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema)
    proposal=p.product(h=p.requested_dt(),attempt=p.int64(0))
    endpoint=q.next.point
    # The output schema is concrete in field/domain authority, with no retagging of field reads.
    envelope_type=Product(continuation=schema.write_type,increment=Product(delta=type_of(initial["count"])),
                          observations=Product(previous_count=type_of(initial["count"])),checks=Product())
    trial_type=Result(envelope_type,Failure())
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=trial_type)
    def trial(g,args):
        b=args['base'];h=args['proposal']['h']
        old=g.match(b['mode'],cold=lambda t,x:x,warm=lambda t,x:x)
        count=b['count']+g.int64(1)
        delta=old+g.int64(1)+g.at(b['samples'],b['index'])
        value=g.value('scientific_recurrence',(b['state'][0]+ProgramScalar(g.float64(delta)),),at=endpoint)
        next_samples=g.map(b['domain'],lambda t,i:t.at(b['samples'],i)+count)
        continuation=g.product(state=value,time=b['time']+h,step=b['step']+g.int64(1),count=count,
            mode=g.variant(mode_type,'warm',count),domain=b['domain'],index=b['index'],samples=next_samples)
        envelope=g.product(continuation=continuation,increment=g.product(delta=delta),
                            observations=g.product(previous_count=old),checks=g.product())
        return g.variant(trial_type,'ok',envelope)
    check_type=Product(accepted=Bool,checks=Product(endpoint_error=Float64,attempt=Int64))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope_type),output=check_type)
    def assessment(g,args):
        defect=args['candidate']['continuation']['time']-(args['base']['time']+args['proposal']['h'])
        valid=defect==g.float64(0.)
        if mode=='assessment_reject':valid=g.boolean(False)
        if mode=='retry':valid=args['proposal']['attempt']>g.int64(0)
        return g.product(accepted=valid,checks=g.product(endpoint_error=defect,attempt=args['proposal']['attempt']))
    method=Advance(trial,Assess(assessment))

    def proposed(g):
        envelope=g.require_success(trial(g,base=g.continuation_read(base),proposal=proposal))
        return g.continuation_freeze(base,proposal,envelope)

    if mode in {'stale_transform','wrong_candidate','loop_identity'}:
        candidate=proposed(p);ticket=assess(p,candidate,method.assessment)
        if mode=='stale_transform':
            candidate=p.continuation_transform(candidate,p.continuation_candidate_value(candidate))
        elif mode=='wrong_candidate':candidate=proposed(p)
        else:
            # The body is one SSA freeze site executed twice, not two unrolled source sites.
            seed=p.product(candidate=candidate,ticket=ticket)
            def repeat(g,carried,i):
                current=proposed(g)
                return g.product(candidate=current,ticket=assess(g,carried['candidate'],method.assessment))
            last=p.fold(seed,2,repeat)
            candidate,ticket=last['candidate'],last['ticket']
        approved=p.continuation_accept(candidate,ticket)
    elif mode=='retry':
        controller=p.product(reductions=p.int64(0))
        errors=Failure(assessment_rejected=Product(checks=Product(endpoint_error=Float64,attempt=Int64)))
        retry_type=Result(Product(proposal=type_of(proposal),controller=type_of(controller)),Failure())
        @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),controller=type_of(controller),failure=errors),output=retry_type)
        def retry(g,args):
            updated=g.product(proposal=g.product(h=args['proposal']['h'],attempt=args['proposal']['attempt']+g.int64(1)),
                              controller=g.product(reductions=args['controller']['reductions']+g.int64(1)))
            return g.variant(retry_type,'ok',updated)
        approved=p.require_success(advance(p,Advance(trial,method.assessment,retry),base=base,proposal=proposal,
                                           controller=controller,max_attempts=p.int64(3)))
    else:
        approved=p.require_success(advance_once(p,method,base=base,proposal=proposal))
        if mode=='parent_reject':
            errors=Failure(parent=Product(code=Int64));failure_type=Result(type_of(approved),errors)
            p.require_success(p.variant(failure_type,'err',p.variant(errors,'parent',p.product(code=p.int64(77)))))
        if mode=='duplicate_increment':
            p.continuation_consume_increment(approved)
            p.continuation_consume_increment(approved)
    p.continuation_publish(approved)
    p.step_strategy(FixedDt(.1));case.program(p);values=_initial(case,block[state],data)
    expected=steps*(steps+1)*(steps+2)/6
    def check(runtime):
        actual=state_array(runtime,'method',data.shape)
        if actual is None:return non_owner_metrics()
        error=float(np.max(abs(actual-expected)))
        return dict(passed=bool(error<1.e-12),metrics_owner=True,independent_recurrence_error=error)
    return CaseSpec(case,layout,values,{'method':block[state]},steps*.1,.1,check,
                    'Uniform native complete Continuation witness; reception pending')
