"""N02: three finite supports, one exchange, two children and parent recovery.

This is a transaction/composition witness, not a named physical model. Inventories
are real PoPS Arrays; no Field carrier or Python stepping loop substitutes for them.
The deliberately bad parent endpoint exercises reject/Transform/reassessment.
"""
from __future__ import annotations

import pops
from pops.continuations import ContinuationSchema, Advance, Assess, advance_once, assess, accept
from pops.contributions import Contribution, Delivery, Increment, FiniteInterval, LinearMap
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, ValueType, type_of
from pops.time._program.structured_control import value_signature
from .common import CaseSpec
from .models_M01_M14 import _domain


def _sum(p, a):
    return p.reduce(a,p.float64(0),lambda g,x,y:x+y,order='ascending')


def _total(p, state):
    return _sum(p,state['volume'])+_sum(p,state['surface'])+_sum(p,state['circuit'])


def _all(p, conditions):
    result=p.boolean(True)
    for condition in conditions:
        result=p.branch(condition,lambda g,result=result:result,lambda g:g.boolean(False))
    return result


def build_multisupport_children_v14(*, mode='success', steps=3):
    if mode not in {'success','double_consume','stale_assessment','parent_reject'}:
        raise ValueError('unknown multisupport receiver mode')
    if type(steps) is not int or steps<1: raise ValueError('positive step count required')
    _,layout=_domain('multisupport_execution_context',(4,4))
    case=pops.Case('N02_native_multisupport_'+mode);p=pops.Program('shared_exchange_children')
    domains={key:p.range_set(n,capacity=n,ownership='replicated')
             for key,n in (('volume',2),('surface',3),('circuit',1))}
    fields={key:p.map(domain,lambda g,i,key=key:g.float64(2 if key=='volume' else 1))
            for key,domain in domains.items()}
    def packet(g,j,begin,end):
        return Contribution('shared',j,FiniteInterval(begin,end,'root')).pack(g)
    ledger=p.product(shared=packet(p,p.float64(0),p.float64(0),p.float64(0)))
    initial=p.product(**fields,histories=p.product(**fields),ledger=ledger,
        time=p.float64(0),step=p.int64(0),attempts=p.int64(0),rejections=p.int64(0),transforms=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);proposal=p.product(h=p.requested_dt())
    envelope_type=Product(continuation=schema.write_type,increment=type_of(ledger),
                         observations=Product(initial_total=Float64),checks=Product())
    errors=Failure(inventory_domain=Product(begin=Float64,end=Float64))
    trial_type=Result(envelope_type,errors)

    maps={}
    for key,coefficient in (('volume',-1.),('surface',.25),('circuit',.75)):
        @graph(inputs=Product(J=Float64,domain=type_of(domains[key])),output=type_of(fields[key]))
        def delivery(g,args,coefficient=coefficient):
            return g.map(args['domain'],lambda q,i:
                coefficient*args['J']/q.float64(q.size(args['domain'])))
        maps[key]=delivery

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=trial_type)
    def child(g,args):
        before=args['base'];h=args['proposal']['h']
        recipients={key:before[key] for key in maps}
        interval=FiniteInterval(before['time'],before['time']+h,'root')
        exchange=Contribution('shared',h,interval)
        deliveries=tuple(Delivery(exchange,recipients[key],LinearMap.differential(maps[key],
            at=g.product(J=g.float64(0),domain=g.domain(recipients[key])),wrt='J')) for key in maps)
        delta=Increment((exchange,),deliveries)
        values={key:delta.apply(g,recipients[key],name='shared_to_'+key) for key in maps}
        continuation=g.product(**values,histories=g.product(**{key:before[key] for key in maps}),
            ledger=delta.pack(g),time=before['time']+h,step=before['step']+g.int64(1),
            attempts=before['attempts'],rejections=before['rejections'],transforms=before['transforms'])
        return g.variant(trial_type,'ok',g.product(continuation=continuation,increment=delta.pack(g),
            observations=g.product(initial_total=_total(g,before)),checks=g.product()))

    assessment_type=Product(accepted=Bool,checks=Product(balance=Float64,endpoint=Float64,delivery=Float64))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope_type),
           output=assessment_type)
    def assess_equations(g,args):
        before=args['base'];candidate=args['candidate'];after=candidate['continuation'];h=args['proposal']['h']
        j=candidate['increment']['shared']['value'];window=candidate['increment']['shared']['interval']
        balance=_total(g,after)-_total(g,before)
        endpoint=after['time']-before['time']-h
        defects=[]
        for key,c in (('volume',-1.),('surface',.25),('circuit',.75)):
            old,new=before[key],after[key]
            defects.append(_sum(g,g.map(g.domain(new),lambda q,i,old=old,new=new,c=c:
                (q.at(new,i)-q.at(old,i)-c*j/q.float64(q.size(q.domain(new))))**2)))
        delivery=defects[0]+defects[1]+defects[2]
        accepted=_all(g,(balance*balance<1.e-24,endpoint*endpoint<1.e-26,delivery<1.e-24,
            (j-h)*(j-h)<1.e-26,window['begin']==before['time'],window['end']==after['time']))
        return g.product(accepted=accepted,checks=g.product(balance=balance,endpoint=endpoint,delivery=delivery))
    assessment=Assess(assess_equations);method=Advance(child,assessment)

    def parent_attempt(g, *, refuse, recovered):
        def body(q):
            before=q.continuation_read(base);h=proposal['h'];half=q.product(h=h/2)
            one=q.require_success(advance_once(q,method,base=base,proposal=half))
            first_delta=q.continuation_consume_increment(one)
            child_base=q.continuation_child_base(one)
            first_endpoint=q.continuation_read(child_base)['time']
            # Reach the parent's exact floating-point endpoint; two nominal h/2
            # additions need not equal the runtime's one h addition.
            remaining=q.product(h=(before['time']+h)-first_endpoint)
            two=q.require_success(advance_once(q,method,base=child_base,proposal=remaining))
            second_delta=q.continuation_consume_increment(two)
            if mode=='double_consume':
                q.branch(before['step']>0,lambda b:b.continuation_consume_increment(two),lambda b:second_delta)
            reached=q.continuation_candidate_value(q.continuation_accepted_candidate(two))['continuation']
            total_j=first_delta['shared']['value']+second_delta['shared']['value']
            total_ledger=q.product(shared=packet(q,total_j,before['time'],reached['time']))
            continuation=q.product(**{key:reached[key] for key in maps},
                histories=q.product(**{key:before[key] for key in maps}),ledger=total_ledger,
                time=reached['time'],step=before['step']+q.int64(1),
                attempts=before['attempts']+q.int64(2 if recovered else 1),
                rejections=before['rejections']+q.int64(1 if recovered else 0),
                transforms=before['transforms']+q.int64(1))
            def envelope(state):
                return q.product(continuation=state,increment=total_ledger,
                    observations=q.product(initial_total=_total(q,before)),checks=q.product())
            # A parent-local bad endpoint is rejected despite both children being accepted.
            bad=q.product(**{key:continuation[key] for key,_ in value_signature(continuation)['members']
                             if key!='time'},time=reached['time']+h/4)
            # Rebuild in schema order: Product order is an executable ABI, not a dict convention.
            bad=q.product(**{key:bad[key] for key,_ in value_signature(continuation)['members']})
            candidate=q.continuation_freeze(base,proposal,envelope(bad))
            rejected_ticket=assess(q,candidate,assessment)
            refusal=accept(q,candidate,rejected_ticket)
            was_rejected=q.match(refusal,ok=lambda b,a:b.boolean(False),err=lambda b,e:b.boolean(True))
            candidate=q.continuation_transform(candidate,envelope(continuation))
            if mode=='stale_assessment':
                # This ticket passes every numerical predicate. A second identity
                # transformation invalidates it, so refusal proves exact version
                # authority rather than merely testing an already-false decision.
                previously_valid=assess(q,candidate,assessment)
                candidate=q.continuation_transform(candidate,envelope(continuation))
                approved=q.branch(before['step']>0,
                    lambda b:b.continuation_accept(candidate,previously_valid),
                    lambda b:b.require_success(accept(b,candidate,assess(b,candidate,assessment))))
            else:
                approved=q.require_success(accept(q,candidate,assess(q,candidate,assessment)))
            failure=Failure(parent_trial=Product(total=Float64,step=Int64))
            refused_type=Result(type_of(approved),failure)
            def reject(b):
                return b.variant(refused_type,'err',b.variant(failure,'parent_trial',
                    b.product(total=_total(b,before),step=before['step'])))
            # A successful candidate cannot hide the deliberately failed first assessment.
            approved=q.require_success(q.branch(was_rejected,
                lambda b:b.variant(refused_type,'ok',approved),reject))
            if refuse or mode=='parent_reject':
                condition=q.boolean(True) if refuse else before['step']>0
                approved=q.require_success(q.branch(condition,reject,
                    lambda b:b.variant(refused_type,'ok',approved)))
            return approved
        return g.continuation_capture(body)

    first=parent_attempt(p,refuse=True,recovered=False)
    accepted_type=ValueType(dict(value_signature(first)['members'])['ok'])
    def recover(g,failure):
        error_type=type_of(failure)
        def propagate(q,unused):
            return q.require_success(q.variant(Result(accepted_type,error_type),'err',failure))
        handlers={tag:propagate for tag,_ in value_signature(failure)['members']}
        handlers['parent_trial']=lambda q,data:q.require_success(parent_attempt(q,refuse=False,recovered=True))
        return g.match(failure,**handlers)
    approved=p.match(first,ok=lambda g,a:a,err=recover)
    p.continuation_publish(approved);p.step_strategy(FixedDt(.1));case.program(p)
    def check(runtime):
        return {'passed':runtime.macro_step()==steps and abs(runtime.time()-.1*steps)<1.e-14,
                'frontier_only':True}
    return CaseSpec(case,layout,{}, {},.1*steps,.1,check,
        'N02 native zero-block three-support exchange, child transactions and parent recovery')
