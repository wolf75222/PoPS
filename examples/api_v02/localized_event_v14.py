"""An interior event reconstructed by authored finite graphs, without a model opcode.

q'=1, z=q² and J=integral(q dt). The first upward crossing q=1 is located
by bounded bisection; q then resets to zero. Geometry is absent from this scalar
witness. Every retained constraint, integral, history and branch is reconstructed
at the reached time. ComputedDt is an existing native PoPS strategy.
"""
from __future__ import annotations

import pops
from pops.continuations import Advance, Assess, ContinuationSchema, advance_once
from pops.graph_functions import graph
from pops.relations import Accuracy, Algorithm, Approximation, Check, Evidence, Problem, RelationErrors, solve
from pops.time import ComputedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def _all(p, conditions):
    result = p.boolean(True)
    for condition in conditions:
        result = p.branch(result, lambda b, condition=condition:condition,
                          lambda b:b.boolean(False))
    return result


def _ok(p, value):
    return p.variant(Result(type_of(value), RelationErrors), 'ok', value)


def _event_problem():
    U = Product(duration=Float64)
    D = Product(q0=Float64, h=Float64, target=Float64)
    R = Product(event=Float64)
    A = Product(atol=Float64, rtol=Float64, scale=Float64, request=Product())
    E = Product(estimates=Product(), valid=Bool)
    arguments = Product(unknown=U, data=D)
    residual = graph(inputs=arguments, output=R)(lambda p,x:p.product(
        event=x['data']['q0']+x['unknown']['duration']-x['data']['target']))
    admissible = graph(inputs=arguments, output=Bool)(lambda p,x:_all(p,(
        x['unknown']['duration']>0, x['unknown']['duration']<=x['data']['h'],
        x['data']['q0']<x['data']['target'],
        x['data']['q0']+x['data']['h']>=x['data']['target'])))
    true = graph(inputs=arguments, output=Bool)(lambda p,x:p.boolean(True))
    request = graph(inputs=Product(unknown=U, data=D, residual=R, accuracy=A, evidence=E),
                    output=Result(Product(accepted=Bool, observations=Product()), RelationErrors))(
        lambda p,x:_ok(p,Check(p,accepted=p.boolean(True),observations=p.product())))
    problem = Problem(residual, admissible, true, true, request)
    approximation = Product(values=U, evidence=E, termination=Product(iterations=Int64))

    @graph(inputs=Product(data=D, guess=U, accuracy=A, budget=Product(steps=Int64)),
           output=Result(approximation, RelationErrors))
    def bisect(p, args):
        data = args['data']
        initial = p.product(lo=p.float64(0), hi=data['h'])
        def step(b, state):
            middle = (state['lo']+state['hi'])*.5
            error = data['q0']+middle-data['target']
            return b.branch(error == 0,
                lambda c:c.product(lo=middle, hi=middle),
                lambda c:c.branch(error>0,
                    lambda d:d.product(lo=state['lo'],hi=middle),
                    lambda d:d.product(lo=middle,hi=state['hi'])))
        result = p.require_success(p.iterate(initial, step,
            lambda b,state:(state['hi']-state['lo'])**2<=args['accuracy']['atol'],
            max_steps=args['budget']['steps']))
        duration = (result['state']['lo']+result['state']['hi'])*.5
        return _ok(p,Approximation(p,values=p.product(duration=duration),
            evidence=Evidence(p,estimates=p.product(),valid=p.boolean(True)),
            termination=p.product(iterations=result['iterations'])))
    norm = graph(inputs=Product(residual=R), output=Float64)(
        lambda p,x:x['residual']['event']**2)
    return problem, Algorithm(bisect), norm


def build_localized_event_v14(*, mode='success'):
    if mode not in ('success','stale_dense'):
        raise ValueError('unknown localized event witness mode')
    _,layout = _domain('event_native_context',(4,4))
    case = pops.Case('localized_event_'+mode)
    p = pops.Program('localized_event_equations')
    initial = p.product(q=p.float64(0),z=p.float64(0),integral=p.float64(0),
        last_integral=p.float64(0),previous_q=p.float64(0),previous_z=p.float64(0),
        before_reset=p.float64(0),duration=p.float64(0),branch=p.int64(0),
        time=p.float64(0),step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
    base = p.continuation_open(initial,schema)
    proposal = p.product(h=p.requested_dt())
    increment = Product(duration=Float64, production=Float64, reset=Float64, flux=Float64)
    envelope = Product(continuation=schema.write_type, increment=increment,
                       observations=Product(hit=Bool),checks=Product())
    problem,algorithm,norm = _event_problem()

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),
           output=Result(envelope,RelationErrors))
    def trial(g,args):
        old=args['base'];h=args['proposal']['h'];q0=old['q']
        hit=_all(g,(old['branch']==0,q0+h>=1.))
        def locate(b):
            parameters=b.product(atol=b.float64(1.e-28),rtol=b.float64(0),
                                 scale=b.float64(1),request=b.product())
            answer=solve(b,problem,data=b.product(q0=q0,h=h,target=b.float64(1)),
                algorithm=algorithm,guess=b.product(duration=h*.5),
                accuracy=Accuracy(parameters,norm),budget=b.product(steps=b.int64(60)))
            return b.require_success(answer)['values']['duration']
        duration=g.branch(hit,locate,lambda b:h)
        before_reset=q0+duration
        reset=g.branch(hit,lambda b:-before_reset,lambda b:b.float64(0))
        q=before_reset+reset
        z=q*q
        flux=q0*duration+.5*duration*duration
        if mode=='stale_dense':
            # Only on the event: clamping/resetting q while retaining the proposed
            # endpoint constraint and integral is the precise invalid construction.
            z=g.branch(hit,lambda b:(q0+h)**2,lambda b:z)
            flux=g.branch(hit,lambda b:q0*h+.5*h*h,lambda b:flux)
        state=g.product(q=q,z=z,integral=old['integral']+flux,last_integral=flux,
            previous_q=q0,previous_z=old['z'],before_reset=before_reset,duration=duration,
            branch=old['branch']+g.branch(hit,lambda b:b.int64(1),lambda b:b.int64(0)),
            time=old['time']+duration,step=old['step']+g.int64(1))
        return _ok(g,g.product(continuation=state,
            increment=g.product(duration=duration,production=duration,reset=reset,flux=flux),
            observations=g.product(hit=hit),checks=g.product()))

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),
           output=Product(accepted=Bool,checks=Product(defect=Float64)))
    def assess(g,args):
        old=args['base'];candidate=args['candidate'];new=candidate['continuation']
        event=candidate['observations']['hit'];inc=candidate['increment'];d=inc['duration']
        q_before=old['q']+d
        expected_q=g.branch(event,lambda b:b.float64(0),lambda b:q_before)
        expected_flux=old['q']*d+.5*d*d
        defect=(new['q']-expected_q)**2+(new['z']-new['q']**2)**2
        defect=defect+(new['last_integral']-expected_flux)**2+(inc['flux']-expected_flux)**2
        defect=defect+(new['integral']-old['integral']-expected_flux)**2
        defect=defect+(new['previous_q']-old['q'])**2+(new['previous_z']-old['z'])**2
        defect=defect+(new['before_reset']-q_before)**2+(new['time']-old['time']-d)**2
        defect=defect+(new['duration']-d)**2+(new['q']-old['q']-inc['production']-inc['reset'])**2
        event_ok=g.branch(event,lambda b:(q_before-1.)**2<=1.e-26,lambda b:b.boolean(True))
        expected_branch=old['branch']+g.branch(event,lambda b:b.int64(1),lambda b:b.int64(0))
        accepted=_all(g,(d>0,d<=args['proposal']['h'],event_ok,defect<1.e-24,
                        new['branch']==expected_branch,new['step']==old['step']+1))
        return g.product(accepted=accepted,checks=g.product(defect=defect))

    accepted=p.require_success(advance_once(p,Advance(trial,Assess(assess)),base=base,proposal=proposal))
    candidate=p.continuation_accepted_candidate(accepted)
    duration=p.continuation_candidate_value(candidate)['increment']['duration']
    p.reached_duration(duration)
    p.continuation_publish(accepted)
    p.step_strategy(ComputedDt(.6,endpoint_ulps=4,max_rejections=0))
    case.program(p)
    return CaseSpec(case,layout,{}, {},1.6,.6,
        lambda runtime:{'passed':runtime.macro_step()==3},
        'Native scalar event by bounded bisection; q, algebraic z=q², integrated flux, '
        'history and reset branch reconstructed at the computed interior endpoint')
