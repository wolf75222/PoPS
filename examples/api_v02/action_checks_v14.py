"""A singular phi1 action as three Python graphs, with no inverse or native dispatch.

The target domain here is exactly finite real 2x2 matrices satisfying A*A=0.
On this domain phi1(hA)v = v + h*A*v/2. This is not a general expm routine.
The independent contract measures the full vector error, including ker(A).
"""
import pops
from pops.actions import Action, ActionErrors
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.relations import Approximation, Evidence
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain

Vec = Product(x=Float64, y=Float64)
Matrix = Product(a=Float64, b=Float64, c=Float64, d=Float64)
Data = Product(matrix=Matrix, vector=Vec, h=Float64)
Request = Product(atol=Float64, rtol=Float64, scale=Float64, request=Product())
Budget = Product(terms=Int64)
Target = Product(exact=Vec, domain=Bool)
Estimates = Product(claimed_error=Float64)
EvidenceType = Product(estimates=Estimates, valid=Bool)
Termination = Product(terms=Int64)
ApproximationType = Product(values=Vec, evidence=EvidenceType, termination=Termination)
CheckType = Product(accepted=Bool, error_bound=Float64,
                    observations=Product(error=Vec, claimed_error=Float64, terms=Int64))


def _both(p, a, b):
    return p.branch(a, lambda q:b, lambda q:q.boolean(False))


def _apply(p, matrix, vector):
    return p.product(x=matrix['a']*vector['x']+matrix['b']*vector['y'],
                     y=matrix['c']*vector['x']+matrix['d']*vector['y'])


def _ok(p, value):
    return p.variant(Result(type_of(value), ActionErrors), 'ok', value)


def _norm(p, value):
    x=p.branch(value['x']>=0,lambda q:value['x'],lambda q:-value['x'])
    y=p.branch(value['y']>=0,lambda q:value['y'],lambda q:-value['y'])
    return p.branch(x>=y,lambda q:x,lambda q:y)


def make_nilpotent_action(*, dishonest=False):
    @graph(inputs=Product(data=Data), output=Target)
    def target(p, args):
        data=args['data'];A=data['matrix'];v=data['vector'];h=data['h']
        domain=p.boolean(True)
        for scalar in (A['a'],A['b'],A['c'],A['d'],v['x'],v['y'],h):
            domain=_both(p,domain,p.isfinite(scalar))
        for scalar in (A['a']*A['a']+A['b']*A['c'],A['a']*A['b']+A['b']*A['d'],
                       A['c']*A['a']+A['d']*A['c'],A['c']*A['b']+A['d']*A['d']):
            domain=_both(p,domain,scalar==0)
        Av=_apply(p,A,v)
        return p.product(exact=p.product(x=v['x']+h*Av['x']/2,y=v['y']+h*Av['y']/2),domain=domain)

    @graph(inputs=Product(target=Target,data=Data,accuracy=Request,budget=Budget),
           output=Result(ApproximationType,ActionErrors))
    def body(p,args):
        data=args['data'];initial=p.product(total=p.product(x=p.float64(0),y=p.float64(0)),term=data['vector'])
        def term(q,state,k):
            At=_apply(q,data['matrix'],state['term']);denominator=q.float64(k)+2
            return q.product(total=q.product(x=state['total']['x']+state['term']['x'],
                                             y=state['total']['y']+state['term']['y']),
                term=q.product(x=data['h']*At['x']/denominator,y=data['h']*At['y']/denominator))
        result=p.fold(initial,args['budget']['terms'],term,max_steps=16)
        values=result['total']
        if dishonest:
            # For the witness A=[[0,b],[0,0]], this error lies in ker(A).
            # A*w-based residual would miss it. The final action contract must not.
            values=p.product(x=values['x']+7,y=values['y'])
        return _ok(p,Approximation(p,values=values,
            evidence=Evidence(p,estimates=p.product(claimed_error=p.float64(0)),valid=args['target']['domain']),
            termination=p.product(terms=args['budget']['terms'])))

    @graph(inputs=Product(target=Target,data=Data,values=Vec,evidence=EvidenceType,
                          termination=Termination,accuracy=Request,budget=Budget),
           output=Result(CheckType,ActionErrors))
    def contract(p,args):
        error=p.product(x=args['values']['x']-args['target']['exact']['x'],
                        y=args['values']['y']-args['target']['exact']['y'])
        accepted=_both(p,args['target']['domain'],args['termination']['terms']==args['budget']['terms'])
        accepted=_both(p,accepted,args['termination']['terms']>=p.int64(1))
        return _ok(p,p.product(accepted=accepted,error_bound=_norm(p,error),
            observations=p.product(error=error,claimed_error=args['evidence']['estimates']['claimed_error'],
                                   terms=args['termination']['terms'])))
    return Action(target,body,contract)


def build_action_checks_v14(mode='exact', *, coefficient=1.):
    if mode not in {'exact','fabricated','budget','domain'}:raise ValueError('unknown action witness')
    _,layout=_domain('action_context',(4,4));case=pops.Case('action_v14_'+mode)
    p=pops.Program('action_equations')
    initial=p.product(value=p.product(x=p.float64(0),y=p.float64(0)),error=p.float64(0),
                      time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);current=p.continuation_read(base)
    data=p.product(matrix=p.product(a=p.float64(1 if mode=='domain' else 0),b=p.float64(coefficient),
        c=p.float64(0),d=p.float64(0)),vector=p.product(x=p.float64(1),y=p.float64(3)),h=p.float64(.2))
    accuracy=p.product(atol=p.float64(1e-12),rtol=p.float64(0),scale=p.float64(1),request=p.product())
    checked=p.require_success(make_nilpotent_action(dishonest=mode=='fabricated').evaluate(p,data=data,
        accuracy=accuracy,budget=p.product(terms=p.int64(1 if mode=='budget' else 2))))
    duration=p.requested_dt();proposal=p.product(h=duration)
    after=p.product(value=checked['values'],error=checked['checks']['error_bound'],
                    time=current['time']+duration,step=current['step']+p.int64(1))
    envelope=p.product(continuation=after,increment=p.product(),observations=p.product(),checks=p.product())
    candidate=p.continuation_freeze(base,proposal,envelope)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=type_of(envelope)),
           output=Product(accepted=Bool,checks=Product()))
    def assess(q,args):
        return q.product(accepted=q.isfinite(args['candidate']['continuation']['error']),checks=q.product())
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(.01));case.program(p)
    return CaseSpec(case,layout,{}, {},.01,.01,lambda runtime:{'passed':runtime.macro_step()==1},
        'Zero-block singular nilpotent phi1 action, native series and independent full-vector error')
