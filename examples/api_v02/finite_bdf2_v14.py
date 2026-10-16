"""Nonlinear inventory, coarsening and retained BDF2 ledger in actual PoPS.

This is a discrete inventory/transfer witness, not an implicit PDE solve claim.
The supplied current rate and solve defect have declared, independently checked
values. Averaging H(q), averaging q, and reevaluating its history are distinct.
"""
import pops
from pops.math import sqrt
from pops.bundles import BundleSchema
from pops.continuations import ContinuationSchema, Advance, Assess, advance_once
from pops.contributions import Contribution, Delivery, Increment, FiniteInterval, LinearMap, bdf2_increment
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def _mean(p, array):
    return p.reduce(array, p.float64(0), lambda q,a,b:a+b, order='ascending') / p.float64(p.size(p.domain(array)))


def build_finite_bdf2_v14(*, mode='success', steps=2):
    if mode not in {'success', 'wrong_domain', 'negative_history', 'parent_reject'}:
        raise ValueError('unknown finite BDF2 receiver mode')
    _, layout = _domain('finite_ledger_context', (4,4))
    case = pops.Case('finite_inventory_bdf2_v14_'+mode)
    p = pops.Program('finite_inventory_bdf2')
    domain = p.range_set(2, capacity=8, ownership='replicated')
    q = p.map(domain, lambda g,i: 2*g.float64(g.index_value(i)))
    inventory = p.map(domain, lambda g,i: g.at(q,i)+g.at(q,i)**2)
    previous_delta = p.map(domain, lambda g,i: g.float64(-.4))
    previous_rate = p.map(domain, lambda g,i: -g.at(q,i)**3)
    raw = p.product(instance=domain, fields=p.product(q=q,H=inventory),
        geometry=p.product(volume=p.map(domain,lambda g,i:g.float64(.5))),
        maps=p.product(), histories=p.product(delta=previous_delta, rate=previous_rate))
    bundle_schema = BundleSchema(raw, domains={'current':('instance',)},
        bindings={('fields','q'):'current',('fields','H'):'current',
                  ('geometry','volume'):'current',('histories','delta'):'current',
                  ('histories','rate'):'current'})
    bundle = bundle_schema.bind(p,raw)
    initial = p.product(bundle=bundle,time=p.float64(0),step=p.int64(0),
        previous_begin=p.float64(-.2), previous_end=p.float64(0), h_previous=p.float64(.2),
        mean_q=p.float64(0), mean_H=p.float64(0), transferred_F=p.float64(0), reevaluated_F=p.float64(0))
    schema = ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base = p.continuation_open(initial,schema); proposal = p.product(h=p.requested_dt())
    envelope = Product(continuation=schema.write_type,increment=Product(delta=Float64),
                       observations=Product(),checks=Product())
    trial_result = Result(envelope,Failure(inventory_domain=Product(begin=Float64,end=Float64),
                                         differential=Product(code=Int64)))

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=trial_result)
    def trial(g,args):
        old = args['base']; b = old['bundle']; h = args['proposal']['h']
        d = g.range_set(1,family=b['instance'])
        mean_H = _mean(g,b['fields']['H']); mean_q = _mean(g,b['fields']['q'])
        transferred_F = _mean(g,b['histories']['rate'])
        H = g.map(d,lambda z,i:mean_H)
        source_domain = (g.branch(old['step']>0,lambda z:z.range_set(1,family=d),lambda z:d)
                         if mode=='wrong_domain' else d)
        rate_value = g.map(source_domain,lambda z,i:z.float64(-2))
        history_value = g.map(d,lambda z,i:_mean(z,b['histories']['delta']))
        defect_value = g.map(d,lambda z,i:z.float64(.01))
        window = FiniteInterval(old['time'],old['time']+h,'physical')
        old_begin = (g.branch(old['step']>0,lambda z:old['previous_end']+h,
                             lambda z:old['previous_begin']) if mode=='negative_history' else old['previous_begin'])
        old_window = FiniteInterval(old_begin,old['previous_end'],'physical')
        @graph(inputs=Product(value=type_of(H)),output=type_of(H))
        def identity(z,x): return x['value']
        action = LinearMap.differential(identity,at=g.product(value=H),wrt='value')
        def ledger(name,value,interval,measure='increment',kind='exchange'):
            c = Contribution(name,value,interval,measure,kind)
            return Increment((c,),(Delivery(c,H,action),))
        rate = ledger('rate',rate_value,window,'rate')
        history = ledger('history',history_value,old_window)
        defect = ledger('residual',defect_value,window,kind='defect')
        weighted = bdf2_increment(g,rate,history,h=h,h_previous=old['h_previous'],
                                 solve_residual=defect,interval=window)
        updated = weighted.apply(g,H,name='inventory_update')
        weighted.require_application(updated)
        delta = g.map(d,lambda z,i:z.at(updated,i)-z.at(H,i))
        # An explicitly selected positive branch reconstructs q from H=q+q².
        q_next = g.map(d,lambda z,i:(sqrt(1+4*z.at(updated,i))-1)/2)
        new_bundle = bundle_schema.bind(g,g.product(instance=d,
            fields=g.product(q=q_next,H=updated),geometry=g.product(volume=g.map(d,lambda z,i:z.float64(1))),
            maps=g.product(),histories=g.product(delta=delta,rate=g.map(d,lambda z,i:transferred_F))))
        out = g.product(bundle=new_bundle,time=old['time']+h,step=old['step']+1,
            previous_begin=old['time'],previous_end=old['time']+h,h_previous=h,
            mean_q=mean_q,mean_H=mean_H,transferred_F=transferred_F,reevaluated_F=-mean_q**3)
        return g.variant(trial_result,'ok',g.product(continuation=out,
            increment=g.product(delta=_mean(g,delta)),observations=g.product(),checks=g.product()))

    decision = Product(accepted=Bool,checks=Product(original_defect=Float64,inventory=Float64))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),output=decision)
    def assess(g,args):
        old=args['base']; after=args['candidate']['continuation']; h=args['proposal']['h']
        before_H=_mean(g,old['bundle']['fields']['H']); prior_delta=_mean(g,old['bundle']['histories']['delta'])
        H=_mean(g,after['bundle']['fields']['H']); qv=_mean(g,after['bundle']['fields']['q'])
        r=h/old['h_previous']; alpha=(1+2*r)/(1+r); beta=r*r/(1+r)
        defect=alpha*H-(1+r)*before_H+beta*(before_H-prior_delta)+2*h
        inventory_error=H-qv-qv*qv
        accepted=g.branch((defect-.01)**2<1.e-24,lambda z:inventory_error**2<1.e-24,
                          lambda z:z.boolean(False))
        if mode=='parent_reject':
            accepted=g.branch(old['step']>0,lambda z:z.boolean(False),lambda z:accepted)
        return g.product(accepted=accepted,checks=g.product(original_defect=defect,inventory=inventory_error))
    accepted=p.require_success(advance_once(p,Advance(trial,Assess(assess)),base=base,proposal=proposal))
    p.continuation_publish(accepted); p.step_strategy(FixedDt(.1)); case.program(p)
    return CaseSpec(case,layout,{}, {},.1*steps,.1,
        lambda runtime:{'passed':runtime.macro_step()==steps},
        'Native nonlinear H inventory, complete Bundle coarsening, BDF2 history and solve defect')
