"""N07: a finite two-level ALE hierarchy and two explicit flux barriers.

The two coarse cells cover [0,1]; coarse cell 1 owns two fine children.
The active composite grid therefore has three cells.  Periodic advection uses
positive speed .4 and moving interior faces.  A first barrier replaces h*F0
and reconstructs the predictor; the second replaces the Heun integral, after
recomputing F1 from that predictor.  Every inventory, index, volume and retained
flux is native PoPS Array/Bundle state inside the Trial.  Uniform is only the
execution context.  This is a fixed refined hierarchy, not an adaptation rule
or a claim to implement a particular historical AMR time integrator.
"""
from __future__ import annotations

import pops
from pops.bundles import BundleSchema
from pops.continuations import ContinuationSchema, Advance, Assess, advance_once
from pops.contributions import Contribution, Delivery, Increment, FiniteInterval, LinearMap
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def _at(p, values, ordinal):
    return p.at(values,p.index(p.domain(values),ordinal))


def _map(p, values, body):
    return p.map(p.domain(values),body)


def _sum(p, values):
    return p.reduce(values,p.float64(0),lambda g,x,y:x+y,order='ascending')


def _combine(p, a, b, coefficient=1):
    return _map(p,a,lambda g,i:g.at(a,i)+coefficient*g.at(b,i))


def _scale(p, values, coefficient):
    return _map(p,values,lambda g,i:coefficient*g.at(values,i))


def _divergence(p, flux, cells):
    n=p.size(cells)
    return p.map(cells,lambda g,i:_at(g,flux,g.index_value(i))-
                 _at(g,flux,(g.index_value(i)+1)%n))


def _move(p, volume, velocity, h):
    return _combine(p,volume,_divergence(p,velocity,p.domain(volume)),-h)


def _flux(p, inventory, volume, velocity, faces, *, coarse_prediction=False):
    n=p.size(p.domain(inventory))
    def face(g,i):
        k=g.index_value(i); donor=(k+n-1)%n
        density=_at(g,inventory,donor)/_at(g,volume,donor)
        if coarse_prediction:
            # At the periodic coarse/fine interface, the provisional coarse
            # flux uses the covered parent mean. Fine interior flux is retained.
            density=g.branch(k==0,
                lambda z:(_at(z,inventory,1)+_at(z,inventory,2))/(_at(z,volume,1)+_at(z,volume,2)),
                lambda z:density)
        return (.4-g.at(velocity,i))*density
    return p.map(faces,face)


def _heun_reference(p, inventory, volume, velocity, faces, h):
    """Original composite finite equations, independent of ledger replacement."""
    F0=_flux(p,inventory,volume,velocity,faces)
    moved=_move(p,volume,velocity,h)
    stage=_combine(p,inventory,_divergence(p,F0,p.domain(inventory)),h)
    F1=_flux(p,stage,moved,velocity,faces)
    J=_scale(p,_combine(p,F0,F1),h/2)
    return _combine(p,inventory,_divergence(p,J,p.domain(inventory))),moved,J


def build_hierarchy_flux_v14(*, mode='success', steps=2):
    modes={'success','constant','stale_stage','last_flux','double_h','old_volume','parent_reject'}
    if mode not in modes: raise ValueError('unknown hierarchy flux receiver mode')
    _,layout=_domain('hierarchy_execution_context',(4,4))
    p=pops.Program('two_barrier_hierarchy_ale_'+mode)
    active=p.range_set(3,capacity=3,ownership='replicated')
    coarse=p.range_set(2,capacity=2,ownership='replicated')
    fine=p.range_set(2,capacity=2,ownership='replicated')
    faces=p.range_set(3,capacity=3,ownership='replicated')
    volume=p.map(active,lambda g,i:g.branch(g.index_value(i)==0,lambda z:z.float64(.5),lambda z:z.float64(.25)))
    density=p.map(active,lambda g,i:g.float64(2) if mode=='constant' else 1+g.float64(g.index_value(i)))
    inventory=p.map(active,lambda g,i:g.at(volume,i)*g.at(density,i))
    velocity=p.map(faces,lambda g,i:g.branch(g.index_value(i)==0,lambda z:z.float64(0),
        lambda z:z.branch(z.index_value(i)==1,lambda q:q.float64(.02),lambda q:q.float64(-.01))))
    parents=p.map(fine,lambda g,i:g.index(coarse,1))
    fine_active=p.map(fine,lambda g,i:g.index(active,g.index_value(i)+1))
    active_parent=p.map(active,lambda g,i:g.index(coarse,g.branch(g.index_value(i)==0,
        lambda z:z.int64(0),lambda z:z.int64(1))))
    domains=p.product(active=active,coarse=coarse,fine=fine,faces=faces)
    maps=p.product(domains=p.product(coarse=coarse,fine=fine,faces=faces),
                   parent=parents,fine_active=fine_active,active_parent=active_parent)
    zero=p.map(faces,lambda g,i:g.float64(0))
    def hierarchy(g,domains,maps,amount,vol,velocity,first,retained):
        def restrict(values):
            return g.map(domains['coarse'],lambda z,i:z.branch(z.index_value(i)==0,
                lambda q:_at(q,values,0),lambda q:_at(q,values,1)+_at(q,values,2)))
        def children(values):
            return g.map(domains['fine'],lambda z,i:z.at(values,z.at(maps['fine_active'],i)))
        positions=g.map(domains['faces'],lambda z,i:z.branch(z.index_value(i)==0,
            lambda q:q.float64(0),lambda q:q.branch(q.index_value(i)==1,
                lambda b:_at(b,vol,0),lambda b:_at(b,vol,0)+_at(b,vol,1))))
        return g.product(instance=domains['active'],
            fields=g.product(active=amount,coarse=restrict(amount),fine=children(amount)),
            geometry=g.product(active=vol,coarse=restrict(vol),fine=children(vol),
                position=positions,velocity=velocity),
            maps=maps,
            histories=g.product(first_integral=first,retained_integral=retained))
    raw=hierarchy(p,domains,maps,inventory,volume,velocity,zero,zero)
    bindings={('maps','parent'):'fine',('maps','parent','[]'):'coarse',
              ('maps','fine_active'):'fine',('maps','fine_active','[]'):'active',
              ('maps','active_parent'):'active',('maps','active_parent','[]'):'coarse',
              ('geometry','position'):'faces',('geometry','velocity'):'faces',
              ('histories','first_integral'):'faces',('histories','retained_integral'):'faces'}
    bindings.update({(kind,level):level for kind in ('fields','geometry') for level in ('active','coarse','fine')})
    bundle_schema=BundleSchema(raw,domains={'active':('instance',),
        **{key:('maps','domains',key) for key in ('coarse','fine','faces')}},bindings=bindings)
    bundle=bundle_schema.bind(p,raw)
    initial=p.product(bundle=bundle,time=p.float64(0),step=p.int64(0),barriers=p.int64(0),
                      begin=p.float64(0),end=p.float64(0))
    schema=ContinuationSchema(initial,time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);proposal=p.product(h=p.requested_dt())
    envelope=Product(continuation=schema.write_type,increment=Product(flux=type_of(zero)),
                     observations=Product(stage_change=Float64),checks=Product())
    errors=Failure(inventory_domain=Product(begin=Float64,end=Float64),differential=Product(code=Int64))
    trial_type=Result(envelope,errors)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=trial_type)
    def trial(g,args):
        before=args['base'];b=before['bundle'];h=args['proposal']['h']
        Q=b['fields']['active'];V=b['geometry']['active'];w=b['geometry']['velocity']
        d=g.product(active=b['instance'],**{key:b['maps']['domains'][key] for key in ('coarse','fine','faces')})
        interval=FiniteInterval(before['time'],before['time']+h,'physical')
        @graph(inputs=Product(flux=type_of(zero),cells=type_of(active)),output=type_of(Q))
        def incidence(z,a):return _divergence(z,a['flux'],a['cells'])
        action=LinearMap.differential(incidence,at=g.product(flux=g.map(d['faces'],lambda z,i:z.float64(0)),cells=d['active']),wrt='flux')
        Fcoarse=_flux(g,Q,V,w,d['faces'],coarse_prediction=True)
        F0=_flux(g,Q,V,w,d['faces'])
        coarse_first=Contribution('interface',_scale(g,Fcoarse,h),interval)
        stage_ledger=Increment((coarse_first,),(Delivery(coarse_first,Q,action),))
        preliminary=stage_ledger.apply(g,Q,name='provisional_stage')
        corrected_first=Contribution('interface',_scale(g,F0,h),interval)
        first=stage_ledger.replace(coarse_first,corrected_first)
        stage=first.apply(g,Q,name='recomputed_stage')
        first.require_application(stage)
        moved=_move(g,V,w,h)
        F1_stale=_flux(g,preliminary,moved,w,d['faces'])
        F1=_flux(g,stage,moved,w,d['faces'])
        predicted=Contribution('interface',_scale(g,_combine(g,Fcoarse,F1_stale),h/2),interval)
        final_ledger=Increment((predicted,),(Delivery(predicted,Q,action),))
        J=_scale(g,_combine(g,F0,F1_stale if mode=='stale_stage' else F1),h/2)
        if mode=='last_flux': J=F1
        if mode=='double_h': J=_scale(g,J,h)
        actual=Contribution('interface',J,interval)
        final=final_ledger.replace(predicted,actual)
        updated=final.apply(g,Q,name='composite_retained_integral')
        final.require_application(updated)
        volumes=V if mode=='old_volume' else moved
        result=bundle_schema.bind(g,hierarchy(g,d,b['maps'],updated,volumes,w,corrected_first.value,J))
        stage_change=_sum(g,_map(g,stage,lambda z,i:(z.at(stage,i)-z.at(preliminary,i))**2))
        after=g.product(bundle=result,time=before['time']+h,step=before['step']+1,
            barriers=before['barriers']+2,begin=before['time'],end=before['time']+h)
        return g.variant(trial_type,'ok',g.product(continuation=after,increment=g.product(flux=J),
            observations=g.product(stage_change=stage_change),checks=g.product()))
    decision=Product(accepted=Bool,checks=Product(equations=Float64,mass=Float64,gcl=Float64))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),output=decision)
    def assess(g,args):
        before=args['base'];after=args['candidate']['continuation'];b=before['bundle'];a=after['bundle'];h=args['proposal']['h']
        expected,V,J=_heun_reference(g,b['fields']['active'],b['geometry']['active'],
            b['geometry']['velocity'],b['maps']['domains']['faces'],h)
        errors=_sum(g,_map(g,expected,lambda z,i:(z.at(expected,i)-z.at(a['fields']['active'],i))**2))
        errors=errors+_sum(g,_map(g,J,lambda z,i:(z.at(J,i)-z.at(args['candidate']['increment']['flux'],i))**2))
        gcl=_sum(g,_map(g,V,lambda z,i:(z.at(V,i)-z.at(a['geometry']['active'],i))**2))
        mass=_sum(g,a['fields']['active'])-_sum(g,b['fields']['active'])
        checks=(errors<1.e-25,gcl<1.e-25,mass*mass<1.e-25,
            after['barriers']==before['barriers']+2,after['begin']==before['time'],after['end']==before['time']+h)
        accepted=g.boolean(True)
        for check in checks:accepted=g.branch(check,lambda z,accepted=accepted:accepted,lambda z:z.boolean(False))
        return g.product(accepted=accepted,checks=g.product(equations=errors,mass=mass,gcl=gcl))
    approved=p.require_success(advance_once(p,Advance(trial,Assess(assess)),base=base,proposal=proposal))
    if mode=='parent_reject':
        failure=Failure(parent=Product(barriers=Int64));rejected=Result(type_of(approved),failure)
        p.require_success(p.branch(p.continuation_read(base)['step']>0,
            lambda g:g.variant(rejected,'err',g.variant(failure,'parent',g.product(barriers=g.int64(2)))),
            lambda g:g.variant(rejected,'ok',approved)))
    p.continuation_publish(approved);p.step_strategy(FixedDt(.05))
    case=pops.Case('N07_hierarchy_flux_'+mode);case.program(p)
    return CaseSpec(case,layout,{}, {},steps*.05,.05,
        lambda runtime:{'passed':runtime.macro_step()==steps,'frontier_only':True},
        'Native finite two-level fixed hierarchy, ALE/GCL, two flux replacements and dependent predictor recomputation')
