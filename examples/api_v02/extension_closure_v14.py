"""A new equation and method composed without engine edits or a physical opcode.

On a four-node periodic ring, H(q)=q+q^3 and dH/dt=kappa*L(q).
Crank--Nicolson solves H(z)-H(q)-h*kappa*(L(z)+L(q))/2=0.
For |q|<=1/2 and 0<h*kappa<=.2, z -> z-R(z)/2 maps the cube
[-1/2,1/2]^4 into itself and has infinity-norm Lipschitz constant <=1/2:
its diagonal derivative is 1-(1+3*z_i^2+h*kappa)/2 >=.025,
its two neighbor derivatives are h*kappa/4, and the row sum <=1/2.
The iteration checks its actual residual, and a separate assessment recomputes
all four equations and the H balance before native continuation publication.
This finite ring is not a convergence theorem for a continuum PDE.
"""
import math
import pops
from pops.bundles import BundleSchema
from pops.continuations import ContinuationSchema, Advance, Assess, advance_once
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def _all(p,conditions):
    result=p.boolean(True)
    for condition in conditions:
        result=p.branch(condition,lambda b,result=result:result,lambda b:b.boolean(False))
    return result


def _lookup(p,index,values,*,real=False):
    ctor=p.float64 if real else p.int64
    value=ctor(values[-1])
    for slot in range(len(values)-2,-1,-1):
        value=p.branch(index==slot,lambda b,slot=slot: (b.float64 if real else b.int64)(values[slot]),
                       lambda b,value=value:value)
    return value


def _sum(p,values):
    return p.reduce(values,p.float64(0),lambda b,s,x:s+x,order='ascending')


def _squared(p,values):
    return p.reduce(values,p.float64(0),lambda b,s,x:s+x*x,order='ascending')


def _inventory(p,values):
    return p.map(p.domain(values),lambda b,i:b.at(values,i)+b.at(values,i)**3)


def _laplacian(p,values,left,right):
    return p.map(p.domain(values),lambda b,i:
        b.at(values,b.at(left,i))-2*b.at(values,i)+b.at(values,b.at(right,i)))


def _residual(p,z,q,left,right,h,kappa):
    lz=_laplacian(p,z,left,right);lq=_laplacian(p,q,left,right)
    return p.map(p.domain(q),lambda b,i:
        b.at(z,i)+b.at(z,i)**3-b.at(q,i)-b.at(q,i)**3-
        h*kappa*(b.at(lz,i)+b.at(lq,i))/2)


def build_extension_closure_v14(*,permutation=(0,1,2,3),names=('q','H'),kappa=.4,h=.1,steps=3):
    permutation=tuple(permutation);names=tuple(names)
    if sorted(permutation)!=list(range(4)) or len(names)!=2 or len(set(names))!=2 or not all(type(n)is str and n.isidentifier() for n in names):
        raise ValueError('four-node permutation and two distinct field names are required')
    if not (math.isfinite(kappa) and kappa>0 and math.isfinite(h) and h>0 and h*kappa<=.2) or type(steps)is not int or steps<1:
        raise ValueError('positive time/kappa, h*kappa<=.2, positive step count are required')
    qname,hname=names;inverse={physical:slot for slot,physical in enumerate(permutation)}
    left_slots=tuple(inverse[(physical-1)%4] for physical in permutation)
    right_slots=tuple(inverse[(physical+1)%4] for physical in permutation)
    _,layout=_domain('extension_context',(4,4))
    case=pops.Case('equation_extension_'+qname);p=pops.Program('ring_inventory_crank_nicolson')
    domain=p.range_set(p.int64(4),capacity=8,ownership='replicated')
    q=p.map(domain,lambda b,i:_lookup(b,b.index_value(i),tuple(.1*(physical+1) for physical in permutation),real=True))
    H=_inventory(p,q)
    left=p.map(domain,lambda b,i:b.index(domain,_lookup(b,b.index_value(i),left_slots)))
    right=p.map(domain,lambda b,i:b.index(domain,_lookup(b,b.index_value(i),right_slots)))
    geometry=p.product(volume=p.map(domain,lambda b,i:b.float64(.25)),
        coordinate=p.map(domain,lambda b,i:(_lookup(b,b.index_value(i),permutation,real=True)+.5)/4))
    raw=p.product(instance=domain,fields=p.product(**{qname:q,hname:H}),geometry=geometry,
        maps=p.product(left=left,right=right),histories=p.product(previous=q))
    bindings={('fields',qname):'ring',('fields',hname):'ring',('geometry','volume'):'ring',
        ('geometry','coordinate'):'ring',('maps','left'):'ring',('maps','left','[]'):'ring',
        ('maps','right'):'ring',('maps','right','[]'):'ring',('histories','previous'):'ring'}
    bundle_schema=BundleSchema(raw,domains={'ring':('instance',)},bindings=bindings)
    initial=p.product(bundle=bundle_schema.bind(p,raw),time=p.float64(0),step=p.int64(0),
        residual_squared=p.float64(0),balance=p.float64(0),iterations=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);proposal=p.product(h=p.requested_dt(),kappa=p.float64(kappa))
    envelope=Product(continuation=schema.write_type,increment=Product(inventory=Float64),
        observations=Product(residual_squared=Float64),checks=Product())
    errors=Failure(contract=Product(code=Int64));result_type=Result(envelope,errors)

    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=result_type)
    def trial(g,args):
        old=args['base'];before=old['bundle'];q=before['fields'][qname]
        domain=before['instance'];left=before['maps']['left'];right=before['maps']['right']
        h=args['proposal']['h'];k=args['proposal']['kappa']
        bounded=g.reduce(q,g.boolean(True),lambda b,s,x:_all(b,(s,x>=-.5,x<=.5)),order='ascending')
        guard=Result(Float64,errors)
        valid=_all(g,(bounded,h>0,k>0,h*k<=.2))
        factor=g.require_success(g.branch(valid,lambda b:b.variant(guard,'ok',b.float64(1)),
            lambda b:b.variant(guard,'err',b.variant(errors,'contract',b.product(code=b.int64(14030))))))
        def update(b,z):
            r=_residual(b,z,q,left,right,h,k)
            return b.map(domain,lambda c,i:c.at(z,i)-factor*c.at(r,i)/2)
        def converged(b,z):return _squared(b,_residual(b,z,q,left,right,h,k))<1.e-26
        solved=g.require_success(g.iterate(q,update,converged,max_steps=80))
        z=solved['state'];newH=_inventory(g,z)
        r=_squared(g,_residual(g,z,q,left,right,h,k))
        balance=(_sum(g,newH)-_sum(g,before['fields'][hname]))/4
        bundle=bundle_schema.bind(g,g.product(instance=domain,fields=g.product(**{qname:z,hname:newH}),
            geometry=before['geometry'],maps=before['maps'],histories=g.product(previous=q)))
        out=g.product(bundle=bundle,time=old['time']+h,step=old['step']+1,
            residual_squared=r,balance=balance,iterations=solved['iterations'])
        return g.variant(result_type,'ok',g.product(continuation=out,
            increment=g.product(inventory=balance),observations=g.product(residual_squared=r),checks=g.product()))

    checked=Product(accepted=Bool,checks=Product(equations=Float64,balance=Float64,inventory=Float64))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),output=checked)
    def assess(g,args):
        old=args['base']['bundle'];new=args['candidate']['continuation']['bundle'];h=args['proposal']['h'];k=args['proposal']['kappa']
        q=old['fields'][qname];z=new['fields'][qname]
        r=_squared(g,_residual(g,z,q,old['maps']['left'],old['maps']['right'],h,k))
        inventory=g.map(new['instance'],lambda b,i:b.at(new['fields'][hname],i)-b.at(z,i)-b.at(z,i)**3)
        inv=_squared(g,inventory);balance=(_sum(g,new['fields'][hname])-_sum(g,old['fields'][hname]))/4
        bounded=g.reduce(z,g.boolean(True),lambda b,s,x:_all(b,(s,x>=-.5,x<=.5)),order='ascending')
        return g.product(accepted=_all(g,(bounded,r<1.e-26,balance*balance<1.e-26,inv<1.e-28)),
            checks=g.product(equations=r,balance=balance,inventory=inv))
    accepted=p.require_success(advance_once(p,Advance(trial,Assess(assess)),base=base,proposal=proposal))
    p.continuation_publish(accepted);p.step_strategy(FixedDt(h));case.program(p)
    return CaseSpec(case,layout,{}, {},h*steps,h,
        lambda runtime:{'passed':runtime.macro_step()==steps and abs(runtime.time()-h*steps)<1.e-13},
        'New Python equations: ring4 H=q+q^3, residual-converged Crank-Nicolson; '
        'zero-block complete Bundle, bounded contraction; finite profile, not continuum qualification')
