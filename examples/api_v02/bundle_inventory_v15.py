"""Three inventory coordinates plus a separately remapped geometric instance.

The conserved inventory keeps its exact support. The geometry receives a fresh
instance and an explicit cyclic parent map. This does not assert that a transfer
between distinct inventory supports is an identity application. All numerical
values come from the real Program; expected values occur only in receivers.
"""
import pops
from pops.bundles import BundleSchema,Remap,remap
from pops.continuations import Candidate,ContinuationSchema
from pops.contributions import Contribution,Delivery,Increment,FiniteInterval
from pops.graph_functions import graph
from pops.ledger_provenance import InventoryBinding
from pops.time import FixedDt
from pops.time.variant_types import Bool,Float64,Product,Result,Failure,type_of
from examples.api_v02.common import CaseSpec
from examples.api_v02.inventory_publication_v15 import DT,_layout,_identity_map


def build_bundle_inventory_v15(*,steps=2):
    p=pops.Program('bundle_inventory_and_geometric_remap')
    domain=p.range_set(3,capacity=3,ownership='replicated')
    q=p.map(domain,lambda g,i:g.float64(g.index_value(i)+1))
    center=p.map(domain,lambda g,i:g.float64(g.index_value(i)))
    parent=p.map(domain,lambda g,i:g.index(domain,g.index_value(i)))
    raw=p.product(instance=domain,fields=p.product(q=q),geometry=p.product(center=center),
        maps=p.product(source=domain,inventory=domain,parent=parent),histories=p.product(q=q))
    bs=BundleSchema(raw,domains={'geometry':('instance',),'source':('maps','source'),
        'inventory':('maps','inventory')},bindings={('fields','q'):'inventory',
        ('geometry','center'):'geometry',('maps','parent'):'geometry',
        ('maps','parent','[]'):'source',('histories','q'):'inventory'})
    initial=p.product(bundle=bs.bind(p,raw),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base);b=old['bundle']
    h=p.requested_dt();proposal=p.product(h=h);recipient=b['fields']['q']
    delta=p.map(p.domain(recipient),lambda g,i:g.float64((old['step']+1)*(g.index_value(i)+1)))
    c=Contribution('coordinate_production',delta,FiniteInterval(old['time'],old['time']+h,'physical'))
    inc=Increment((c,),(Delivery(c,recipient,_identity_map(recipient)),))
    applied=inc.apply(p,recipient,name='current_inventory')
    data=p.product(q=applied,history=recipient)
    result_type=Result(bs.value_type,Failure())
    @graph(inputs=Product(bundle=bs.value_type,data=type_of(data)),output=result_type)
    def geometry_remap(g,a):
        before=a['bundle'];new=g.range_set(g.size(before['instance']),family=before['instance'])
        parents=g.map(new,lambda z,i:z.index(before['instance'],(z.index_value(i)+1)%3))
        centers=g.map(new,lambda z,i:z.at(before['geometry']['center'],z.at(parents,i)))
        candidate=bs.bind(g,g.product(instance=new,fields=g.product(q=a['data']['q']),
            geometry=g.product(center=centers),maps=g.product(source=before['instance'],
                inventory=before['maps']['inventory'],parent=parents),
            histories=g.product(q=a['data']['history'])))
        return g.variant(result_type,'ok',candidate)
    moved=p.require_success(remap(p,Remap(geometry_remap,source_instance=('maps','source'),
        histories={'q':'transfer'}),b,data=data))
    after=p.product(bundle=moved,time=old['time']+h,step=old['step']+1)
    packet=inc.pack(p,bindings=(InventoryBinding(recipient,('bundle','fields','q')),))
    candidate=Candidate(p,base=base,proposal=proposal,continuation=after,increment=packet,
        observations=p.product(),checks=p.product())
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),
        candidate=type_of(p.continuation_candidate_value(candidate))),
        output=Product(accepted=Bool,checks=Product(residual=Float64)))
    def assess(g,a):
        before=a['base'];now=a['candidate']['continuation'];q0=before['bundle']['fields']['q'];q1=now['bundle']['fields']['q']
        errors=g.map(g.domain(q1),lambda z,i:
            (z.at(q1,i)-z.at(q0,i)-z.float64((before['step']+1)*(z.index_value(i)+1)))**2+
            (z.at(now['bundle']['histories']['q'],i)-z.at(q0,i))**2)
        residual=g.reduce(errors,g.float64(0),lambda z,s,v:s+v,order='ascending')
        accepted=g.branch(residual==0,lambda z:now['time']==before['time']+a['proposal']['h'],
                          lambda z:z.boolean(False))
        return g.product(accepted=accepted,checks=g.product(residual=residual))
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(DT));case=pops.Case('bundle_inventory_and_geometric_remap');case.program(p)
    return CaseSpec(case,_layout(),{}, {},steps*DT,DT,
        lambda runtime:{'passed':runtime.macro_step()==steps and runtime.time()==steps*DT},
        'Three replicated inventory coordinates on exact persistent support; an independent '
        'three-point geometry is cyclically remapped. Mandatory ledger-to-fields.q path, '
        'complete Bundle history/maps, two accepted steps and strict checkpoint/restart.')
