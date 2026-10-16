"""A regular overdetermined relation whose data domain is a proper submanifold."""
import pops
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.graph_derivatives import GraphPairings
from pops.implicit_derivatives import ImplicitRelation,implicit_jvp,implicit_vjp,linearize
from pops.relations import Algorithm,Approximation,Accuracy,Check,Evidence,Problem,RelationErrors
from pops.time import FixedDt
from pops.time.variant_types import Bool,Float64,Int64,Product,Result,type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def build_implicit_domain_v14(mode='compatible'):
    if mode not in ('compatible','incompatible','transpose'):raise ValueError('unknown implicit data-domain probe')
    _,layout=_domain('implicit_data_domain_context',(4,4));case=pops.Case('implicit_data_domain')
    p=pops.Program('implicit_data_domain')
    initial=p.product(value=p.float64(1),answer=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    continuation=p.continuation_open(initial,schema);current=p.continuation_read(continuation)
    U,D,R=Product(u=Float64),Product(a=Float64,b=Float64),Product(first=Float64,second=Float64)
    args=Product(unknown=U,data=D);full=Product(equations=R,boundary=Product(),gauge=Product())
    parameters=Product(atol=Float64,rtol=Float64,scale=Float64,request=Product())
    evidence=Product(estimates=Product(),valid=Bool);termination=Product(iterations=Int64)
    check=Product(accepted=Bool,observations=Product())
    def ok(q,value):return q.variant(Result(type_of(value),RelationErrors),'ok',value)
    residual=graph(inputs=args,output=R)(lambda q,x:q.product(
        first=x['unknown']['u']-x['data']['a'],second=x['unknown']['u']-x['data']['b']))
    true=graph(inputs=args,output=Bool)(lambda q,x:q.boolean(True))
    empty=graph(inputs=args,output=Product())(lambda q,x:q.product())
    # [1,1]^T has full column rank, but data must satisfy a=b. Rank does not assert
    # an extension of the solution map to independent ambient coordinates (a,b).
    valid=graph(inputs=args,output=Product(differentiable=Bool,regular=Bool,observations=Product()))(
        lambda q,x:q.product(differentiable=q.boolean(True),regular=q.boolean(True),observations=q.product()))
    def request(inputs):return graph(inputs=inputs,output=Result(check,RelationErrors))(
        lambda q,x:ok(q,Check(q,accepted=q.boolean(True),observations=q.product())))
    base_request=request(Product(unknown=U,data=D,residual=R,accuracy=parameters,evidence=evidence))
    relation=ImplicitRelation(Problem(residual,true,true,true,base_request),empty,empty,valid)
    tangent_data=Product(base_unknown=U,base_data=D,direction=D)
    lin=linearize(relation,request_test=request(Product(unknown=U,data=tangent_data,
        residual=full,accuracy=parameters,evidence=evidence)))
    approximation=Product(values=U,evidence=evidence,termination=termination)
    method=graph(inputs=Product(data=tangent_data,guess=U,accuracy=parameters,budget=Product()),
        output=Result(approximation,RelationErrors))(lambda q,x:ok(q,Approximation(q,
            values=q.product(u=x['data']['direction']['a']),
            evidence=Evidence(q,estimates=q.product(),valid=q.boolean(True)),
            termination=q.product(iterations=q.int64(1)))))
    norm=graph(inputs=Product(residual=full),output=Float64)(lambda q,x:
        x['residual']['equations']['first']**2+x['residual']['equations']['second']**2)
    accuracy=Accuracy(p.product(atol=p.float64(1e-24),rtol=p.float64(0),scale=p.float64(1),request=p.product()),norm)
    base=p.product(values=p.product(u=current['value']),data_identity=p.product(a=current['value'],b=current['value']),
        evidence=Evidence(p,estimates=p.product(),valid=p.boolean(True)),termination=p.product(iterations=p.int64(0)))
    options=dict(base=base,algorithm=Algorithm(method),guess=p.product(u=p.float64(0)),
        base_accuracy=accuracy,accuracy=accuracy,budget=p.product())
    if mode=='transpose':
        identity=lambda typ:graph(inputs=Product(value=typ),output=typ)(lambda q,x:x['value'])
        receipt=implicit_vjp(p,lin,covector=p.product(u=p.float64(1)),
            pairings=GraphPairings(identity(U),identity(D)),**options)
        answer=p.require_success(receipt)['values']['a']
    else:
        receipt=implicit_jvp(p,lin,direction=p.product(a=p.float64(1),b=p.float64(1 if mode=='compatible' else 0)),**options)
        answer=p.require_success(receipt)['values']['u']
    h=p.requested_dt();proposal=p.product(h=h)
    packet=p.product(value=current['value'],answer=answer,time=current['time']+h,step=current['step']+p.int64(1))
    envelope=p.product(continuation=packet,increment=p.product(),observations=p.product(),checks=p.product())
    candidate=p.continuation_freeze(continuation,proposal,envelope)
    assessment=graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=type_of(envelope)),
        output=Product(accepted=Bool,checks=Product()))(lambda q,x:q.product(accepted=q.boolean(True),checks=q.product()))
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assessment)))
    p.step_strategy(FixedDt(.01));case.program(p)
    return CaseSpec(case,layout,{}, {},.01,.01,lambda runtime:{'passed':runtime.macro_step()==1},
        'overdetermined exact residual and explicit nonambient implicit derivative domain')
