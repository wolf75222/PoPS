"""Original-equation implicit Array derivative, with native cardinality and full receipts."""
from __future__ import annotations

import numpy as np
import pops
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.graph_derivatives import GraphPairings, tangent_type
from pops.implicit_derivatives import ImplicitRelation, implicit_jvp, implicit_vjp, linearize
from pops.relations import Algorithm, Approximation, Check, Evidence, Problem, RelationErrors, Accuracy
from pops.time import FixedDt
from pops.time.expressions import ProgramScalar
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, ValueType, type_of
from .common import CaseSpec, non_owner_metrics, state_array
from .methods_T01_T12 import _initial, _storage


def _ok(p, value):
    return p.variant(Result(type_of(value), RelationErrors), 'ok', value)


def build_implicit_array_v14(count=3, *, transpose=False, dishonest=False, capacity=4, native_state=True, distributed_ranks=None, redundant=False):
    names = tuple('d_'+str(i) for i in range(capacity))
    transport = 8*1024*1024 if distributed_ranks is not None else None
    if distributed_ranks is not None and (not native_state or distributed_ranks not in (1,2,3)):
        raise ValueError('distributed implicit reception requires a zero-block MPI1/2/3 context')
    order = 'global_gid_ascending' if distributed_ranks is not None else 'ascending'
    def map_(q,D,body):
        return q.map(D,body,**({'byte_capacity':transport} if transport else {}))
    def reduce_(q,values,initial,body):
        return q.reduce(values,initial,body,order=order,**({'byte_capacity':transport} if transport else {}))
    if native_state:
        from .models_M01_M14 import _domain
        _, layout = _domain('implicit_array_context', (4, 4))
        case = pops.Case('implicit_array_complete_v14')
        p = pops.Program('implicit_array_equations')
        if distributed_ranks is None:
            domain = p.range_set(count, capacity=capacity, ownership='replicated')
        else:
            from pops.time._program.distributed_arrays import Distributed
            offsets = ((0,count) if distributed_ranks==1 else (0,min(1,count),count)
                       if distributed_ranks==2 else (0,min(1,count),min(1,count),count))
            domain = Distributed(p,storage_bytes=512*1024*1024,storage_versions=16384).domain(offsets,capacity=capacity)
        u = map_(p,domain, lambda z, i: 1+(z.float64(z.index_value(i))+1)/4)
        algorithm_errors=(RelationErrors if distributed_ranks is None else Failure(
            relation=Product(code=Int64),user=ValueType(dict(type_of(u.inputs[0]).signature['members'])['err'])))
        d = map_(p,domain, lambda z, i: z.at(u, i)*z.at(u, i))
        initial = p.product(domain=domain, roots=u, data=d,
            answer=map_(p,domain, lambda z, i: z.float64(0)),
            time=p.float64(0), step=p.int64(0))
        schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
        continuation_base = p.continuation_open(initial, schema)
        current = p.continuation_read(continuation_base)
        domain, u, d = current['domain'], current['roots'], current['data']
        n = p.size(domain)  # cardinality comes from the accepted native continuation
    else:
        case, layout, model, block, state, p, q, initial = _storage(
            'implicit_array_v14', names, (float(count),)*capacity, cells=(8, 4), sampling='cell_average')
        n = p.int64(p.norm_inf(q.n))
        domain = p.range_set(n, capacity=capacity, ownership='replicated')
        u = map_(p,domain, lambda z, i: 1+(z.float64(z.index_value(i))+1)/4)
        d = map_(p,domain, lambda z, i: z.at(u, i)*z.at(u, i))
    if not native_state: algorithm_errors=RelationErrors
    U, D, R = Product(u=type_of(u)), Product(d=type_of(d)), Product(r=type_of(u))
    if redundant: R=Product(r=type_of(u),redundant=type_of(u))
    full = Product(equations=R, boundary=Product(), gauge=Product())
    parameters = Product(atol=Float64, rtol=Float64, scale=Float64, request=Product())
    evidence_type = Product(estimates=Product(), valid=Bool)
    termination = Product(iterations=Int64)
    check_type = Product(accepted=Bool, observations=Product())
    args = Product(unknown=U, data=D)
    @graph(inputs=args, output=R)
    def residual(z, a):
        unknown, data = a['unknown']['u'], a['data']['d']
        first=map_(z,z.domain(unknown), lambda w,i:w.at(unknown,i)**2-w.at(data,i))
        return z.product(r=first,**({'redundant':map_(z,z.domain(unknown),lambda w,i:2*(w.at(unknown,i)**2-w.at(data,i)))} if redundant else {}))
    true = graph(inputs=args, output=Bool)(lambda z, a: z.boolean(True))
    empty = graph(inputs=args, output=Product())(lambda z, a: z.product())
    validity = graph(inputs=args,
        output=Product(differentiable=Bool, regular=Bool, observations=Product()))(
        lambda z, a: z.product(differentiable=z.boolean(True),
            regular=reduce_(z,a['unknown']['u'], z.boolean(True),
                lambda w, carry, value: w.branch(carry, lambda r: value>0, lambda r: r.boolean(False))), observations=z.product()))
    request = graph(inputs=Product(unknown=U, data=D, residual=R,
        accuracy=parameters, evidence=evidence_type), output=Result(check_type, RelationErrors))(
        lambda z, a: _ok(z, Check(z, accepted=z.boolean(True), observations=z.product())))
    relation = ImplicitRelation(Problem(residual, true, true, true, request), empty, empty, validity)
    linear_data = Product(base_unknown=U, base_data=D, direction=tangent_type(D))
    tangent_request = graph(inputs=Product(unknown=tangent_type(U), data=linear_data,
        residual=tangent_type(full), accuracy=parameters, evidence=evidence_type),
        output=Result(check_type, RelationErrors))(
        lambda z, a: _ok(z, Check(z, accepted=z.boolean(True), observations=z.product())))
    lin = linearize(relation, request_test=tangent_request, transport_byte_capacity=transport)
    approx = Product(values=tangent_type(U), evidence=evidence_type, termination=termination)
    @graph(inputs=Product(data=linear_data, guess=tangent_type(U), accuracy=parameters, budget=Product()),
           output=Result(approx, algorithm_errors))
    def method(z, a):
        base, direction = a['data']['base_unknown']['u'], a['data']['direction']['d']
        values = map_(z,z.domain(base), lambda w, i:
            w.at(direction, i)/((1 if dishonest else 2)*w.at(base, i)))
        return _ok(z, Approximation(z, values=z.product(u=values),
            evidence=Evidence(z, estimates=z.product(), valid=z.boolean(True)),
            termination=z.product(iterations=z.int64(1))))
    norm = graph(inputs=Product(residual=full), output=Float64)(lambda z, a:
        sum((reduce_(z,a['residual']['equations'][key],z.float64(0),
                 lambda w,carry,value:carry+value*value) for key,_ in R.signature['members']),z.float64(0)))
    accuracy = Accuracy(p.product(atol=p.float64(1e-22), rtol=p.float64(0), scale=p.float64(1), request=p.product()), norm)
    base = p.product(values=p.product(u=u), data_identity=p.product(d=d),
        evidence=Evidence(p, estimates=p.product(), valid=p.boolean(True)),
        termination=p.product(iterations=p.int64(0)))
    guess = p.product(u=map_(p,domain, lambda z, i: z.float64(0)))
    direction = p.product(d=map_(p,domain, lambda z, i: 1/(z.float64(z.index_value(i))+1)))
    if transpose:
        output_pairing = graph(inputs=Product(value=tangent_type(U)), output=tangent_type(U))(
            lambda z, a: z.product(u=map_(z,z.domain(a['value']['u']),
                lambda w, i: (w.float64(w.index_value(i))+2)*w.at(a['value']['u'], i))))
        input_pairing = graph(inputs=Product(value=tangent_type(D)), output=tangent_type(D))(
            lambda z, a: z.product(d=map_(z,z.domain(a['value']['d']),
                lambda w, i: w.at(a['value']['d'], i)/(w.float64(w.index_value(i))+3))))
        receipt = implicit_vjp(p, lin, base=base,
            covector=p.product(u=map_(p,domain, lambda z, i: z.float64(1/3))),
            pairings=GraphPairings(output_pairing, input_pairing), algorithm=Algorithm(method),
            guess=guess, base_accuracy=accuracy, accuracy=accuracy, budget=p.product(), transport_byte_capacity=transport)
        answer = p.require_success(receipt)['values']['d']
    else:
        receipt = implicit_jvp(p, lin, base=base, direction=direction, algorithm=Algorithm(method),
            guess=guess, base_accuracy=accuracy, accuracy=accuracy, budget=p.product(), transport_byte_capacity=transport)
        answer = p.require_success(receipt)['values']['u']
    if native_state:
        h = p.requested_dt(); proposal = p.product(h=h)
        continued = p.product(domain=domain, roots=u, data=d, answer=answer,
            time=current['time']+h, step=current['step']+p.int64(1))
        envelope = p.product(continuation=continued, increment=p.product(),
                             observations=p.product(), checks=p.product())
        candidate = p.continuation_freeze(continuation_base, proposal, envelope)
        @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
               output=Product(accepted=Bool, checks=Product()))
        def assess(z, a):
            result = a['candidate']['continuation']
            same = z.same_domain(result['domain'], z.domain(result['answer']))
            return z.product(accepted=same, checks=z.product())
        p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    else:
        values = [p.branch(n>i, lambda z, i=i: z.at(answer, z.index(domain, i)),
                           lambda z: z.float64(0)) for i in range(capacity)]
        out = p.value('implicit_array', tuple(0*q.n[i]+ProgramScalar(value) for i, value in enumerate(values)))
        p.commit(q.next, p.value('accepted', out, at=q.next.point))
    p.step_strategy(FixedDt(.01)); case.program(p)
    expected = np.zeros(capacity)
    roots = 1+(np.arange(count)+1)/4
    jacobian = np.diag(2*roots)
    if count:
        expected[:count] = (np.linalg.solve(jacobian.T, (np.arange(count)+2)/3)/(np.arange(count)+3)
                            if transpose else np.linalg.solve(jacobian, 1/(np.arange(count)+1)))
    if native_state:
        return CaseSpec(case, layout, {}, {}, .01, .01,
            lambda runtime: {'passed': runtime.macro_step()==1, 'frontier_only': True},
            'zero-block native '+('ArrayDist' if distributed_ranks else 'Array')+' implicit derivative, complete accepted Continuation')
    subject = block[state]
    def invariants(runtime):
        actual = state_array(runtime, 'method', initial.shape)
        if actual is None: return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected[:,None,None])))
        return {'passed': bool(np.isfinite(actual).all() and error<2e-12), 'metrics_owner': True,
                'max_error': error, 'native_count':count, 'transpose':transpose,
                'all_array_coordinates_checked':True}
    return CaseSpec(case, layout, _initial(case, subject, initial), {'method':subject}, .01, .01,
                    invariants, 'original-equation dynamic Array implicit derivative')
