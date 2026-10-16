"""Native cardinality, repeated gathers and weighted AD; NumPy is only the oracle."""
from __future__ import annotations

import json
import numpy as np

from pops.graph_derivatives import GraphPairings, jvp, tangent_type, vjp
from pops.graph_functions import GraphFunction, graph
from pops.time import FixedDt
from pops.time.expressions import ProgramScalar
from pops.time.variant_types import Product, ValueType, type_of

from .common import CaseSpec, non_owner_metrics, state_array
from .methods_T01_T12 import _initial, _storage


def array_equations(program, values, indices):
    """f(x,J)_j=x[J_j]^2; J and both live domains are held fixed by AD."""
    output = ValueType(dict(type_of(indices).signature, element=type_of(values).signature['element']))
    @graph(inputs=Product(values=type_of(values), indices=type_of(indices)), output=output)
    def equations(p, args):
        squared = p.map(p.domain(args['values']), lambda q, i: q.at(args['values'], i)**2)
        return p.gather(squared, args['indices'])
    ti, to = tangent_type(equations.inputs), tangent_type(equations.output)
    @graph(inputs=Product(value=to), output=to)
    def output_pairing(p, args):
        value = args['value']
        return p.map(p.domain(value), lambda q, i: (q.float64(q.index_value(i))+1)*q.at(value, i))
    @graph(inputs=Product(value=ti), output=ti)
    def input_pairing(p, args):
        value = args['value']['values']
        return p.product(values=p.map(p.domain(value), lambda q, i:
            q.at(value, i)/(q.float64(q.index_value(i))+2)), indices=args['value']['indices'])
    return equations, GraphPairings(output_pairing, input_pairing)


def build_array_derivative_v14(count=3, *, capacity=8, wrong_domain=False):
    names = tuple('g_'+str(i) for i in range(capacity))+('primal_sum', 'duality')
    case, layout, model, block, state, p, q, initial = _storage(
        'array_ad_v14', names, (float(count),)*len(names), cells=(8, 4), sampling='cell_average')
    # Cardinality is read from native state, not supplied as a Python unrolling bound.
    native_count = p.int64(p.norm_inf(q.n))
    domain = p.range_set(native_count, capacity=capacity, ownership='replicated')
    targets = p.range_set(2*native_count, capacity=2*capacity, ownership='replicated')
    values = p.map(domain, lambda a, i: (a.float64(a.index_value(i))+1)/4)
    indices = p.map(targets, lambda a, j: a.index(domain, a.index_value(j)//2))
    equations, pairings = array_equations(p, values, indices)
    transformed = vjp(equations, pairings=pairings)
    # Reload the closed graph with its exact dependent type witnesses; no callback survives.
    transformed = GraphFunction.from_data(json.loads(json.dumps(transformed.to_data())),
                                          inputs=transformed.inputs, output=transformed.output)
    covector_domain = (p.range_set(2*native_count, capacity=2*capacity, ownership='replicated',
                                  family=targets) if wrong_domain else targets)
    covector = p.map(covector_domain, lambda a, j: a.float64(1/3))
    primal = p.product(values=values, indices=indices)
    derivative = transformed(p, primal=primal, covector=covector)
    direction = p.map(domain, lambda a, i: 1/(a.float64(a.index_value(i))+1))
    tangent = jvp(equations)(p, primal=primal,
        tangent=p.product(values=direction, indices=p.map(targets, lambda a, j: a.product())))['tangent']
    pulled = derivative['pullback']['values']
    left_terms = p.map(targets, lambda a, j:
        (a.float64(a.index_value(j))+1)*a.at(tangent, j)*a.at(covector, j))
    right_terms = p.map(domain, lambda a, i:
        (a.float64(a.index_value(i))+2)*a.at(direction, i)*a.at(pulled, i))
    total = lambda values: p.reduce(values, p.float64(0), lambda a, x, y: x+y, order='ascending')
    components = [p.branch(native_count > i,
        lambda a, i=i: a.at(pulled, a.index(domain, i)), lambda a: a.float64(0))
        for i in range(capacity)]
    components.extend((total(derivative['value']), total(left_terms)-total(right_terms)))
    selected = p.value('array_derivatives', tuple(0*q.n[i]+ProgramScalar(value)
                                               for i, value in enumerate(components)))
    p.commit(q.next, p.value('accepted', selected, at=q.next.point))
    p.step_strategy(FixedDt(.01)); case.program(p)
    # Independent finite Jacobian plus explicit diagonal pairings, including multiplicity.
    x = (np.arange(count)+1)/4
    index = np.repeat(np.arange(count), 2)
    jacobian = np.zeros((2*count, count))
    if count:
        jacobian[np.arange(2*count), index] = 2*x[index]
    expected_pull = (jacobian.T@((np.arange(2*count)+1)/3))/(np.arange(count)+2)
    expected = np.zeros(len(names)); expected[:count] = expected_pull
    expected[-2] = np.sum(x[index]**2)
    subject = block[state]
    def invariants(runtime):
        actual = state_array(runtime, 'method', initial.shape)
        if actual is None: return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected[:, None, None])))
        return {'passed': bool(np.isfinite(actual).all() and error < 2e-12),
                'metrics_owner': True, 'max_error': error, 'native_count': count,
                'graph_identity': transformed.identity, 'all_cotangent_entries_checked': True}
    return CaseSpec(case, layout, _initial(case, subject, initial), {'method':subject}, .01, .01,
                    invariants, 'runtime Array AD: repeated gather, full weighted cotangent and duality')
