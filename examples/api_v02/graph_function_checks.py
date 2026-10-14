"""Closed GraphFunction equations for reception by the actual installed PoPS compiler.

These are finite language witnesses, not a substitute numerical runtime. Expected values use
ordinary NumPy/Python only in the independent acceptance oracle, never in the executed graph.
"""
from __future__ import annotations

import json
import numpy as np
import pops
from pops.graph_functions import GraphFunction, graph
from pops.time.variant_types import Failure, Float64, Int64, Product, Result, type_of
from pops.time import FixedDt
from .common import CaseSpec, non_owner_metrics, state_array
from .methods_T01_T12 import _initial, _storage


def scalar_algorithm(coefficient):
    """A closed equation and a finite algorithm, both serialized without a callback."""
    @graph(inputs=Product(x=Float64), output=Float64, captures={'a': coefficient})
    def update(p, args, constants):
        return p.branch(args['x'] >= 0,
                        lambda q: constants['a']*args['x']+1,
                        lambda q: constants['a']*args['x']-1)

    @graph(inputs=Product(x=Float64, n=Int64), output=Float64)
    def advance(p, args):
        return p.fold(args['x'], args['n'],
                      lambda q, value, index: update(q, x=value), max_steps=4)
    return GraphFunction.from_data(json.loads(json.dumps(advance.to_data())))


def _oracle(coefficient, x, count):
    for _ in range(count):
        x = coefficient*x+(1 if x >= 0 else -1)
    return x


def build_graph_function_check(kind='serialized', *, coefficient=2.):
    case, layout, model, block, state, p, q, initial = _storage(
        'graph_'+kind, ('a', 'b', 'c'), (1., 2., 3.), cells=(8, 4), sampling='cell_average')
    q.n
    initial += np.arange(8)[None, None, :]/32.
    expected = initial.copy()
    if kind == 'serialized':
        function = scalar_algorithm(coefficient)
        left = function(p, x=p.float64(1), n=p.int64(2))
        right = function(p, x=p.float64(-1), n=p.int64(3))
        @graph(inputs=Product(n=Int64), output=Int64)
        def sum_squares(g, args):
            domain = g.range_set(args['n'], capacity=8, ownership='replicated')
            squares = g.map(domain, lambda b, index: b.index_value(index)*b.index_value(index))
            return g.reduce(squares, g.int64(0), lambda b, total, value: total+value,
                            order='ascending')
        sums = GraphFunction.from_data(json.loads(json.dumps(sum_squares.to_data())))
        # Two function-local domains must have distinct families even in one caller.
        squares = p.float64(sums(p, n=p.int64(4))+sums(p, n=p.int64(3)))
        fields = [p.value('broadcast_'+str(i), scalar*q.n)
                  for i, scalar in enumerate((left, right, squares))]
        selected = p.value('equations', tuple(fields[i][i] for i in range(3)))
        expected *= np.array([_oracle(coefficient, 1., 2), _oracle(coefficient, -1., 3),
                              sum(i*i for i in range(4))+sum(i*i for i in range(3))])[:, None, None]
    elif kind == 'field':
        field_type = type_of(q.n)
        @graph(inputs=Product(value=field_type), output=field_type,
               captures={'factor': coefficient})
        def scale(g, args, constants):
            return g.value('scaled', constants['factor']*args['value'])
        function = GraphFunction.from_data(json.loads(json.dumps(scale.to_data())),
                                           inputs=scale.inputs, output=scale.output)
        selected = function(p, value=q.n)
        expected *= coefficient
    elif kind in ('capture', 'propagate'):
        result_type = Result(Float64, Failure())
        @graph(inputs=Product(x=Float64), output=result_type)
        def reciprocal(g, args):
            return g.variant(result_type, 'ok', 1/args['x'])
        function = GraphFunction.from_data(json.loads(json.dumps(reciprocal.to_data())))
        attempted = function(p, x=p.float64(0))
        if kind == 'capture':
            marker = p.match(attempted,
                ok=lambda b, value: value,
                err=lambda b, error: b.match(error, runtime=lambda c, payload:
                    c.branch(payload['origin'] == 2,
                             lambda d: d.float64(-7), lambda d: d.float64(-99))))
            selected = p.value('handled', marker*q.n)
            expected *= -7
        else:
            marker = p.require_success(attempted)
            selected = p.value('rejected', marker*q.n)
    elif kind == 'jvp':
        from pops.graph_derivatives import jvp
        @graph(inputs=Product(x=Float64, n=Int64), output=Float64)
        def algorithm(g, args):
            return g.fold(args['x'], args['n'], lambda b, x, index:
                b.branch(x > 0, lambda c: x*x, lambda c: 1/(x-2)), max_steps=4)
        function = jvp(algorithm)
        result = function(p, primal=p.product(x=p.float64(2), n=p.int64(2)),
                          tangent=p.product(x=p.float64(1), n=p.product()))
        @graph(inputs=Product(x=Float64), output=Float64)
        def array_energy(g, args):
            domain = g.range_set(4, capacity=4, ownership='replicated')
            values = g.map(domain, lambda b, index: args['x']*b.float64(b.index_value(index)+1))
            return g.reduce(values, g.float64(0), lambda b, total, value: total+value*value,
                            order='ascending')
        energy = jvp(array_energy)(p, primal=p.product(x=p.float64(2)),
                                   tangent=p.product(x=p.float64(1)))
        factors = (result['value'], result['tangent'], energy['tangent'])
        fields = [p.value('dual_'+str(i), scalar*q.n) for i, scalar in enumerate(factors)]
        selected = p.value('forward', tuple(fields[i][i] for i in range(3)))
        expected *= np.array([2**4, 4*2**3, 2*2*sum(i*i for i in range(1, 5))])[:, None, None]
    elif kind == 'vjp':
        from pops.graph_derivatives import GraphPairings, jvp, tangent_type, vjp
        @graph(inputs=Product(x=Float64, y=Float64), output=Product(a=Float64, b=Float64))
        def equations(g, args):
            x, y = args['x'], args['y']
            return g.product(a=x*x+y, b=x*y)
        ti, to = tangent_type(equations.inputs), tangent_type(equations.output)
        output_pairing = graph(inputs=Product(value=to), output=to)(lambda b, args:
            b.product(a=2*args['value']['a'], b=3*args['value']['b']))
        input_pairing = graph(inputs=Product(value=ti), output=ti)(lambda b, args:
            b.product(x=args['value']['x']/5, y=args['value']['y']/7))
        function = vjp(equations, pairings=GraphPairings(output_pairing, input_pairing))
        primal = p.product(x=p.float64(2), y=p.float64(3))
        covector = p.product(a=p.float64(.5), b=p.float64(.25))
        pulled = function(p, primal=primal, covector=covector)['pullback']
        direction = p.product(x=p.float64(.4), y=p.float64(-.2))
        tangent = jvp(equations)(p, primal=primal, tangent=direction)['tangent']
        lhs = 2*tangent['a']*covector['a']+3*tangent['b']*covector['b']
        rhs = 5*direction['x']*pulled['x']+7*direction['y']*pulled['y']
        # This cell component is zero only if the two explicit pairings agree with transposition.
        factors = (pulled['x'], pulled['y'], lhs-rhs)
        fields = [p.value('adjoint_'+str(i), scalar*q.n) for i, scalar in enumerate(factors)]
        selected = p.value('transpose', tuple(fields[i][i] for i in range(3)))
        expected *= np.array([1.25, 2.5/7, 0.])[:, None, None]
    else:
        raise ValueError('unknown graph-function check '+kind)
    p.commit(q.next, p.value('accepted', selected, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    subject = block[state]
    def invariants(runtime):
        values = state_array(runtime, 'method', initial.shape)
        if values is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(values-expected)))
        return {'passed': bool(np.isfinite(values).all() and error < 1e-13),
                'metrics_owner': True, 'max_error': error,
                'graph_identity': function.identity, 'graph_effects': sorted(function.effects)}
    return CaseSpec(case, layout, _initial(case, subject, initial), {'method': subject},
                    .01, .01, invariants,
                    'closed serialized native GraphFunction: '+kind+'; coefficient='+str(coefficient))
