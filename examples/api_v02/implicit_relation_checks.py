"""A coupled nonlinear relation with nontrivial boundary/gauge tangents.

R=(x*x+y+b-a, x+y*y+g-c), B=b-h, G=g-k. The selected base is supplied independently
and revalidated before linearization. No Newton trace is differentiated. A case-authored dense
elimination method builds its matrix by applying the generated linear residual to basis vectors.
NumPy appears only in independent reception oracles and initial data.
"""
from __future__ import annotations

import numpy as np
from pops.graph_functions import graph
from pops.graph_derivatives import GraphPairings, tangent_type
from pops.implicit_derivatives import ImplicitRelation, implicit_jvp, implicit_vjp, linearize
from pops.relations import Accuracy, Algorithm, Approximation, Check, Evidence, Problem, RelationErrors
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, type_of
from examples.api_v02.common import CaseSpec, non_owner_metrics, state_array
from examples.api_v02.methods_T01_T12 import _initial, _storage


U = Product(x=Float64, y=Float64, b=Float64, g=Float64)
D = Product(a=Float64, c=Float64, h=Float64, k=Float64)
R = Product(first=Float64, second=Float64)
Boundary = Product(wall=Float64)
Gauge = Product(reference=Float64)
FullResidual = Product(equations=R, boundary=Boundary, gauge=Gauge)
LinearData = Product(base_unknown=U, base_data=D, direction=D)
Parameters = Product(atol=Float64, rtol=Float64, scale=Float64, request=Product())
Estimates = Product(residual_norm=Float64)
EvidenceType = Product(estimates=Estimates, valid=Bool)
Termination = Product(iterations=Int64)
Approx = Product(values=U, evidence=EvidenceType, termination=Termination)
CheckType = Product(accepted=Bool, observations=Product())
Budget = Product(steps=Int64)


def _ok(p, value):
    return p.variant(Result(type_of(value), RelationErrors), 'ok', value)


def _both(p, a, b):
    return p.branch(a, lambda q: b, lambda q: q.boolean(False))


def _flatten(value):
    return (value['equations']['first'], value['equations']['second'],
            value['boundary']['wall'], value['gauge']['reference'])


def _norm(p, values):
    result = p.float64(0.)
    for x in values:
        absolute = p.branch(x >= 0, lambda q: x, lambda q: -x)
        result = p.branch(result >= absolute, lambda q: result, lambda q: absolute)
    return result


def make_linearization(*, mode='jvp'):
    args = Product(unknown=U, data=D)
    @graph(inputs=args, output=R)
    def residual(p, a):
        u, d = a['unknown'], a['data']
        return p.product(first=u['x']*u['x']+u['y']+u['b']-d['a'],
                         second=u['x']+u['y']*u['y']+u['g']-d['c'])
    boundary = graph(inputs=args, output=Boundary)(lambda p, a:
        p.product(wall=a['unknown']['b']-a['data']['h']))
    gauge = graph(inputs=args, output=Gauge)(lambda p, a:
        p.product(reference=a['unknown']['g']-a['data']['k']))
    true = graph(inputs=args, output=Bool)(lambda p, a: p.boolean(True))
    selected = graph(inputs=args, output=Bool)(lambda p, a:
        _both(p, a['unknown']['x'] > 0, a['unknown']['y'] > 0))
    original_request = graph(inputs=Product(unknown=U, data=D, residual=R,
        accuracy=Parameters, evidence=EvidenceType), output=Result(CheckType, RelationErrors))(
        lambda p, a: _ok(p, Check(p, accepted=p.boolean(True), observations=p.product())))
    validity_type = Product(differentiable=Bool, regular=Bool, observations=Product(determinant=Float64))
    @graph(inputs=args, output=validity_type)
    def validity(p, a):
        determinant = 4*a['unknown']['x']*a['unknown']['y']-1
        return p.product(differentiable=p.boolean(mode != 'nondifferentiable'),
                         regular=determinant > 1e-8,
                         observations=p.product(determinant=determinant))
    relation = ImplicitRelation(Problem(residual, true, selected, true, original_request),
                                boundary, gauge, validity)
    tangent_request = graph(inputs=Product(unknown=U, data=LinearData, residual=FullResidual,
        accuracy=Parameters, evidence=EvidenceType), output=Result(CheckType, RelationErrors))(
        lambda p, a: _ok(p, Check(p, accepted=p.boolean(True), observations=p.product())))
    return linearize(relation, request_test=tangent_request)


def dense_tangent_algorithm(linearization, *, dishonest=False, omit_constraints=False):
    """A supplied numerical method over four named finite coordinates; no copied Jacobian law."""
    names = ('x', 'y', 'b', 'g')
    @graph(inputs=Product(data=LinearData, guess=U, accuracy=Parameters, budget=Budget),
           output=Result(Approx, RelationErrors))
    def algorithm(p, args):
        def evaluate(q):
            zero = q.product(**{key: q.float64(0.) for key in names})
            constant = _flatten(linearization.problem.residual(q, unknown=zero, data=args['data']))
            if dishonest:
                values = zero
            else:
                columns = []
                for j in range(4):
                    basis = q.product(**{key: q.float64(1. if i == j else 0.) for i, key in enumerate(names)})
                    image = _flatten(linearization.problem.residual(q, unknown=basis, data=args['data']))
                    columns.append([image[i]-constant[i] for i in range(4)])
                matrix = [[columns[j][i] for j in range(4)] for i in range(4)]
                rhs = [-x for x in constant]
                if omit_constraints:
                    rhs[2], rhs[3] = q.float64(0.), q.float64(0.)
                # Fixed-order elimination is a declared method profile. The validity predicate
                # and this witness's matrix ensure nonzero pivots; there is no pivoting claim.
                for k in range(4):
                    for i in range(k+1, 4):
                        factor = matrix[i][k]/matrix[k][k]
                        rhs[i] = rhs[i]-factor*rhs[k]
                        for j in range(k+1, 4):
                            matrix[i][j] = matrix[i][j]-factor*matrix[k][j]
                answer = [None]*4
                for i in reversed(range(4)):
                    answer[i] = (rhs[i]-sum((matrix[i][j]*answer[j] for j in range(i+1, 4)), q.float64(0.)))/matrix[i][i]
                values = q.product(**dict(zip(names, answer)))
            residual = linearization.problem.residual(q, unknown=values, data=args['data'])
            # A dishonest zero estimate cannot bypass relations.solve's original-residual test.
            estimate = q.float64(0.) if dishonest or omit_constraints else _norm(q, _flatten(residual))
            evidence = Evidence(q, estimates=q.product(residual_norm=estimate), valid=q.boolean(True))
            return _ok(q, Approximation(q, values=values, evidence=evidence,
                                       termination=q.product(iterations=q.int64(4))))
        def exhausted(q):
            return q.variant(Result(Approx, RelationErrors), 'err',
                q.variant(RelationErrors, 'relation', q.product(code=q.int64(9401))))
        return p.branch(args['budget']['steps'] >= 4, evaluate, exhausted)
    return Algorithm(algorithm)


def _parameters(p, tolerance=1e-11):
    return p.product(atol=p.float64(tolerance), rtol=p.float64(0.), scale=p.float64(1.), request=p.product())


def independent_oracle(base, direction, dual):
    """Analytic Jacobian + independent dense NumPy linear algebra for reception only."""
    x, y, b, g = base
    jacobian = np.array([[2*x, 1., 1., 0.], [1., 2*y, 0., 1.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
    tangent = np.linalg.solve(jacobian, direction)
    input_metric, output_metric = np.array([11., 13., 17., 19.]), np.array([2., 3., 5., 7.])
    pullback = np.linalg.solve(jacobian.T, output_metric*dual)/input_metric
    defect = float(tangent @ (output_metric*dual)-direction @ (input_metric*pullback))
    return tangent, pullback, defect


def build_implicit_check(mode='jvp', *, recover=True):
    case, layout, model, block, state, p, q, initial = _storage(
        'implicit_'+mode, ('a', 'b', 'c', 'd', 'duality'), (1., 2., 3., 4., 1.), cells=(8, 4), sampling='cell_average')
    q.n
    linearization = make_linearization(mode=mode)
    algorithm = dense_tangent_algorithm(linearization, dishonest=mode == 'bad_tangent',
                                         omit_constraints=mode == 'omit_constraints')
    base_coordinates = np.array([.5, .5, .5, -.25]) if mode == 'singular' else np.array([2., 3., .5, -.25])
    x, y, b, g = map(float, base_coordinates)
    data_coordinates = [x*x+y+b, x+y*y+g, b, g]
    data = p.product(**{name: p.float64(value) for name, value in zip(('a', 'c', 'h', 'k'), data_coordinates)})
    proposed = base_coordinates.copy()
    if mode == 'bad_base': proposed[0] += .1
    if mode == 'bad_gauge': proposed[3] += .1
    values = p.product(**{name: p.float64(float(value)) for name, value in zip(('x', 'y', 'b', 'g'), proposed)})
    base = p.product(values=values, data_identity=data,
        evidence=Evidence(p, estimates=p.product(residual_norm=p.float64(0.)), valid=p.boolean(True)),
        termination=p.product(iterations=p.int64(0)))
    norm = graph(inputs=Product(residual=FullResidual), output=Float64)(lambda z, a: _norm(z, _flatten(a['residual'])))
    accuracy = Accuracy(_parameters(p), norm)
    base_accuracy = Accuracy(_parameters(p), norm)
    budget = p.product(steps=p.int64(0 if mode == 'budget' else 4))
    direction_values, dual_values = np.array([.3, -.2, .4, .1]), np.array([.2, -.3, .5, -.7])
    direction = p.product(**{name: p.float64(float(value)) for name, value in zip(('a', 'c', 'h', 'k'), direction_values)})
    guess = p.product(**{name: p.float64(0.) for name in ('x', 'y', 'b', 'g')})
    if mode == 'vjp':
        output_map = graph(inputs=Product(value=U), output=U)(lambda z, a:
            z.product(**{name: weight*a['value'][name] for name, weight in zip(('x', 'y', 'b', 'g'), (2., 3., 5., 7.))}))
        input_map = graph(inputs=Product(value=D), output=D)(lambda z, a:
            z.product(**{name: a['value'][name]/weight for name, weight in zip(('a', 'c', 'h', 'k'), (11., 13., 17., 19.))}))
        dual = p.product(**{name: p.float64(float(value)) for name, value in zip(('x', 'y', 'b', 'g'), dual_values)})
        result = implicit_vjp(p, linearization, base=base, covector=dual,
            pairings=GraphPairings(output_map, input_map), algorithm=algorithm, guess=guess,
            base_accuracy=base_accuracy, accuracy=accuracy, budget=budget)
        names = ('a', 'c', 'h', 'k')
        expected_factors = independent_oracle(base_coordinates, direction_values, dual_values)[1]
    else:
        result = implicit_jvp(p, linearization, base=base, direction=direction,
            algorithm=algorithm, guess=guess, base_accuracy=base_accuracy, accuracy=accuracy, budget=budget)
        names = ('x', 'y', 'b', 'g')
        expected_factors = (independent_oracle(base_coordinates, direction_values, dual_values)[0]
                            if mode == 'jvp' else None)
    if expected_factors is None and recover:
        codes = {'bad_base': 2107, 'bad_gauge': 2107, 'bad_tangent': 2107, 'omit_constraints': 2107,
                 'singular': 9302, 'nondifferentiable': 9301, 'budget': 9401}
        factor = p.match(result, ok=lambda z, value: z.float64(999), err=lambda z, error:
            z.match(error, implicit=lambda a, payload: -a.float64(payload['code']),
                    relation=lambda a, payload: -a.float64(payload['code']),
                    runtime=lambda a, payload: a.float64(-9999)))
        selected = p.value('recovered', factor*q.n)
        expected = initial*(-codes[mode])
    else:
        receipt = p.require_success(result)
        accepted = receipt['values']
        duality = p.float64(0.)
        if mode == 'vjp':
            tangent = {name: sum((float(direction_values[j])*receipt['columns']['column_'+str(j)]['values'][name]
                                  for j in range(4)), p.float64(0.)) for name in ('x', 'y', 'b', 'g')}
            lhs = sum((weight*float(dual_values[i])*tangent[name] for i, (name, weight) in
                       enumerate(zip(('x', 'y', 'b', 'g'), (2., 3., 5., 7.)))), p.float64(0.))
            rhs = sum((weight*float(direction_values[i])*accepted[name] for i, (name, weight) in
                       enumerate(zip(('a', 'c', 'h', 'k'), (11., 13., 17., 19.)))), p.float64(0.))
            duality = lhs-rhs
        selected = p.value('implicit_derivative', tuple(q.n[i]*accepted[name] for i, name in enumerate(names))
                           +(q.n[4]*duality,))
        expected = initial*np.r_[expected_factors, 0.][:, None, None] if expected_factors is not None else initial.copy()
    p.commit(q.next, p.value('accepted', selected, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    subject = block[state]
    def invariants(runtime):
        actual = state_array(runtime, 'method', initial.shape)
        if actual is None: return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        return {'passed': bool(np.isfinite(actual).all() and error < 3e-11), 'max_error': error,
                'metrics_owner': True, 'linearization_identity': linearization.identity}
    return CaseSpec(case, layout, _initial(case, subject, initial), {'method': subject}, .01, .01,
                    invariants, 'finite implicit '+mode+': original residual + boundary + gauge; no Newton-trace AD')
