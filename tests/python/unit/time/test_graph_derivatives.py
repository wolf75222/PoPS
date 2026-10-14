"""Source/compiler contracts for derivatives of real closed Program graphs."""
import json

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.graph_functions import GraphFunction, graph
from pops.graph_derivatives import GraphPairings, jvp, tangent_type, vjp
from pops.time import Program
from pops.time.variant_types import Float64, Int64, Product
from typed_program_support import typed_state


def _cpp(function, make_args):
    loaded = GraphFunction.from_data(json.loads(json.dumps(function.to_data())))
    p = Program('derivative')
    state = typed_state(p, 'field', state_name='U')
    result = loaded(p, **make_args(p))
    leaf = result['tangent'] if 'tangent' in dict(function.output.signature['members']) else result['pullback']['x']
    if leaf.vtype == 'product': leaf = leaf['x']
    p.record_scalar('derivative', leaf)
    p.commit(state.next, p.value('accepted', state.n, at=state.next.point))
    return emit_cpp_program(p)


def test_derivative_is_closed_graph_without_retaining_original_builder():
    visits = []
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        visits.append(1)
        return a['x']*a['x']+3*a['x']
    derivative = jvp(f)
    cpp = _cpp(derivative, lambda p: dict(primal=p.product(x=p.float64(2)),
                                          tangent=p.product(x=p.float64(1))))
    assert visits == [1]
    assert 'ctx.record_scalar("derivative"' in cpp
    assert 'graph_argument' not in cpp
    assert derivative.to_data()['nodes'] != f.to_data()['nodes']


def test_lazy_branch_derivative_checks_switching_surface_in_native_control():
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        return p.branch(a['x'] > 0, lambda q: a['x']*a['x'], lambda q: 1/a['x'])
    cpp = _cpp(jvp(f), lambda p: dict(primal=p.product(x=p.float64(2)),
                                     tangent=p.product(x=p.float64(1))))
    assert '91001' in cpp
    assert 'branch' in cpp and 'StepAttemptRejected' in cpp
    # The native compiler retains control regions, not eager evaluation of both derivatives.
    assert cpp.count('} else {') >= 2


def test_fold_carries_tangent_and_discrete_count_has_no_direction():
    @graph(inputs=Product(x=Float64, n=Int64), output=Float64)
    def f(p, a):
        return p.fold(a['x'], a['n'], lambda q, x, i: x*x, max_steps=4)
    derivative = jvp(f)
    cpp = _cpp(derivative, lambda p: dict(primal=p.product(x=p.float64(2), n=p.int64(2)),
        tangent=p.product(x=p.float64(1), n=p.product())))
    assert 'for (; iteration_' in cpp
    p = Program('wrong_direction')
    with pytest.raises(TypeError, match='different type'):
        derivative(p, primal=p.product(x=p.float64(2), n=p.int64(2)),
                   tangent=p.product(x=p.float64(1), n=p.float64(1)))


@pytest.mark.parametrize('mode', ['map_reduce', 'scan', 'gather'])
def test_replicated_array_transform_keeps_domain_identity_and_native_loops(mode):
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        domain = p.range_set(4, capacity=4, ownership='replicated')
        values = p.map(domain, lambda q, i: a['x']*q.float64(q.index_value(i)+1))
        if mode == 'scan':
            return p.scan(domain, a['x'], lambda q, carry, i:
                q.product(state=carry*q.at(values, i), value=carry), order='ascending')['state']
        if mode == 'gather':
            indices = p.map(domain, lambda q, i: q.index(domain, 3-q.index_value(i)))
            values = p.gather(values, indices)
        return p.reduce(values, p.float64(0), lambda q, carry, value: carry+value*value,
                        order='ascending')
    derivative = jvp(f)
    cpp = _cpp(derivative, lambda p: dict(primal=p.product(x=p.float64(2)),
                                          tangent=p.product(x=p.float64(1))))
    assert 'pops_algebra_finite' in cpp
    assert 'graph_argument' not in cpp


def test_elementary_expression_uses_original_dag_derivative_rules():
    import pops.math as pm
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        return pm.exp(a['x'])+pm.sin(a['x'])+pm.log(a['x'])+a['x']**3
    cpp = _cpp(jvp(f), lambda p: dict(primal=p.product(x=p.float64(2)),
                                     tangent=p.product(x=p.float64(1))))
    assert 'std::exp' in cpp and 'Kokkos::cos' in cpp


def test_vjp_requires_explicit_closed_pairings_and_serializes_transposed_columns():
    @graph(inputs=Product(x=Float64, y=Float64), output=Product(a=Float64, b=Float64))
    def f(p, args):
        x, y = args['x'], args['y']
        return p.product(a=x*x+y, b=x*y)
    ti, to = tangent_type(f.inputs), tangent_type(f.output)
    output_pairing = graph(inputs=Product(value=to), output=to)(
        lambda p, args: p.product(a=2*args['value']['a'], b=3*args['value']['b']))
    input_pairing = graph(inputs=Product(value=ti), output=ti)(
        lambda p, args: p.product(x=args['value']['x']/5, y=args['value']['y']/7))
    with pytest.raises(TypeError): vjp(f, pairings=None)
    derivative = vjp(f, pairings=GraphPairings(output_pairing, input_pairing))
    cpp = _cpp(derivative, lambda p: dict(primal=p.product(x=p.float64(2), y=p.float64(3)),
        covector=p.product(a=p.float64(.5), b=p.float64(.25))))
    assert 'ctx.record_scalar("derivative"' in cpp


def test_convergence_control_is_not_misrepresented_as_implicit_differentiation():
    @graph(inputs=Product(x=Float64), output=Float64)
    def solve_trace(p, args):
        result = p.iterate(args['x'], lambda q, x: x/2, lambda q, x: x<.01, max_steps=4)
        return p.require_success(result)['state']
    with pytest.raises(NotImplementedError, match='structured_iterate'):
        jvp(solve_trace)


def test_conversion_from_continuous_unknown_to_integer_is_not_differentiable():
    @graph(inputs=Product(x=Float64), output=Int64)
    def integer(p, args): return p.int64(args['x'])
    with pytest.raises(NotImplementedError, match='int64_cast'):
        jvp(integer)
