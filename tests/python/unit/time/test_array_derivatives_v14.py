"""Array AD is a closed Program transform, not a Python evaluator or size unroll."""
import json

import pytest

from pops.graph_derivatives import GraphPairings, jvp, tangent_type, vjp
from pops.graph_functions import GraphFunction, graph
from pops.time import Program
from pops.time.variant_types import Float64, Product, type_of


def _fixture(capacity=8):
    p = Program('array_derivative')
    domain = p.range_set(p.int64(3), capacity=capacity, ownership='replicated')
    values = p.map(domain, lambda q, i: q.float64(q.index_value(i)+1))
    @graph(inputs=Product(x=type_of(values)), output=Float64)
    def energy(q, args):
        return q.reduce(args['x'], q.float64(0), lambda r, acc, x: acc+x*x, order='ascending')
    ti, to = tangent_type(energy.inputs), tangent_type(energy.output)
    pairings = GraphPairings(graph(inputs=Product(value=to), output=to)(lambda q,a: a['value']),
                            graph(inputs=Product(value=ti), output=ti)(lambda q,a: a['value']))
    return p, domain, values, energy, pairings


def _ops(snapshot):
    def walk(nodes):
        for node in nodes:
            yield node['op']
            for block in node['blocks'].values(): yield from walk(block)
    return list(walk(snapshot['nodes']))


def test_native_array_vjp_is_closed_and_cardinality_independent():
    snapshots = []
    for capacity in (8, 10000):
        p, domain, values, f, pairings = _fixture(capacity)
        derivative = vjp(f, pairings=pairings)
        restored = GraphFunction.from_data(json.loads(json.dumps(derivative.to_data())),
            inputs=derivative.inputs, output=derivative.output)
        result = restored(p, primal=p.product(x=values), covector=p.float64(1))
        assert result['pullback']['x'].vtype == 'array'
        snapshots.append(_ops(derivative.to_data()))
    assert snapshots[0] == snapshots[1]
    assert 'finite_same_domain' in snapshots[0]
    assert 'finite_map' in snapshots[0] and 'structured_fold' in snapshots[0]


def test_domain_identity_guard_is_emitted_before_any_array_loop():
    p, domain, values, f, unused = _fixture()
    derivative = jvp(f)
    # The guard is a real SSA node even when the actual runtime cardinality is zero.
    nodes = derivative.to_data()['nodes']
    def execution_nodes(records):
        for node in records:
            yield node
            for body in node['blocks'].values(): yield from execution_nodes(body)
    ops = [node['op'] for node in execution_nodes(nodes)]
    assert 'finite_same_domain' in ops
    assert ops.index('finite_same_domain') < ops.index('finite_map')


def test_discrete_indices_have_no_real_direction_and_no_phantom_coordinates():
    p, domain, values, f, pairings = _fixture()
    indices = p.map(domain, lambda q,i: q.index(domain,q.index_value(i)))
    tangent = tangent_type(type_of(indices)).signature
    assert tangent['element']['vtype'] == 'product'
    assert not tangent['element']['members']


def test_dynamic_transpose_lowers_to_actual_program_cpp():
    from pops.codegen.program_codegen import emit_cpp_program
    from typed_program_support import typed_state
    p, domain, values, f, pairings = _fixture()
    state = typed_state(p, 'field', state_name='U')
    result = vjp(f, pairings=pairings)(p, primal=p.product(x=values), covector=p.float64(1))
    total = p.reduce(result['pullback']['x'], p.float64(0), lambda q,a,b:a+b, order='ascending')
    p.record_scalar('gradient_sum', total)
    p.commit(state.next, p.value('accepted', state.n, at=state.next.point))
    cpp = emit_cpp_program(p)
    assert 'pops_program_abi_key' in cpp
    assert 'pops_algebra_finite' in cpp and '91002' in cpp


def test_variant_tag_is_a_fixed_discrete_part_of_the_tangent():
    from pops.time.variant_types import Variant
    typ = Variant(first=Float64, second=Product(x=Float64))
    @graph(inputs=Product(value=typ), output=Float64)
    def function(p, args):
        return p.match(args['value'], first=lambda q,x:x*x, second=lambda q,v:3*v['x'])
    ti, to = tangent_type(function.inputs), tangent_type(function.output)
    pairings = GraphPairings(graph(inputs=Product(value=to), output=to)(lambda q,a:a['value']),
                            graph(inputs=Product(value=ti), output=ti)(lambda q,a:a['value']))
    derivative = vjp(function, pairings=pairings)
    p = Program('variant_array_ad')
    value = p.variant(typ, 'first', p.float64(2))
    result = derivative(p, primal=p.product(value=value), covector=p.float64(1))
    assert result['pullback']['value'].vtype == 'variant'


@pytest.mark.parametrize('count', (0, 1, 3, 7))
def test_real_case_builder_resolves_without_model_dispatch(count):
    import pops
    from examples.api_v02.array_derivative_v14 import build_array_derivative_v14
    spec = build_array_derivative_v14(count)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert plan is not None


def test_finite_iterate_derivative_requires_explicit_trace_contract_and_native_stop_guard():
    from pops.graph_derivatives import jvp
    from pops.time.variant_types import Float64
    from pops.codegen.program_codegen import emit_cpp_program
    from typed_program_support import typed_state
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        result = p.iterate(a['x'], lambda q, x: x*x/4,
                           lambda q, x: x<.05, max_steps=8)
        return p.require_success(result)['state']
    derivative = jvp(f, differentiate_iterations=True)
    p = Program('finite_iteration_derivative')
    state = typed_state(p, 'field', state_name='U')
    result = derivative(p, primal=p.product(x=p.float64(.4)),
                         tangent=p.product(x=p.float64(1.)))
    p.record_scalar('tangent', result['tangent'])
    p.commit(state.next, p.value('accepted', state.n, at=state.next.point))
    cpp = emit_cpp_program(p)
    assert '91001' in cpp
    assert 'while (' in cpp or 'for (' in cpp


def test_finite_iterate_derivative_retains_budget_failure_fallback():
    from pops.graph_derivatives import jvp
    from pops.time.variant_types import Float64
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        result = p.iterate(a['x'], lambda q, x: x*x/4,
                           lambda q, x: x<.05, max_steps=0)
        return p.match(result, ok=lambda q, item: item['state'],
                       err=lambda q, failure: a['x']*3)
    derivative = jvp(f, differentiate_iterations=True)
    assert derivative.output == Product(value=Float64, tangent=Float64)


def test_typed_result_capture_retains_real_error_payload_derivative():
    from pops.graph_derivatives import jvp
    from pops.time.variant_types import Float64, Failure, Result
    errors = Failure(problem=Product(value=Float64))
    result = Result(Float64, errors)
    @graph(inputs=Product(x=Float64), output=result)
    def f(p, a):
        return p.branch(a['x']>0,
            lambda q:q.variant(result,'ok',a['x']*a['x']),
            lambda q:q.variant(result,'err',q.variant(errors,'problem',q.product(value=3*a['x']))))
    derivative = jvp(f)
    assert derivative.output == Product(value=result, tangent=tangent_type(result))
    assert 'capture_result' in str(derivative.to_data())


def test_typed_failure_iterate_can_be_differentiated_as_a_finite_algorithm():
    from pops.graph_derivatives import jvp
    from pops.time.variant_types import Float64, Failure, Result
    errors = Failure(problem=Product(value=Float64))
    result = Result(Float64, errors)
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        iteration = p.iterate(a['x'],
            lambda q,x:q.variant(result,'ok',x*x/4), lambda q,x:x<.05,max_steps=8)
        return p.require_success(iteration)['state']
    derivative = jvp(f,differentiate_iterations=True)
    assert derivative.output == Product(value=Float64,tangent=Float64)


def test_second_derivative_of_guarded_branch_remains_a_closed_native_graph():
    from pops.graph_derivatives import jvp
    from pops.time.variant_types import Float64
    @graph(inputs=Product(x=Float64),output=Float64)
    def f(p,a):
        return p.branch(a['x']>0,lambda q:a['x']**3,lambda q:a['x']**2)
    first = jvp(f)
    second = jvp(first)
    assert second.output == Product(value=first.output,tangent=tangent_type(first.output))


@pytest.mark.parametrize('mode',('iteration','budget','surface','typed_error','second'))
def test_complete_finite_algorithm_derivative_resolves_without_carrier(mode):
    import pops
    from examples.api_v02.finite_iteration_ad_v14 import build_finite_iteration_ad_v14
    spec=build_finite_iteration_ad_v14(mode)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert not plan.time._block_indices() and not spec.initial_values
