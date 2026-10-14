"""Closed function data specializes actual Program IR; no callback is retained at calls."""
import copy
import json
import weakref

import pytest

from pops.graph_functions import GraphFunction, call, graph
from pops.time import Program
from pops.time.variant_types import Bool, Failure, Float64, Int64, Product, Result, type_of
from typed_program_support import typed_state


def _roundtrip(function):
    return GraphFunction.from_data(json.loads(json.dumps(function.to_data())))


def test_builder_runs_once_and_is_not_retained_after_serialization():
    visits = []
    def builder(p, a):
        visits.append(1)
        return 2*a['x']+1
    reference = weakref.ref(builder)
    f = graph(inputs=Product(x=Float64), output=Float64)(builder)
    del builder
    assert reference() is None
    loaded = _roundtrip(f)
    p = Program('caller')
    a, b = loaded(p, x=p.float64(3)), loaded(p, x=p.float64(4))
    assert visits == [1]
    assert a.id != b.id and a.prog is p and b.prog is p
    assert f.identity == loaded.identity
    assert all(v.op != 'graph_argument' for v in p._values)


def test_formal_product_and_keyword_call_have_the_same_function():
    @graph(inputs=Product(x=Float64, count=Int64), output=Product(y=Float64, count=Int64))
    def f(p, a):
        return p.product(y=a['x']*3, count=a['count']+1)
    p = Program('caller')
    args = p.product(x=p.float64(2), count=p.int64(4))
    result = call(_roundtrip(f), args)
    assert result.vtype == 'product'
    assert result['count'].vtype == 'int64'


def test_nested_graph_calls_are_inlined_in_the_stored_graph():
    @graph(inputs=Product(x=Float64), output=Float64)
    def square(p, a):
        return a['x']*a['x']
    @graph(inputs=Product(x=Float64), output=Float64)
    def composed(p, a):
        return square(p, x=square(p, x=a['x']))+1
    snapshot = composed.to_data()
    assert len([v for v in snapshot['nodes'] if v['op'] == 'scalar_op']) == 3
    assert _roundtrip(composed).identity == composed.identity


def test_capture_values_are_frozen_and_affect_equation_identity():
    values = {'coefficient': [2., 3.]}
    @graph(inputs=Product(x=Float64), output=Float64, captures=values)
    def f(p, a, constants):
        with pytest.raises(TypeError):
            constants['coefficient'][0] = 8
        return a['x']*constants['coefficient'][0]+constants['coefficient'][1]
    before = f.to_data()
    values['coefficient'][0] = 99
    assert f.to_data() == before
    assert _roundtrip(f).to_data() == before


@pytest.mark.parametrize('wrong', ['missing', 'extra', 'type', 'owner'])
def test_argument_failure_is_atomic(wrong):
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        return a['x']*2
    p = Program('caller')
    value = p.float64(2)
    kwargs = dict(x=value)
    if wrong == 'missing': kwargs = {}
    elif wrong == 'extra': kwargs['y'] = value
    elif wrong == 'type': kwargs['x'] = p.int64(2)
    elif wrong == 'owner': kwargs['x'] = Program('foreign').float64(2)
    before = (p._next_id, len(p._values))
    with pytest.raises((TypeError, ValueError)):
        f(p, **kwargs)
    assert (p._next_id, len(p._values)) == before


def test_foreign_live_capture_and_ambient_step_are_rejected():
    foreign = Program('foreign').float64(2)
    with pytest.raises(ValueError):
        graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: a['x']+foreign)
    with pytest.raises(ValueError, match='effect contract'):
        graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: p.requested_dt())
    with pytest.raises((TypeError, ValueError)):
        graph(inputs=Product(x=Float64), output=Float64, captures={'bad': foreign})


def test_effects_are_inferred_and_cannot_be_erased_by_declaration():
    with pytest.raises(ValueError, match='declared effects'):
        graph(inputs=Product(x=Float64), output=Float64, effects=())(lambda p, a: 1/a['x'])
    f = graph(inputs=Product(x=Float64), output=Float64,
              effects=('failure', 'collective'))(lambda p, a: 1/a['x'])
    assert f.effects == {'failure', 'collective'}
    data = f.to_data()
    data['effects'] = []
    with pytest.raises(ValueError, match='effect summary'):
        GraphFunction.from_data(data)


def test_unknown_effectful_operation_is_rejected_including_dead_output():
    def body(p, a):
        p.record_scalar('hidden_side_effect', a['x'])
        return a['x']
    with pytest.raises(ValueError, match='record_scalar'):
        graph(inputs=Product(x=Float64), output=Float64)(body)


def test_field_authorities_roundtrip_with_explicit_witnesses_and_no_name_lookup():
    p = Program('field')
    left = typed_state(p, 'left', state_name='U')
    right = typed_state(p, 'right', state_name='U')
    value = left.n
    T = type_of(value)
    @graph(inputs=Product(value=T), output=T)
    def twice(p, a):
        return p.value('twice', 2*a['value'])
    data = json.loads(json.dumps(twice.to_data()))
    with pytest.raises(ValueError, match='authority reconstruction'):
        GraphFunction.from_data(data)
    loaded = GraphFunction.from_data(data, inputs=twice.inputs, output=twice.output)
    result = loaded(p, value=value)
    assert result.state_ref is value.state_ref and result.space is value.space
    with pytest.raises(TypeError, match='authority'):
        loaded(p, value=right.n)


def test_dependent_scalar_point_is_not_erased():
    p = Program('caller')
    v = p.float64(1)
    T = type_of(v)
    f = graph(inputs=Product(x=T), output=T)(lambda q, a: a['x'])
    assert f(p, x=v).point == v.point
    other = Program('other')
    with pytest.raises(TypeError, match='authority'):
        f(other, x=other.float64(1))


def test_dependent_and_abstract_scalar_arguments_share_declared_evaluation_context():
    p = Program('caller')
    state = typed_state(p, 'field')
    norm = p.norm_inf(state)
    T = type_of(norm)
    f = graph(inputs=Product(left=T, right=Float64), output=T)(
        lambda q, a: a['left']*a['right'])
    result = f(p, left=norm, right=p.float64(2))
    assert result.point == norm.point and result.state_ref == norm.state_ref


def test_nested_fold_regions_are_fresh_on_every_call():
    @graph(inputs=Product(k=Int64), output=Int64)
    def f(p, a):
        return p.fold(a['k'], 3, lambda q, x, i:
                      q.fold(x, 2, lambda r, y, j: y+i+j))
    f = _roundtrip(f)
    p = Program('caller')
    a = f(p, k=p.int64(0))
    b = f(p, k=p.int64(1))
    assert a.attrs['body_region'] != b.attrs['body_region']
    assert a.attrs['body'].attrs['body_region'] != b.attrs['body'].attrs['body_region']
    assert a.attrs['body_argument'].prog is p


def test_call_inside_region_cannot_leak_result_or_reuse_other_region_values():
    f = graph(inputs=Product(k=Int64), output=Int64)(lambda p, a: a['k']+1)
    p = Program('caller')
    leaked = []
    def body(q, x, i):
        y = f(q, k=x)
        leaked.append(y)
        return y
    p.fold(p.int64(0), 2, body)
    before = p._next_id
    with pytest.raises(ValueError, match='region'):
        f(p, k=leaked[0])
    assert p._next_id == before


def test_typed_result_is_a_real_capture_boundary_without_nested_result():
    T = Result(Float64, Failure())
    @graph(inputs=Product(x=Float64), output=T)
    def reciprocal(p, a):
        return p.variant(T, 'ok', 1/a['x'])
    loaded = _roundtrip(reciprocal)
    assert loaded.effects == {'collective'}
    p = Program('caller')
    result = loaded(p, x=p.float64(0))
    assert result.op == 'capture_result' and result.vtype == 'variant'
    assert result.attrs['success_signature']['vtype'] == 'scalar'
    assert result.attrs['body_is_variant_result'] is True


@pytest.mark.parametrize('corruption', ['ref', 'duplicate', 'operation', 'effect', 'output', 'type'])
def test_corrupted_closed_data_is_rejected_before_call(corruption):
    f = graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: a['x']*2)
    data = copy.deepcopy(f.to_data())
    if corruption == 'ref': data['nodes'][0]['inputs'] = [999]
    elif corruption == 'duplicate': data['nodes'].append(copy.deepcopy(data['nodes'][-1]))
    elif corruption == 'operation': data['nodes'][0]['op'] = 'python_callback'
    elif corruption == 'effect': data['nodes'][-1]['effects'] = []
    elif corruption == 'output': data['result'] = 999
    elif corruption == 'type':
        data['nodes'][0]['signature']['$map'][0][1] = 'bool'
        data['nodes'][0]['vtype'] = 'bool'
    with pytest.raises((KeyError, TypeError, ValueError)):
        GraphFunction.from_data(data)


def test_output_type_cannot_silently_change():
    with pytest.raises(TypeError, match='declared output'):
        graph(inputs=Product(x=Float64), output=Bool)(lambda p, a: a['x'])


def test_finite_domains_created_in_function_are_alpha_renamed_on_each_call():
    @graph(inputs=Product(n=Int64), output=Int64)
    def sum_squares(p, a):
        domain = p.range_set(a['n'], capacity=8, ownership='replicated')
        values = p.map(domain, lambda q, i: q.index_value(i)*q.index_value(i))
        return p.reduce(values, p.int64(0), lambda q, x, y: x+y, order='ascending')
    f = _roundtrip(sum_squares)
    p = Program('caller')
    existing = p.range_set(2, capacity=8, ownership='replicated')
    a = f(p, n=p.int64(3))
    b = f(p, n=p.int64(4))
    domains = [v for v in p._values if v.op == 'finite_range_set']
    assert len({v.attrs['signature']['family'] for v in domains}) == 3
    assert domains[0] is existing and a.id != b.id
    # Native checked indices retain their own domain, never a shape-only compatibility.
    maps = [v for v in p._values if v.op == 'finite_map']
    with pytest.raises(ValueError, match='domain family'):
        p.at(maps[0], p.index(domains[2], 0))


@pytest.mark.parametrize("field", ("attrs", "signature"))
@pytest.mark.parametrize("encoded", (None, {"$float": "0x1.0000000000000p+0"}, {"$tuple": []}))
def test_closed_operation_requires_mapping_payloads(field, encoded):
    function = graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: a['x']*2)
    snapshot = function.to_data()
    snapshot['nodes'][-1][field] = encoded
    with pytest.raises(ValueError, match="must decode to.*mapping"):
        GraphFunction.from_data(snapshot)


def test_untyped_builder_output_is_rejected_before_snapshot():
    with pytest.raises(TypeError, match="must return a ProgramValue"):
        graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: object())
