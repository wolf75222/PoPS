"""Source lowering of mandatory inventory aliases; no native execution claim."""
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_ops import _emit_op
from pops.codegen.program_source_map import source_map
from pops.contributions import Contribution, Delivery, Increment, FiniteInterval, LinearMap
from pops.graph_functions import graph
from pops.time import Program
from pops.time._program.structured_control import new_typed, value_signature
from pops.time.variant_types import Float64, Product
from typed_program_support import typed_state


@graph(inputs=Product(value=Float64), output=Float64)
def _identity(p, a):
    return a['value']


def _increment(p, base):
    interval = FiniteInterval(p.float64(0), p.float64(1), 'test_interval')
    contribution = Contribution('J', p.float64(1), interval)
    action = LinearMap.differential(_identity, at=p.product(value=base), wrt='value')
    return Increment((contribution,), (Delivery(contribution, base, action),))


def _updated(p, base):
    return _increment(p, base).apply(p, base, name='updated')


def _emit_one(p, value, env):
    lines = []
    _emit_op(p, value, None, frozenset(), env, None, lines, [],
             block_idx={}, field_plans={}, target='system')
    return lines


def test_packet_alias_preserves_the_exact_structured_binding_without_native_copy():
    p = Program('packet_alias')
    packet = _increment(p, p.float64(2)).pack(p)
    assert packet.op == 'provenance_bind'
    binding = {'J': {'value': 'source_value', 'interval': {'begin': 't0', 'end': 't1'}}}
    env = {packet.inputs[0].id: binding}
    assert _emit_one(p, packet, env) == []
    assert env[packet.id] is binding


@pytest.mark.parametrize('backend', ('principal', 'distributed_local'))
@pytest.mark.parametrize('substitution', ('old', 'double_new'))
def test_emitters_reject_copied_receipt_even_if_computed_reference_is_also_rewritten(backend, substitution):
    p = Program('tampered_alias')
    old = p.float64(2)
    alias = _updated(p, old)
    wrong = old if substitution == 'old' else alias.inputs[0]*2
    attrs = dict(alias.attrs)
    attrs['contract'] = dict(attrs['contract'], computed=wrong)
    forged = new_typed(p, value_signature(alias), 'provenance_bind', (wrong,), attrs,
                       point=alias.point)
    env = {wrong.id: 'wrong_value'}
    with pytest.raises(ValueError, match='arithmetic/control|different numerical result'):
        if backend == 'principal':
            _emit_one(p, forged, env)
        else:
            from pops.codegen.program_emit_distributed_arrays import _local
            _local((forged,), env, [])


def _finish(p, q, answer):
    p.commit(q.next, p.value('answer', answer*q.n, at=q.next.point))
    cpp = emit_cpp_program(p)
    mapping = source_map(cpp)
    assert mapping['complete'] and not mapping['unlocated']
    aliases = [row for row in mapping['operations']
               if row['op'] == 'provenance_bind' and row['channel'] != 'install']
    assert aliases
    assert all(row['status'] == 'no_direct_statements' for row in aliases)
    assert all(row['effective_ranges'] and row['binding_origins'] for row in aliases)
    return cpp, mapping, aliases


def test_scalar_alias_is_located_at_its_real_existing_binding():
    p = Program('scalar_alias')
    q = typed_state(p, 'material', state_name='U'); q.n
    _, mapping, aliases = _finish(p, q, _updated(p, p.float64(2)))
    assert len(aliases) == 1
    assert aliases[0]['location_basis'] == 'identical_emitter_binding'


def test_branch_arms_in_fold_do_not_borrow_each_others_binding_origins():
    p = Program('nested_aliases')
    q = typed_state(p, 'material', state_name='U'); q.n
    answer = p.fold(p.float64(2), 2,
        lambda b, state, index: b.branch(state > 0,
            lambda c: _updated(c, state), lambda c: _updated(c, -state)), max_steps=2)
    _, mapping, aliases = _finish(p, q, answer)
    assert len(aliases) == 2
    rows = {row['emission']: row for row in mapping['operations']}
    for alias in aliases:
        for key in alias['binding_origins']:
            origin = rows[key]
            assert alias['lexical_regions'][:len(origin['lexical_regions'])] == origin['lexical_regions']
    assert aliases[0]['cpp_binding'] != aliases[1]['cpp_binding']
    locations = [{line for first,last in alias['effective_ranges']
                  for line in range(first,last+1)} for alias in aliases]
    assert locations[0].isdisjoint(locations[1])


@pytest.mark.parametrize('ownership', ('replicated', 'distributed'))
def test_alias_inside_real_finite_map_reuses_the_local_value_and_is_source_mapped(ownership):
    p = Program('array_alias_' + ownership)
    q = typed_state(p, 'material', state_name='U'); q.n
    if ownership == 'replicated':
        domain = p.range_set(3, capacity=4, ownership='replicated')
        values = p.map(domain, lambda b, i: _updated(b, b.float64(b.index_value(i))))
        answer = p.reduce(values, p.float64(0), lambda b,s,x:s+x, order='ascending')
    else:
        from pops.time._program.distributed_arrays import Distributed
        d = Distributed(p)
        domain = d.domain((0, 1, 1, 3), capacity=4)
        values = p.map(domain, lambda b, i: _updated(b, b.float64(d.gid(i))), byte_capacity=65536)
        answer = p.reduce(values, p.float64(0), lambda b,s,x:s+x,
                          order='global_gid_ascending', byte_capacity=65536)
    cpp, mapping, aliases = _finish(p, q, answer)
    if ownership == 'distributed':
        assert aliases[0]['channel'] == 'distributed_local'
        assert all('all_reduce' not in cpp.splitlines()[line-1]
                   for first,last in aliases[0]['ranges'] for line in range(first,last+1))
