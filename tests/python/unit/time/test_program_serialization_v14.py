"""Canonical graph bytes and hooks, independently of compilation or native runtime."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from pops.time import Program
from pops.time.canonical_data import _json_ready
from pops.time._program.detach import detach_compiled_program
from pops.time._program.serialization import _ProgramSerialization
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Int64, Result


REGIONS = ('branch', 'structured_fold', 'structured_iterate', 'capture_result',
           'finite_map', 'finite_scan', 'result_match', 'distributed_map')


def region_program(kind, coefficient=3):
    """Eight real frontend region constructors, with nested control/metadata."""
    p = Program('serialization_' + kind)
    seed = p.int64(2)
    def nested(b, value):
        return b.fold(value, 2, lambda c, state, k: c.branch(
            k == 0, lambda z: state + coefficient, lambda z: state - 1))
    if kind == 'branch':
        p.branch(seed > 0, lambda b: nested(b, seed), lambda b: b.int64(-1))
    elif kind == 'structured_fold':
        p.fold(seed, 2, lambda b, state, k: nested(b, state))
    elif kind == 'structured_iterate':
        p.iterate(seed, lambda b, state: nested(b, state),
                  lambda b, state: state >= 6, max_steps=3)
    elif kind == 'capture_result':
        p.capture(lambda b: nested(b, b.require_success(b.success(seed))))
    elif kind in ('finite_map', 'finite_scan'):
        domain = p.range_set(3, capacity=4, ownership='replicated')
        if kind == 'finite_map':
            p.map(domain, lambda b, i: nested(b, b.index_value(i)))
        else:
            p.scan(domain, seed, lambda b, state, i: b.product(
                state=nested(b, state), value=b.index_value(i)), order='ascending')
    elif kind == 'result_match':
        p.match(p.failure(seed, 17), ok=lambda b, value: nested(b, value),
                err=lambda b, error: nested(b, error['code']))
    elif kind == 'distributed_map':
        distributed = Distributed(p)
        domain = distributed.domain((0, 1, 1, 3), capacity=4)
        distributed.local_map(domain, lambda b, i: b.variant(Result(Int64, Int64), 'ok',
            nested(b, distributed.gid(i))), byte_capacity=65536)
    else:
        raise AssertionError(kind)
    return p


def _json(value, *, sorted_keys=False):
    return json.dumps(value, sort_keys=sorted_keys, separators=(',', ':'), allow_nan=False)


def _nodes(rows):
    for row in rows:
        yield row
        for key in ('cond_block', 'body_block', 'true_block', 'false_block', 'apply_block', 'residual_block'):
            block = row['attrs'].get(key)
            if isinstance(block, list):
                yield from _nodes(block)


@pytest.mark.parametrize('kind', ('branch', 'structured_fold', 'structured_iterate', 'capture_result',
                                  'finite_map', 'finite_scan', 'result_match', 'distributed_map'))
@pytest.mark.parametrize('provenance', (False, True))
def test_real_nested_regions_preserve_projection_hash_and_detached_copies(kind, provenance):
    p = region_program(kind)
    projected = p._serialize(include_provenance=provenance)
    encoded = _json(projected)
    rows = list(_nodes(projected['nodes']))
    assert kind in {row['op'] for row in rows}
    assert all(('provenance' in row) is provenance for row in rows)
    detached = detach_compiled_program(p)
    # Detach may record its own provenance; the scientific projection must agree.
    detached_encoded = _json(detached._serialize(include_provenance=provenance))
    assert _json(detached._serialize(include_provenance=False)) == _json(p._serialize(include_provenance=False))
    canonical = _json(p._serialize(include_provenance=False), sorted_keys=True)
    assert p._ir_hash() == hashlib.sha256(canonical.encode()).hexdigest()
    # Returned nested data are independent of both the live graph and later projections.
    rows[-1]['attrs']['injected_by_consumer'] = {'mutable': [1, 2]}
    rows[0]['name'] = 'consumer_mutation'
    assert _json(p._serialize(include_provenance=provenance)) == encoded
    assert _json(detached._serialize(include_provenance=provenance)) == detached_encoded


@pytest.mark.parametrize('kind', ('branch', 'structured_fold', 'structured_iterate', 'capture_result',
                                  'finite_map', 'finite_scan', 'result_match', 'distributed_map'))
def test_nested_equation_changes_canonical_identity(kind):
    assert region_program(kind, 3)._ir_hash() != region_program(kind, 4)._ir_hash()


class _Integer(int):
    def to_data(self): return {'integer_hook': str(int(self))}


class _Float(float):
    def to_data(self): return {'float_hook': float(self).hex()}


class _Text(str):
    def to_data(self): return {'text_hook': str(self)}


class _List(list):
    def to_data(self): return {'list_hook': list(self)}


class _Dict(dict):
    def to_data(self): return {'dict_hook': dict(self)}


@pytest.mark.parametrize('value,expected', (
    (_Integer(7), {'integer_hook': '7'}),
    (_Float(-0.), {'float_hook': '-0x0.0p+0'}),
    (_Text('name'), {'text_hook': 'name'}),
    (_List([1, 2]), {'list_hook': [1, 2]}),
    (_Dict(x=3), {'dict_hook': {'x': 3}})))
def test_builtin_subclasses_keep_their_projection_protocol(value, expected):
    assert _json_ready(value) == expected


def test_reference_resolution_precedes_metadata_hook_and_preserves_copies():
    calls = []
    class Resolved:
        def to_data(self):
            calls.append('to_data')
            return {'payload': (1, -0.)}
    class Original:
        def declaration_references(self): return ('declared_reference',)
        def resolve_references(self, resolver):
            assert callable(resolver)
            calls.append('resolve')
            return Resolved()
        def to_data(self): raise AssertionError('unresolved metadata hook used')
    value = Original()
    first = _json_ready(value)
    first['payload'][0] = 999
    second = _json_ready(value)
    assert second == {'payload': [1, -0.]}
    assert _json(second).endswith(',-0.0]}')
    assert calls == ['resolve', 'to_data', 'resolve', 'to_data']


def test_projection_failure_is_not_hidden_by_fast_paths_or_block_named_metadata():
    class Broken(str):
        def to_data(self): raise ValueError('metadata projection failed')
    for value in (Broken('bad'), {'body_block': [Broken('bad')]}):
        with pytest.raises(ValueError, match='metadata projection failed'):
            _json_ready(value)


def test_nested_child_shell_and_nonregion_body_block_receive_protocol_projection():
    class Provenance:
        def to_data(self): return {'labels': ('a', 'b')}
    def node(op, attrs, ident):
        return SimpleNamespace(op=op, attrs=attrs, id=ident, name=_Text('node'),
            vtype='int64', block=None, state_ref=None, point=None, inputs=(),
            space=None, field_context=None, provenance=Provenance())
    leaf = node('int64_const', {'value': 1, 'body_block': [_Integer(8)]}, 1)
    inner = node('structured_fold', {'body_block': [leaf], 'body': leaf}, 2)
    outer = node('structured_fold', {'body_block': [inner], 'body': inner}, 3)
    row = _ProgramSerialization._serialize_node(outer)
    child = row['attrs']['body_block'][0]['attrs']['body_block'][0]
    assert child['name'] == {'text_hook': 'node'}
    assert child['provenance'] == {'labels': ['a', 'b']}
    assert child['attrs']['body_block'] == [{'integer_hook': '8'}]
    child['attrs']['body_block'][0]['integer_hook'] = 'mutated'
    again = _ProgramSerialization._serialize_node(outer)
    assert again['attrs']['body_block'][0]['attrs']['body_block'][0]['attrs']['body_block'] == [{'integer_hook': '8'}]
