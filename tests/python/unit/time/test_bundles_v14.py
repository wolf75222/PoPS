"""Source contracts only; native remap/rollback acceptance is a separate receiver."""
import pytest

from pops.bundles import BundleSchema, Remap, remap
from pops.graph_functions import graph
from pops.time import Program
from pops.time.variant_types import Failure, Product, Result, type_of
from pops.time._program.detach import detach_compiled_program
from pops.time._program.structured_control import value_signature
from pops.codegen.program_codegen import emit_cpp_program
from typed_program_support import typed_state


def package(p, domain, *, source=None, history=None):
    source = domain if source is None else source
    history = domain if history is None else history
    values = p.map(domain, lambda q, i: q.float64(q.index_value(i)) + 1.)
    volumes = p.map(domain, lambda q, i: q.float64(1.) / q.float64(q.size(domain)))
    parents = p.map(domain, lambda q, i: q.index(source, 0))
    previous = p.map(history, lambda q, i: q.float64(2.))
    return p.product(instance=domain, fields=p.product(q=values), geometry=p.product(volume=volumes),
                     maps=p.product(source=source, parent=parents), histories=p.product(previous=previous))


def schema(value, *, omit_history=False):
    bindings = {('fields', 'q'): 'current', ('geometry', 'volume'): 'current',
                ('maps', 'parent'): 'current', ('maps', 'parent', '[]'): 'source'}
    if not omit_history:
        bindings[('histories', 'previous')] = 'current'
    return BundleSchema(value, domains={'current': ('instance',), 'source': ('maps', 'source')},
                        bindings=bindings)


def fixture():
    p = Program('dependent_bundle_source')
    q = typed_state(p, 'left', state_name='U'); q.n
    domain = p.range_set(2, capacity=8, ownership='replicated')
    raw = package(p, domain)
    contract = schema(raw)
    return p, q, contract, contract.bind(p, raw)


def source(p, q, bundle):
    total = p.reduce(bundle['fields']['q'], p.float64(0.), lambda b, s, x: s+x, order='ascending')
    p.commit(q.next, p.value('bundle_result', q.n * total, at=q.next.point))
    assert p.validate()
    detached = detach_compiled_program(p)
    assert detached._ir_hash() == p._ir_hash()
    return emit_cpp_program(p)


def test_bundle_bindings_are_exhaustive_and_retained_in_real_graph_and_cpp():
    p, q, contract, bundle = fixture()
    with pytest.raises(ValueError, match='every Array'):
        schema(package(p, bundle['instance']), omit_history=True)
    cpp = source(p, q, bundle)
    assert 'ContinuationDomainBinding' in cpp
    assert 'continuation_each' in cpp
    assert 'codec.validate(' in cpp
    assert 'Bundle dependency or remap validation failed' in cpp
    assert value_signature(bundle)['bundle'] == contract.value_type.signature['bundle']


def test_same_family_stale_history_is_checked_natively_not_accepted_by_shape():
    p, q, contract, before = fixture()
    after = p.range_set(p.size(before['instance']), family=before['instance'])
    # This deliberately reaches the native pointer-identity guard. Static family
    # and element types agree; an equal length is insufficient evidence.
    stale = contract.bind(p, package(p, after, source=before['instance'], history=before['instance']))
    cpp = source(p, q, stale)
    assert 'codec.validate(' in cpp
    assert 'ctx.continuation_new_domain(' in cpp


def test_remap_requires_whole_closed_result_and_every_history_policy():
    p, q, contract, before = fixture()
    data = p.product(size=p.int64(4))
    result_type = Result(type_of(before), Failure())

    @graph(inputs=Product(bundle=type_of(before), data=type_of(data)), output=result_type)
    def refine(g, args):
        old = args['bundle']
        new_domain = g.range_set(args['data']['size'], family=old['instance'])
        after = contract.bind(g, package(g, new_domain, source=old['instance']))
        return g.variant(result_type, 'ok', after)

    with pytest.raises(ValueError, match='every retained history'):
        remap(p, Remap(refine, source_instance=('maps', 'source'), histories={}), before, data=data)
    result = remap(p, Remap(refine, source_instance=('maps', 'source'),
                           histories={'previous': 'reconstruct'}), before, data=data)
    cpp = source(p, q, p.require_success(result))
    assert 'new_instance == old_instance || source_instance != old_instance' in cpp
    assert 'codec.validate(' in cpp


def test_bundle_contract_cannot_hide_an_unbound_index_in_a_variant():
    p, q, contract, before = fixture()
    from pops.time.variant_types import Variant
    d = before['instance']
    choice = p.variant(Variant(index=type_of(p.index(d, 0)), empty=Product()), 'index', p.index(d, 0))
    raw = p.product(instance=d, fields=p.product(q=before['fields']['q']),
                    geometry=before['geometry'], maps=p.product(source=d, selected=choice),
                    histories=before['histories'])
    with pytest.raises(ValueError, match='every Array'):
        BundleSchema(raw, domains={'current': ('instance',), 'source': ('maps', 'source')},
                     bindings={('fields', 'q'): 'current', ('geometry', 'volume'): 'current',
                               ('histories', 'previous'): 'current'})


def test_fixed_communicator_repartition_is_a_zero_block_closed_program():
    from examples.api_v02.bundle_repartition_v14 import build_bundle_repartition_v14
    spec=build_bundle_repartition_v14(ranks=3)
    program=spec.case._time
    assert spec.case.validate() and not program._block_indices() and not spec.initial_values
    assert detach_compiled_program(program)._ir_hash()==program._ir_hash()
    cpp=emit_cpp_program(program)
    assert 'requests_array<' in cpp and 'requests_from_array<' in cpp
    assert 'continuation_publish(' in cpp and 'new_instance == old_instance' in cpp


def test_multisupport_nested_trials_preserve_closed_failure_and_transform_graphs():
    from examples.api_v02.multisupport_children_v14 import build_multisupport_children_v14
    from pops.codegen.program_lowerability import all_ops
    spec=build_multisupport_children_v14()
    program=spec.case._time
    assert spec.case.validate() and not program._block_indices() and not spec.initial_values
    operations=list(all_ops(program))
    names=[value.op for value in operations]
    assert names.count('continuation_consume_increment')>=4
    assert names.count('continuation_transform')>=2
    assert names.count('continuation_attest')>=6
    assert detach_compiled_program(program)._ir_hash()==program._ir_hash()
    cpp=emit_cpp_program(program)
    assert 'consume_increment(' in cpp and 'supersede(' in cpp
