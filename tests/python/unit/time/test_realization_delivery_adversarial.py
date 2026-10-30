"""Read-only counterexamples against v10 delivery staging, using actual PoPS Program IR."""
import pytest

from pops.time import Program
import pops.math as pm
from pops.time.expressions import component_names
from typed_program_support import typed_state
from pops.codegen.typed_delivery import DeliveryReference, delivered_frontiers


def fixture():
    p = Program('adversarial_delivery')
    q = typed_state(p, 'material', state_name='U')
    q.n
    outputs = []
    def delivery(b, state):
        result = b.value('delivery', state)
        outputs.append(result)
        return result
    def check(value, **kwargs):
        return delivered_frontiers(value, lambda node: any(node is x for x in outputs), **kwargs)
    return p, q, delivery, check


def test_legacy_failure_recovery_must_not_borrow_the_success_delivery():
    p, q, delivery, check = fixture()
    result = p.branch(p.sum(q.n) > 0.,
                      lambda b: b.success(delivery(b, q.n)),
                      lambda b: b.failure(q.n, 7))
    candidate = p.match(result, ok=lambda b, value: value,
                       err=lambda b, failure: b.value('recovered', b.float64(failure['code'])*q.n))
    frontiers = check(candidate)
    # Failure recovers successfully to 7*q.n, with no owned delivery on that path.
    assert any(not frontier for frontier in frontiers), frontiers


def test_cell_where_predicate_is_not_a_numerical_delivery():
    p, q, delivery, check = fixture()
    owned = delivery(p, q.n)
    candidate = p.where(p.cell_gt(owned, 0.), q.n, q.n)
    frontiers = check(candidate)
    # Every cell receives exactly q.n independently of the owned predicate value.
    assert any(not frontier for frontier in frontiers), frontiers


def test_expression_where_predicate_is_not_a_numerical_delivery():
    p, q, delivery, check = fixture()
    owned = delivery(p, q.n)
    expressions = tuple(pm.where(owned[0] > 0, lambda i=i: q.n[i], lambda i=i: q.n[i])
                        for i in range(len(component_names(q.n))))
    candidate = p.value('selected', expressions)
    assert check(candidate) == (frozenset(),)


@pytest.mark.parametrize('encoded', [False, True])
def test_cell_selection_refuses_unrepresented_spatial_delivery_partition(encoded):
    p, q, delivery, check = fixture()
    owned = delivery(p, q.n)
    if encoded:
        candidate = p.value('selected', tuple(
            pm.where(q.n[0] > 0, lambda i=i: owned[i], lambda i=i: q.n[i])
            for i in range(len(component_names(q.n)))))
    else:
        candidate = p.where(p.cell_gt(q.n, 0.), owned, q.n)
    with pytest.raises(ValueError, match='different delivery frontiers'):
        check(candidate)


def test_cell_selection_of_same_frontier_is_supported():
    p, q, delivery, check = fixture()
    owned = delivery(p, q.n)
    candidate = p.where(p.cell_gt(q.n, 0.), owned, p.value('other_formula', 2*owned))
    assert len(check(candidate)[0]) == 1


def test_literal_legacy_success_has_no_error_success_frontier():
    p, q, delivery, check = fixture()
    result = p.success(delivery(p, q.n))
    candidate = p.match(result, ok=lambda b, value: value,
                       err=lambda b, failure: b.value('recovered', b.float64(failure['code'])*q.n))
    assert len(check(candidate)) == 1 and len(check(candidate)[0]) == 1


def test_captured_legacy_failure_recovery_keeps_the_empty_frontier():
    p, q, delivery, check = fixture()
    condition = p.sum(q.n) > 0.
    result = p.capture(lambda b: b.branch(condition,
        lambda z: z.success(delivery(z, q.n)), lambda z: z.failure(q.n, 7)))
    candidate = p.match(result, ok=lambda b, value: value,
                       err=lambda b, failure: b.value('recovered', b.float64(failure['code'])*q.n))
    assert any(not frontier for frontier in check(candidate))


def test_nested_fold_retains_every_dynamic_delivery_identity():
    p, q, delivery, check = fixture()
    def outer(b, state, index):
        return b.fold(state, 2, lambda z, carried, inner_index:
                      z.value('accumulated', carried+delivery(z, q.n)))
    frontiers = check(p.fold(q.n, 2, outer))
    assert len(frontiers) == 1 and len(frontiers[0]) == 4
    assert len({reference.value_id for reference in frontiers[0]}) == 1
    assert len({reference.invocation for reference in frontiers[0]}) == 4


@pytest.mark.parametrize('budgets', [dict(max_alternatives=4096), dict(max_iterations=256)])
def test_analysis_has_no_budget_to_pass_or_disable(budgets):
    # CF7 and B4: the analysis is symbolic, so no cap can be supplied either.
    p, q, delivery, check = fixture()
    with pytest.raises(TypeError):
        check(delivery(p, q.n), **budgets)


def test_ref_identity_cannot_be_compared_to_old_integer_retained_ids():
    p, q, delivery, check = fixture()
    emitted = delivery(p, q.n)
    retained = set().union(*check(emitted))
    assert retained == {DeliveryReference(emitted.id, ())}
    assert emitted.id not in retained
    assert emitted.id in {reference.value_id for reference in retained}


def test_correlated_same_bool_branches_retain_single_actual_delivery():
    p, q, delivery, check = fixture()
    condition = p.requested_dt() > 0.
    left = p.branch(condition, lambda b: delivery(b, q.n), lambda b: q.n)
    right = p.branch(condition, lambda b: q.n, lambda b: delivery(b, q.n))
    result = p.value('composed', left+right)
    assert all(len(frontier) == 1 for frontier in check(result))


def test_independent_deliveries_are_counted_without_enumerating_alternatives():
    from pops._ir.refusals import CoverageNotExact
    from pops.codegen.typed_delivery import require_single_delivery
    p, q, delivery, check = fixture()
    owned = []
    def composed(count):
        def tracked(b):
            owned.append(delivery(b, q.n))
            return owned[-1]
        branches = [p.branch(p.requested_dt() > float(i), tracked, tracked) for i in range(count)]
        return p.value('composed', sum(branches[1:], branches[0]))
    rows = check(composed(4))
    assert len(rows) == 16 and all(len(frontier) == 4 for frontier in rows)
    # Forty independent branches have 2**40 success paths; the count saturates at
    # two on the shared decision diagram, and the refusal names the term.
    with pytest.raises(CoverageNotExact, match='term composed_use is delivered at least twice'):
        require_single_delivery(composed(40), lambda node: any(node is x for x in owned),
                                term='composed_use', missing='missing', duplicate='duplicate')


def test_delivered_frontiers_lists_one_row_per_success_path_for_the_provenance_validators():
    # MG7: the provenance validators (current dependencies, ledger provenance) compare these
    # rows, so they follow the 2**k success paths of k independent choices, without a budget;
    # N7 lot 3 (ROM-88) moves them onto the decision diagram. require_single_delivery does not.
    p, q, delivery, check = fixture()
    def tracked(b):
        return delivery(b, q.n)
    branches = [p.branch(p.requested_dt() > float(i), tracked, tracked) for i in range(10)]
    rows = check(p.value('composed', sum(branches[1:], branches[0])), with_choices=True)
    assert len(rows) == len({choices for choices, _ in rows}) == 2**10
    assert all(len(frontier) == 10 for _, frontier in rows)


def test_unreachable_variant_terminal_cannot_become_an_empty_success_by_pruning():
    from pops.codegen.program_lowerability import all_ops
    from pops.time.variant_types import Variant, type_of
    p, q, delivery, check = fixture()
    variant = p.variant(Variant(only=type_of(q.n)), 'only', q.n)
    p.match(variant, only=lambda b, payload: payload)
    unreachable = [v for v in all_ops(p) if v.op == 'variant_unreachable']
    assert len(unreachable) == 1
    assert check(unreachable[0]) == ()


@pytest.mark.parametrize('typed', [False, True])
def test_use_free_dynamic_numerical_loop_is_not_unrolled_for_delivery(typed):
    from pops.time.variant_types import Result, type_of, Int64
    p, q, delivery, check = fixture()
    result_type = Result(type_of(q.n), Int64)
    def body(b, state):
        return b.branch(b.sum(state) > 0.,
            lambda c: c.variant(result_type, 'ok', state) if typed else c.success(state),
            lambda c: c.variant(result_type, 'err', c.int64(7)) if typed else c.failure(state, 7))
    loop = p.iterate(q.n, body, lambda b, state: b.sum(state) < 1.,
                     max_steps=p.int64(p.requested_dt()) + 2)
    state = p.require_success(loop)['state']
    # A numerical solve with no Use can have a successful empty frontier; this
    # says nothing about convergence and grants no delivery to its result.
    assert check(state) == (frozenset(),)
    candidate = p.value('combined', state + delivery(p, q.n))
    assert list(map(len, check(candidate))) == [1]


@pytest.mark.parametrize('typed', [False, True])
def test_use_free_positive_fold_preserves_first_terminal_alias(typed):
    from pops.time.variant_types import Result, type_of, Int64
    p, q, delivery, check = fixture()
    result_type = Result(type_of(q.n), Int64)
    def body(b, state, index):
        error = (b.variant(result_type, 'err', b.int64(7)) if typed else b.failure(state, 7))
        return b.value('terminal_alias', b.require_success(error))
    folded = p.fold(q.n, 1, body, max_steps=p.int64(p.requested_dt()) + 2)
    assert check(folded) == ()


@pytest.mark.parametrize('terminal_condition', [False, True])
def test_use_free_iterate_cannot_turn_terminal_stop_into_success(terminal_condition):
    p, q, delivery, check = fixture()
    def stop(b, state):
        if terminal_condition:
            return b.sum(b.value('terminal_alias', b.require_success(b.failure(state, 7)))) > 0.
        return b.boolean(False)
    loop = p.iterate(q.n, lambda b, state: b.success(state), stop,
                     max_steps=p.int64(p.requested_dt()) + 2)
    assert check(p.require_success(loop)['state']) == ()


def scatter_case(combine, *, initial_owned=False, count=None, read=None):
    """An owned Use realized in each source element of a distributed scatter_reduce.

    ``read(b, d, domain, target, values, indices, result)`` reads what the scatter_reduce
    receives (``values``) and returns (``result``)."""
    from pops.time.variant_types import Float64, Int64, Result
    from pops.time._program.distributed_arrays import Distributed
    p = Program('scatter_delivery')
    q = typed_state(p, 'material', state_name='U')
    q.n
    d = Distributed(p)
    domain, target = d.domain((0, 1, 1, 4), capacity=4), d.domain((0, 1, 1, 2), capacity=2)
    indices = d.requests(domain, target, lambda b, i: d.index(target, d.gid(i) % 2))
    owned = []
    def element(b, i):
        owned.append(b.float64(d.gid(i))*2.)
        return b.variant(Result(Float64, Int64), 'ok', owned[-1])
    def scatter(b):
        values = b.require_success(d.local_map(domain, element, byte_capacity=65536))
        initial = b.float64(0.)
        if initial_owned:
            owned.append(b.float64(1.)*3.)
            initial = owned[-1]
        result = d.scatter_reduce(values, indices, target, initial, lambda c, s, x: combine(c, s, x, owned),
                                  order='global_gid_ascending', byte_capacity=65536)
        return result if read is None else read(b, d, domain, target, values, indices, result)
    result = scatter(p) if count is None else p.fold(scatter(p), count, lambda b, carried, k: scatter(b))
    return result, lambda node: any(node is x for x in owned)


def test_scatter_reduce_delivers_each_source_element_once_to_its_target():
    from pops.codegen.typed_delivery import EACH, require_single_delivery
    result, matches = scatter_case(lambda c, s, x, owned: s+x)
    (frontier,) = delivered_frontiers(result, matches)
    (reference,) = frontier
    assert reference.invocation[-1][1:] == (EACH, 'map')
    require_single_delivery(result, matches, term='face_term', missing='missing', duplicate='duplicate')


def _owned_in_combine(c, owned):
    owned.append(c.float64(1.)*5.)
    return owned[-1]


@pytest.mark.parametrize('combine,message', [
    (lambda c, s, x, owned: s, 'must deliver its element'),
    (lambda c, s, x, owned: x, 'must keep its accumulator'),
    (lambda c, s, x, owned: s+x+_owned_in_combine(c, owned), 'delivers a Use once per combination'),
    (lambda c, s, x, owned: c.branch(x > 1., lambda z: s+x, lambda z: s), 'the same parts on every path'),
])
def test_scatter_reduce_combine_keeps_its_accumulator_and_its_element(combine, message):
    from pops._ir.refusals import DeliveryNotStructural
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = scatter_case(combine)
    with pytest.raises(DeliveryNotStructural, match=message + '.*; term face_term$'):
        require_single_delivery(result, matches, term='face_term', missing='missing', duplicate='duplicate')


def test_scatter_reduce_adds_its_initial_delivery_to_its_source_deliveries():
    from pops._ir.refusals import CoverageNotExact
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = scatter_case(lambda c, s, x, owned: s+x, initial_owned=True)
    assert list(map(len, delivered_frontiers(result, matches))) == [2]
    with pytest.raises(CoverageNotExact, match='term face_term is delivered at least twice'):
        require_single_delivery(result, matches, term='face_term', missing='missing', duplicate='duplicate')


def _local(b, d, domain, element):
    from pops.time.variant_types import Float64, Int64, Result
    return b.require_success(d.local_map(domain, lambda c, i: c.variant(Result(Float64, Int64), 'ok', element(c, i)),
                                         byte_capacity=65536))


SCATTER_READS = {
    'its result at each target': (lambda b, d, D, T, values, indices, result: _local(
        b, d, T, lambda c, j: d.at_local(result, j)+1.), None),
    'its result at target 0 everywhere': (lambda b, d, D, T, values, indices, result: _local(
        b, d, T, lambda c, j: d.at_local(result, d.index(T, 0))), 'a gather'),
    'its result at the target of each source': (lambda b, d, D, T, values, indices, result: _local(
        b, d, D, lambda c, i: d.at_local(result, d.index(T, d.gid(i) % 2))), 'a gather'),
    'its result imported by its sources': (lambda b, d, D, T, values, indices, result: d.import_values(
        result, indices, byte_capacity=65536), 'a gather'),
    'its sources imported by its targets': (lambda b, d, D, T, values, indices, result: d.import_values(
        values, d.requests(T, D, lambda c, j: d.index(D, d.gid(j))), byte_capacity=65536), 'a gather'),
}


@pytest.mark.parametrize('read', sorted(SCATTER_READS))
def test_scatter_reduce_keeps_the_element_of_what_it_receives_and_returns(read):
    from pops._ir.refusals import DeliveryNotStructural
    from pops.codegen.typed_delivery import EACH, require_single_delivery
    reader, refusal = SCATTER_READS[read]
    result, matches = scatter_case(lambda c, s, x, owned: s+x, read=reader)
    if refusal is None:
        (reference,) = require_single_delivery(result, matches, term='face_term', missing='missing',
                                               duplicate='duplicate')
        assert reference.invocation[-1][1:] == (EACH, 'map')
        return
    with pytest.raises(DeliveryNotStructural, match=r'\(%s\).*; term face_term$' % refusal):
        require_single_delivery(result, matches, term='face_term', missing='missing', duplicate='duplicate')


def test_scatter_reduce_refuses_an_initial_value_that_carries_an_element_wise_delivery():
    from pops._ir.refusals import DeliveryNotStructural
    from pops.codegen.typed_delivery import require_single_delivery
    from pops.time.variant_types import Int64, Result, type_of
    from pops.time._program.distributed_arrays import Distributed
    p = Program('scatter_initial_delivery')
    d = Distributed(p)
    domain, target = d.domain((0, 1, 1, 4), capacity=4), d.domain((0, 1, 1, 2), capacity=2)
    e = p.range_set(3, capacity=4, ownership='replicated')
    indices = d.requests(domain, target, lambda b, i: d.index(target, d.gid(i) % 2))
    owned = []
    def own(b, j):
        owned.append(b.float64(b.index_value(j))*2.)
        return owned[-1]
    def element(b, i):
        inner = b.map(e, lambda c, j: c.float64(c.index_value(j))*0.)
        return b.variant(Result(type_of(inner), Int64), 'ok', inner)
    values = p.require_success(d.local_map(domain, element, byte_capacity=65536))
    # Every target starts with the whole owned array, so each of its elements reaches each target.
    result = d.scatter_reduce(values, indices, target, p.map(e, own),
                              lambda c, s, x: c.map(e, lambda g, j: g.at(s, j)+g.at(x, j)),
                              order='global_gid_ascending', byte_capacity=65536)
    with pytest.raises(DeliveryNotStructural, match=r'initial value.*\(a broadcast\); term face_term$'):
        require_single_delivery(result, lambda node: any(node is x for x in owned), term='face_term',
                                missing='missing', duplicate='duplicate')


def test_scatter_reduce_in_a_loop_of_ten_thousand_iterations_delivers_its_last_iteration():
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = scatter_case(lambda c, s, x, owned: s+x, count=10**4)
    (frontier,) = delivered_frontiers(result, matches)
    (reference,) = frontier
    assert reference.invocation[0] == (result.id, 10**4)
    require_single_delivery(result, matches, term='face_term', missing='missing', duplicate='duplicate')


def map_case(consume, *, capture=False):
    """An owned Use realized in each element of a map over d, read by ``consume``."""
    p = Program('element_delivery')
    if capture:
        p.capture_source_locations()
    d, e = p.range_set(4, capacity=8, ownership='replicated'), p.range_set(3, capacity=8, ownership='replicated')
    owned = []
    def own(b, value):
        owned.append(value)
        return value
    array = p.map(d, lambda b, i: own(b, b.float64(b.index_value(i))*2.))
    nested = p.map(d, lambda b, i: b.map(e, lambda c, j: own(c, c.float64(c.index_value(j))*3.)))
    return consume(p, d, e, array, nested), lambda node: any(node is x for x in owned)


ELEMENT_READS = {
    'the array': lambda p, d, e, a, n: a,
    'a pointwise copy': lambda p, d, e, a, n: p.map(d, lambda b, i: b.at(a, i)+1.),
    'its own element twice': lambda p, d, e, a, n: p.map(d, lambda b, i: b.at(a, i)+b.at(a, i)),
    'a copy over its own domain': lambda p, d, e, a, n: p.map(p.domain(a), lambda b, i: b.at(a, i)),
    'its element in a nested map': lambda p, d, e, a, n: p.map(d, lambda b, i: b.map(e, lambda c, j: c.at(a, i))),
    'a nested pointwise copy': lambda p, d, e, a, n: p.map(d, lambda b, i: b.map(e, lambda c, j: c.at(c.at(n, i), j))),
    'its inner arrays': lambda p, d, e, a, n: p.map(d, lambda b, i: b.at(n, i)),
}
GATHERS = {
    'element 0 at every element': lambda p, d, e, a, n: p.map(d, lambda b, i: b.at(a, b.index(d, 0))),
    'element i and element 0': lambda p, d, e, a, n: p.map(d, lambda b, i: b.at(a, i)+b.at(a, b.index(d, 0))),
    'element 0 alone': lambda p, d, e, a, n: p.at(a, p.index(d, 0)),
    'a fold over the elements': lambda p, d, e, a, n: p.fold(p.float64(0.), 4,
                                                             lambda b, c, k: c+b.at(a, b.index(d, k))),
    'one element of each inner array': lambda p, d, e, a, n: p.map(d, lambda b, i: b.at(b.at(n, i), b.index(e, 0))),
    'a smaller set of its family': lambda p, d, e, a, n: p.map(p.range_set(2, family=d), lambda b, i: b.at(a, i)),
}
BROADCASTS = {
    'the whole array in each element': lambda p, d, e, a, n: p.map(e, lambda b, j: b.value('copy', a)),
    'an inner array in each element of a nested map': lambda p, d, e, a, n: p.map(
        d, lambda b, i: (lambda inner: b.map(e, lambda c, j: c.value('copy', inner)))(b.at(n, i))),
}


@pytest.mark.parametrize('read', sorted(ELEMENT_READS))
def test_an_element_wise_delivery_leaves_its_map_as_the_array_or_at_its_own_element(read):
    from pops.codegen.typed_delivery import EACH, require_single_delivery
    result, matches = map_case(ELEMENT_READS[read])
    (frontier,) = delivered_frontiers(result, matches)
    (reference,) = frontier
    assert all(step[1:] == (EACH, 'map') for step in reference.invocation)
    require_single_delivery(result, matches, term='element_term', missing='missing', duplicate='duplicate')


@pytest.mark.parametrize('read,kind', [(name, 'a gather') for name in sorted(GATHERS)] +
                                      [(name, 'a broadcast') for name in sorted(BROADCASTS)])
def test_an_element_wise_delivery_read_at_another_element_is_refused_naming_the_term(read, kind):
    from pops._ir.refusals import DeliveryNotStructural
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = map_case({**GATHERS, **BROADCASTS}[read])
    with pytest.raises(DeliveryNotStructural, match=r'\(%s\).*; term element_term$' % kind):
        require_single_delivery(result, matches, term='element_term', missing='missing', duplicate='duplicate')
    with pytest.raises(DeliveryNotStructural, match=kind):
        delivered_frontiers(result, matches)


def test_a_gather_refusal_names_the_authoring_line_when_the_program_records_it():
    from pops._ir.refusals import DeliveryNotStructural
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = map_case(GATHERS['element 0 at every element'], capture=True)
    with pytest.raises(DeliveryNotStructural) as refused:
        require_single_delivery(result, matches, term='element_term', missing='missing', duplicate='duplicate')
    assert refused.value.region.startswith('finite_at#')
    assert ' at ' in refused.value.region and 'test_realization_delivery_adversarial.py:' in refused.value.region


def test_two_arrays_of_one_term_read_at_the_same_element_deliver_it_twice():
    from pops._ir.refusals import CoverageNotExact
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = map_case(lambda p, d, e, a, n: p.map(
        d, lambda b, i: b.map(e, lambda c, j: c.at(a, i)+c.at(c.at(n, i), j))))
    with pytest.raises(CoverageNotExact, match='duplicate; term element_term is delivered at least twice'):
        require_single_delivery(result, matches, term='element_term', missing='missing', duplicate='duplicate')
