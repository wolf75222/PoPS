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


def scatter_case(combine, *, initial_owned=False, count=None):
    """An owned Use realized in each source element of a distributed scatter_reduce."""
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
        return d.scatter_reduce(values, indices, target, initial, lambda c, s, x: combine(c, s, x, owned),
                                order='global_gid_ascending', byte_capacity=65536)
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
    (lambda c, s, x, owned: s+x+_owned_in_combine(c, owned), 'delivers a Use of its own'),
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


def test_scatter_reduce_in_a_loop_of_ten_thousand_iterations_delivers_its_last_iteration():
    from pops.codegen.typed_delivery import require_single_delivery
    result, matches = scatter_case(lambda c, s, x, owned: s+x, count=10**4)
    (frontier,) = delivered_frontiers(result, matches)
    (reference,) = frontier
    assert reference.invocation[0] == (result.id, 10**4)
    require_single_delivery(result, matches, term='face_term', missing='missing', duplicate='duplicate')
