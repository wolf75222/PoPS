"""Read-only counterexamples against v10 delivery staging, using actual PoPS Program IR."""
import pytest

from pops.time import Program
import pops.math as pm
from pops.time.expressions import component_names
from typed_program_support import typed_state
from pops.codegen.realization_delivery import DeliveryReference, delivered_frontiers


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


@pytest.mark.parametrize('budgets', [dict(max_alternatives=0), dict(max_alternatives=float('nan')),
                                    dict(max_iterations=-1), dict(max_iterations=float('inf'))])
def test_analysis_budgets_cannot_be_disabled_with_invalid_numbers(budgets):
    p, q, delivery, check = fixture()
    with pytest.raises(ValueError, match='budget must be'):
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


def test_alternative_budget_refuses_exponential_independent_deliveries():
    p, q, delivery, check = fixture()
    branches = [p.branch(p.requested_dt() > float(i), lambda b: delivery(b, q.n),
                        lambda b: delivery(b, q.n)) for i in range(4)]
    result = p.value('composed', sum(branches[1:], branches[0]))
    with pytest.raises(ValueError, match='alternative budget'):
        check(result, max_alternatives=3)


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
    assert check(state, max_iterations=0) == (frozenset(),)
    candidate = p.value('combined', state + delivery(p, q.n))
    assert list(map(len, check(candidate, max_iterations=0))) == [1]


@pytest.mark.parametrize('typed', [False, True])
def test_use_free_positive_fold_preserves_first_terminal_alias(typed):
    from pops.time.variant_types import Result, type_of, Int64
    p, q, delivery, check = fixture()
    result_type = Result(type_of(q.n), Int64)
    def body(b, state, index):
        error = (b.variant(result_type, 'err', b.int64(7)) if typed else b.failure(state, 7))
        return b.value('terminal_alias', b.require_success(error))
    folded = p.fold(q.n, 1, body, max_steps=p.int64(p.requested_dt()) + 2)
    assert check(folded, max_iterations=0) == ()


@pytest.mark.parametrize('terminal_condition', [False, True])
def test_use_free_iterate_cannot_turn_terminal_stop_into_success(terminal_condition):
    p, q, delivery, check = fixture()
    def stop(b, state):
        if terminal_condition:
            return b.sum(b.value('terminal_alias', b.require_success(b.failure(state, 7)))) > 0.
        return b.boolean(False)
    loop = p.iterate(q.n, lambda b, state: b.success(state), stop,
                     max_steps=p.int64(p.requested_dt()) + 2)
    assert check(p.require_success(loop)['state'], max_iterations=0) == ()
