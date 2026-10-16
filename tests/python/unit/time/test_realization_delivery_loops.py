"""Counterexamples use real Program regions; this file does not execute numerics."""
import pytest

from pops.time import Program
from typed_program_support import typed_state
from pops.codegen.realization_delivery import delivered_frontiers


def fixture():
    p = Program('delivery_loop')
    q = typed_state(p, 'material', state_name='U')
    outputs = []
    def delivery(builder, state):
        value = builder.value('delivery', state)
        outputs.append(value)
        return value
    def frontiers(value, **kwargs):
        return delivered_frontiers(value, lambda node: any(node is item for item in outputs), **kwargs)
    return p, q, delivery, frontiers


@pytest.mark.parametrize('count', [0, 1, 2])
def test_fold_last_delivery_and_zero_iterations(count):
    p, q, delivery, frontiers = fixture()
    result = p.fold(q.n, count, lambda b, state, i: delivery(b, state))
    values = frontiers(result)
    assert len(values) == 1
    assert len(values[0]) == (0 if count == 0 else 1)
    if count:
        assert next(iter(values[0])).invocation[-1][1] == count


def test_native_count_includes_zero_not_just_the_recorded_body():
    p, q, delivery, frontiers = fixture()
    result = p.fold(q.n, p.int64(p.requested_dt()), lambda b, state, i: delivery(b, state), max_steps=2)
    assert sorted(map(len, frontiers(result))) == [0, 1, 1]


def test_initial_and_later_choices_are_alternatives_not_a_sum():
    p, q, delivery, frontiers = fixture()
    initial = delivery(p, q.n)
    result = p.fold(initial, p.int64(p.requested_dt()), lambda b, state, i: delivery(b, state), max_steps=2)
    values = frontiers(result)
    assert len(values) == 3 and all(len(value) == 1 for value in values)
    assert len(set().union(*values)) == 3


def test_repeated_body_identifier_cannot_hide_accumulated_deliveries():
    p, q, delivery, frontiers = fixture()
    def body(b, state, index):
        return b.value('accumulation', state+delivery(b, q.n))
    result = p.fold(q.n, 2, body)
    values = frontiers(result)
    assert len(values) == 1 and len(values[0]) == 2
    references = tuple(values[0])
    assert references[0].value_id == references[1].value_id
    assert references[0].invocation != references[1].invocation


def test_captured_outer_delivery_is_not_reissued_by_each_iteration():
    p, q, delivery, frontiers = fixture()
    external = delivery(p, q.n)
    result = p.fold(q.n, 2, lambda b, state, i: b.value('captured', external))
    values = frontiers(result)
    assert len(values) == 1 and len(values[0]) == 1
    assert next(iter(values[0])).invocation == ()


def test_loop_product_projection_ignores_other_carried_members():
    p, q, delivery, frontiers = fixture()
    initial = p.product(kept=q.n, unused=delivery(p, q.n))
    result = p.fold(initial, 2, lambda b, state, i:
        b.product(kept=state['kept'], unused=delivery(b, state['unused'])))
    assert frontiers(result['kept']) == (frozenset(),)
    assert len(frontiers(result['unused'])[0]) == 1


@pytest.mark.parametrize('stop,expected', [(True, [0]), (False, [])])
def test_iterate_never_claims_executed_body_on_initial_stop_or_exhaustion(stop, expected):
    p, q, delivery, frontiers = fixture()
    result = p.require_success(p.iterate(q.n, lambda b, state: delivery(b, state),
                                         lambda b, state: b.boolean(stop), max_steps=2))
    assert sorted(map(len, frontiers(result['state']))) == expected


def test_iterate_dynamic_stop_checks_all_possible_successful_iterations():
    p, q, delivery, frontiers = fixture()
    result = p.require_success(p.iterate(q.n, lambda b, state: delivery(b, state),
                                         lambda b, state: b.sum(state)>0, max_steps=2))
    assert sorted(map(len, frontiers(result['state']))) == [0, 1, 1]


def test_analysis_capacity_fails_closed():
    p, q, delivery, frontiers = fixture()
    result = p.fold(q.n, 10, lambda b, state, i: delivery(b, state))
    with pytest.raises(ValueError, match='analysis budget'):
        frontiers(result, max_iterations=2)
