"""Exact local affine budgets distinguish partitions, stage values and unsupported weights."""
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops.codegen.program_partition_stability import (
    emit_partition_stability, partition_stability_groups,
)


def _state(identifier, point=0):
    return SimpleNamespace(id=identifier, op="state", vtype="state", inputs=(), attrs={},
                           block="inventory", point=point)


def _rate(identifier, op, state, **attrs):
    return SimpleNamespace(id=identifier, op=op, vtype="rhs", inputs=(state,), attrs=attrs,
                           block=state.block, point=state.point)


def _combine(identifier, *terms):
    return SimpleNamespace(id=identifier, op="linear_combine", vtype="state",
                           inputs=tuple(node for node, _ in terms),
                           attrs={"coeffs": tuple(weight for _, weight in terms)},
                           block="inventory", point=1)


def _pair(state, offset=0):
    return _rate(10 + offset, "rhs", state), _rate(11 + offset, "diffusive_rhs", state)


def test_nested_rate_sum_keeps_each_exact_fraction_and_one_state_budget():
    state = _state(0)
    transport, diffusion = _pair(state)
    rate_sum = _combine(20, (transport, {0: Fraction(1, 2)}), (diffusion, {0: Fraction(3, 4)}))
    assert partition_stability_groups(rate_sum) == ()
    update = _combine(21, (state, {0: 1}), (rate_sum, {1: Fraction(2, 3)}))
    assert partition_stability_groups(update) == (
        (state, Fraction(1), ((transport, Fraction(1, 3)), (diffusion, Fraction(1, 2)))),)
    lines = []
    emit_partition_stability(update, {("partition_frequency", 10): "transport_budget",
                                      ("partition_frequency", 11): "tensor_budget"},
                             lines, block_index=0)
    assert any("transport_budget" in line and "tensor_budget" in line for line in lines)
    assert any("combined_transport_diffusion_stability" in line for line in lines)


def test_convex_stage_budgets_do_not_accumulate_predictor_ancestry():
    initial = _state(0)
    first_transport, first_diffusion = _pair(initial)
    predictor = _combine(20, (initial, {0: 1}), (first_transport, {1: 1}), (first_diffusion, {1: 1}))
    last_transport, last_diffusion = _pair(predictor, 2)
    accepted = _combine(21, (initial, {0: Fraction(1, 2)}),
                        (predictor, {0: Fraction(1, 2)}),
                        (last_transport, {1: Fraction(1, 2)}),
                        (last_diffusion, {1: Fraction(1, 2)}))
    assert partition_stability_groups(accepted) == (
        (predictor, Fraction(1, 2),
         ((last_transport, Fraction(1, 2)), (last_diffusion, Fraction(1, 2)))),)
    assert partition_stability_groups(predictor)[0][1] == 1


def test_same_timestamp_does_not_merge_independent_state_budgets():
    first, second = _state(0), _state(1)
    a, b = _pair(first)
    c, d = _pair(second, 2)
    accepted = _combine(20, (first, {0: Fraction(1, 3)}), (second, {0: Fraction(2, 3)}),
                        (a, {1: Fraction(1, 6)}), (b, {1: Fraction(1, 6)}),
                        (c, {1: Fraction(1, 2)}), (d, {1: Fraction(1, 2)}))
    groups = partition_stability_groups(accepted)
    assert [(state.id, alpha) for state, alpha, _ in groups] == [(0, Fraction(1, 3)), (1, Fraction(2, 3))]
    assert [tuple(beta for _, beta in rates) for _, _, rates in groups] == [
        (Fraction(1, 6), Fraction(1, 6)), (Fraction(1, 2), Fraction(1, 2))]


@pytest.mark.parametrize("state_weight,rate_weight", [({0: -1}, {1: 1}), ({0: 2}, {1: 1}),
                                                   ({0: 1}, {1: -1}), ({0: 1}, {2: 1})])
def test_unsupported_affine_budgets_fail_with_a_certificate_diagnostic(state_weight, rate_weight):
    state = _state(0)
    transport, diffusion = _pair(state)
    update = _combine(20, (state, state_weight), (transport, {1: 1}), (diffusion, rate_weight))
    with pytest.raises(ValueError, match="convex affine stability certificate"):
        partition_stability_groups(update)


def test_missing_stage_budget_distinct_points_and_scoped_frequency_are_refused():
    state, other = _state(0), _state(1)
    transport, diffusion = _pair(state)
    update = _combine(20, (other, {0: 1}), (transport, {1: 1}), (diffusion, {1: 1}))
    with pytest.raises(ValueError, match="convex affine stability certificate"):
        partition_stability_groups(update)
    update.inputs = (state, transport, diffusion)
    diffusion.point = 1
    with pytest.raises(ValueError, match="distinct evaluation points"):
        partition_stability_groups(update)
    diffusion.point = 0
    diffusion.attrs["schedule"] = object()
    with pytest.raises(ValueError, match="scoped carrier"):
        partition_stability_groups(update)
    diffusion.attrs.clear()
    with pytest.raises(ValueError, match="unavailable in this region"):
        emit_partition_stability(update, {}, [], block_index=0)


def test_implicit_residual_ancestry_is_not_an_explicit_partition():
    state = _state(0)
    transport, diffusion = _pair(state)
    predictor = _combine(20, (state, {0: 1}), (transport, {1: 1}))
    assert partition_stability_groups(predictor) == ()
    assert partition_stability_groups(predictor, include_transport=True) == (
        (state, Fraction(1), ((transport, Fraction(1)),)),)
    solved = SimpleNamespace(id=21, op="solve_spatial_nonlinear", vtype="state",
                             inputs=(predictor,), attrs={"residual_block": (diffusion,)},
                             block=state.block, point=1)
    assert partition_stability_groups(_combine(22, (solved, {0: 1})), include_transport=True) == ()


def test_guard_evidence_and_frequency_tokens_do_not_escape_a_child_region():
    state = _state(0)
    transport, diffusion = _pair(state)
    update = _combine(20, (state, {0: 1}), (transport, {1: 1}), (diffusion, {1: 1}))
    key = ("partition_stability_checked",)
    outer = {key: frozenset({99})}
    inner = dict(outer)
    inner[("partition_frequency", transport.id)] = "transport_local"
    inner[("partition_frequency", diffusion.id)] = "diffusion_local"
    emit_partition_stability(update, inner, [], block_index=0)
    assert inner[key] == frozenset({99, diffusion.id})
    assert outer == {key: frozenset({99})}
    with pytest.raises(ValueError, match="unavailable in this region"):
        emit_partition_stability(update, outer, [], block_index=0)


def test_separate_source_terms_require_their_own_certificate():
    state = _state(0)
    transport, diffusion = _pair(state)
    source = _rate(12, "source", state)
    update = _combine(20, (state, {0: 1}), (transport, {1: 1}),
                      (diffusion, {1: 1}), (source, {1: 1}))
    with pytest.raises(ValueError, match="convex affine stability certificate"):
        partition_stability_groups(update)


def test_each_consumer_checks_its_own_weight_even_when_the_rate_is_reused():
    from pops.codegen.program_partition_stability import require_deferred_partition_bounds

    state = _state(0)
    transport, diffusion = _pair(state)
    safe = _combine(20, (state, {0: 1}), (transport, {1: 1}), (diffusion, {1: 1}))
    twice = _combine(21, (state, {0: 1}), (diffusion, {1: 2}))
    var = {("partition_frequency", transport.id): "transport",
           ("partition_frequency", diffusion.id): "diffusion",
           ("partition_stability_deferred",): frozenset((diffusion.id,))}
    with pytest.raises(ValueError, match="lacks its consuming bound"):
        require_deferred_partition_bounds(var)
    emit_partition_stability(safe, var, [], block_index=0, include_transport=True)
    lines = []
    emit_partition_stability(twice, var, lines, block_index=0, include_transport=True)
    assert partition_stability_groups(twice, include_transport=True) == (
        (state, Fraction(1), ((diffusion, Fraction(2)),)),)
    assert any("partition_frequency_21_0" in line and "diffusion" in line for line in lines)
    require_deferred_partition_bounds(var)
