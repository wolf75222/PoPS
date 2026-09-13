"""Transport ledger weights follow accepted state algebra, never residual/seed ancestry."""
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops.codegen.program_transport_quadrature import accepted_transport_quadrature
from tests.python.unit.codegen.test_partition_stability import _state, _rate, _combine


def test_exact_accepted_weights_include_predictor_once_and_ignore_unused_rates():
    initial = _state(0)
    first = _rate(10, "rhs", initial)
    first_diffusion = _rate(11, "diffusive_rhs", initial)
    predictor = _combine(20, (initial, {0: 1}), (first, {1: 1}), (first_diffusion, {1: 1}))
    last = _rate(12, "rhs", predictor)
    last_diffusion = _rate(13, "diffusive_rhs", predictor)
    accepted = _combine(21, (initial, {0: Fraction(1, 2)}), (predictor, {0: Fraction(1, 2)}),
                        (last, {1: Fraction(1, 2)}), (last_diffusion, {1: Fraction(1, 2)}))
    program = SimpleNamespace(_commits={"state": accepted})
    assert accepted_transport_quadrature(program, {10: first, 12: last, 99: object()}) == (
        (10, {1: Fraction(1, 2)}), (12, {1: Fraction(1, 2)}))


def test_authenticated_implicit_endpoint_counts_previous_but_not_seed_or_residual(monkeypatch):
    from pops.time._program import spatial_solve

    initial = _state(0)
    previous_rate = _rate(10, "rhs", initial)
    seed_rate = _rate(11, "rhs", initial)
    residual_rate = _rate(12, "diffusive_rhs", initial)
    previous = _combine(20, (initial, {0: 1}), (previous_rate, {1: 1}))
    seed = _combine(21, (initial, {0: 1}), (seed_rate, {1: 1}))
    solve = SimpleNamespace(id=30, op="solve_spatial_nonlinear", inputs=(previous, seed),
                            attrs={"residual_block": (residual_rate,)})
    outcome = SimpleNamespace(id=31, op="solve_outcome", inputs=(solve,))
    candidate = SimpleNamespace(id=32, op="solve_outcome_component", inputs=(outcome,))
    wrapped = _combine(33, (candidate, {0: 1}))
    accepted = SimpleNamespace(id=34, op="local_transform", inputs=(wrapped,))
    program = SimpleNamespace(_commits={"state": accepted})
    authenticated = []
    monkeypatch.setattr(spatial_solve, "validate_spatial_commit",
                        lambda actual, token: authenticated.append((actual, token)))
    assert accepted_transport_quadrature(program, {10: previous_rate, 11: seed_rate}) == (
        (10, {1: Fraction(1)}),)
    assert authenticated == [(program, solve)]

    def refuse(actual, token):
        raise ValueError("invalid conservative endpoint")
    monkeypatch.setattr(spatial_solve, "validate_spatial_commit", refuse)
    with pytest.raises(ValueError, match="invalid conservative endpoint"):
        accepted_transport_quadrature(program, {10: previous_rate, 11: seed_rate})


@pytest.mark.parametrize("weight", ({1: -1}, {2: 1}))
def test_invalid_accepted_transport_weights_are_refused(weight):
    state = _state(0)
    rate = _rate(10, "rhs", state)
    accepted = _combine(20, (state, {0: 1}), (rate, weight))
    with pytest.raises(ValueError, match="nonnegative exact dt weights"):
        accepted_transport_quadrature(SimpleNamespace(_commits={"state": accepted}), {10: rate})


def test_source_evaluation_stops_ancestry_but_unknown_transform_does_not_hide_it():
    state = _state(0)
    rate = _rate(10, "rhs", state)
    predictor = _combine(20, (state, {0: 1}), (rate, {1: 1}))
    source = _rate(11, "rhs", predictor, flux=False)
    program = SimpleNamespace(_commits={"state": source})
    assert accepted_transport_quadrature(program, {10: rate}) == ()
    transformed = SimpleNamespace(id=21, op="local_transform", inputs=(predictor,))
    program._commits = {"state": transformed}
    with pytest.raises(ValueError, match="hides transport"):
        accepted_transport_quadrature(program, {10: rate})


def test_branch_results_and_scheduled_consumers_cannot_hide_selected_transport():
    state = _state(0)
    rate = _rate(10, "rhs", state)
    update = _combine(20, (state, {0: 1}), (rate, {1: 1}))
    branch = SimpleNamespace(id=21, op="branch", inputs=(),
                             attrs={"true_result": update, "false_result": state})
    program = SimpleNamespace(_commits={"state": branch})
    with pytest.raises(ValueError, match="hides transport"):
        accepted_transport_quadrature(program, {10: rate})
    update.attrs["schedule"] = object()
    program._commits = {"state": update}
    with pytest.raises(ValueError, match="scoped quadrature carrier"):
        accepted_transport_quadrature(program, {10: rate})
