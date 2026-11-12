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


def _linear_solve(rhs, guess, *, weights=({0: 1},), block=(), offset=40):
    """A consumed ``solve_linear`` A X = b of a traced method (lot « solves » of N15): the apply
    holds ``apply_in`` with each weight of ``weights`` and a laplacian of it, its block reads
    ``block`` beside them, and the commit is the solution component of its outcome."""
    applied = SimpleNamespace(id=offset, op="apply_in", inputs=(), attrs={})
    field = SimpleNamespace(id=offset + 1, op="scalar_field", inputs=(), attrs={})
    laplacian = SimpleNamespace(id=offset + 2, op="laplacian", inputs=(field, applied), attrs={})
    terms = tuple((applied, weight) for weight in weights) + ((laplacian, {1: Fraction(-1, 100)}),)
    operator = SimpleNamespace(
        id=offset + 3, op="matrix_free_operator", inputs=(),
        attrs={"apply_in": applied, "apply_result": SimpleNamespace(terms=terms),
               "apply_block": (applied, field, laplacian) + tuple(block)})
    solve = SimpleNamespace(id=offset + 4, op="solve_linear", inputs=(operator, rhs, guess),
                            attrs={})
    outcome = SimpleNamespace(id=offset + 5, op="solve_outcome", inputs=(solve,), attrs={})
    return SimpleNamespace(id=offset + 6, op="solve_outcome_component", inputs=(outcome,),
                           attrs={})


def test_a_linear_solve_of_unit_weight_carries_the_transport_of_its_right_hand_side():
    """X = u + dt E(u) + dt L(X): the captured transport of E is that of the rhs, weight dt."""
    state = _state(0)
    rate = _rate(10, "rhs", state)
    rhs = _combine(20, (state, {0: 1}), (rate, {1: 1}))
    program = SimpleNamespace(_commits={"state": _linear_solve(rhs, state)})
    assert accepted_transport_quadrature(program, {10: rate}) == ((10, {1: Fraction(1)}),)


@pytest.mark.parametrize("case", ["weight-two", "apply-reads-captured", "unknown-twice"])
def test_a_linear_solve_outside_the_rule_hides_transport(case):
    """A weight of the unknown other than one, an apply block that reads a captured rate, or the
    unknown in two terms of the apply: the attribution to the rhs does not hold, and the commit
    is refused as hiding transport."""
    state = _state(0)
    rate = _rate(10, "rhs", state)
    rhs = _combine(20, (state, {0: 1}), (rate, {1: 1}))
    options = {"weight-two": {"weights": ({0: 2},)},
               "apply-reads-captured": {"block": (rate,)},
               "unknown-twice": {"weights": ({0: 1}, {1: -1})}}[case]
    program = SimpleNamespace(_commits={"state": _linear_solve(rhs, state, **options)})
    with pytest.raises(ValueError, match="hides transport"):
        accepted_transport_quadrature(program, {10: rate})
