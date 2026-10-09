"""Exact local affine budgets distinguish partitions, stage values and unsupported weights."""
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops.codegen.program_partition_stability import (
    emit_partition_stability, emit_user_face_stability, explicit_update_consumers,
    partition_stability_groups,
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
    rate_sum.vtype = "rhs"  # An observed sum of rates is not an accepted state.
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


def test_user_face_guard_uses_consumer_coefficient_and_skips_diagnostic_rate():
    state = _state(0)
    accepted_rate = _rate(10, "rhs", state)
    diagnostic_rate = _rate(11, "rhs", state)
    accepted = _combine(20, (state, {0: 1}), (accepted_rate, {1: Fraction(1, 100)}))
    diagnostic = _combine(21, (state, {0: 1}), (diagnostic_rate, {1: 1}))
    program = SimpleNamespace(_commits={"state": accepted}, _values=(accepted, diagnostic))
    assert explicit_update_consumers(program) == frozenset({20})
    lines = []
    emit_user_face_stability(accepted,
                             {("user_face_frequency", 10): "face_speed",
                              ("partition_frequency", 10): "face_speed"},
                             lines, block_index=0)
    assert any("pops::Real(1) / pops::Real(100)" in line for line in lines)
    assert any("face_speed" in line for line in lines)
    assert any("numerical_face_courant()" in line for line in lines)


def test_user_face_guard_follows_explicit_predictor_but_not_implicit_residual():
    state = _state(0)
    first = _rate(10, "rhs", state)
    predictor = _combine(20, (state, {0: 1}), (first, {1: 1}))
    second = _rate(11, "rhs", predictor)
    accepted = _combine(21, (state, {0: Fraction(1, 2)}),
                        (predictor, {0: Fraction(1, 2)}),
                        (second, {1: Fraction(1, 2)}))
    assert explicit_update_consumers(SimpleNamespace(_commits={"state": accepted})) == \
        frozenset({20, 21})
    implicit = SimpleNamespace(id=30, op="solve_implicit_source", vtype="state",
                               inputs=(predictor,), attrs={}, block=state.block, point=1)
    assert explicit_update_consumers(SimpleNamespace(_commits={"state": implicit})) == frozenset()


def test_coupled_rate_and_loop_result_retain_every_explicit_predictor_budget():
    first, second = _state(0), _state(1)
    first_rate = _rate(10, "rhs", first)
    predictor = _combine(20, (first, {0: 1}), (first_rate, {1: 1}))
    group_rate = SimpleNamespace(id=11, op="principal_rate", vtype="rhs",
                                 inputs=(second, predictor), attrs={"target_input": 0},
                                 block=second.block, point=1)
    accepted = _combine(21, (second, {0: 1}), (group_rate, {1: 1}))
    assert explicit_update_consumers(SimpleNamespace(_commits={"state": accepted})) == \
        frozenset({20, 21})
    loop = SimpleNamespace(id=30, op="range", vtype="state", inputs=(first,),
                           attrs={"body": predictor}, block=first.block, point=1)
    assert explicit_update_consumers(SimpleNamespace(_commits={"state": loop})) == \
        frozenset({20})


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


def test_butcher_ssprk2_recovers_the_retained_stage_without_changing_the_program():
    initial = _state(0)
    first = _rate(10, "rhs", initial)
    predictor = _combine(20, (initial, {0: 1}), (first, {1: 1}))
    second = _rate(30, "rhs", predictor)
    final = _combine(40, (initial, {0: 1}), (first, {1: Fraction(1, 2)}),
                     (second, {1: Fraction(1, 2)}))
    original_inputs, original_coefficients = final.inputs, final.attrs["coeffs"]
    assert partition_stability_groups(final, include_transport=True) == (
        (predictor, Fraction(1, 2), ((second, Fraction(1, 2)),)),)
    assert final.inputs is original_inputs and final.attrs["coeffs"] is original_coefficients


def test_acceptance_guard_keeps_explicit_budget_and_is_only_a_value_alias():
    initial = _state(0)
    condition = SimpleNamespace(id=2, op="constant", vtype="bool", inputs=(), attrs={})

    def guard(identifier, value):
        return SimpleNamespace(id=identifier, op="acceptance_guard", vtype="state",
                               inputs=(value, condition), attrs={}, block=value.block,
                               point=value.point)

    first = _rate(10, "rhs", guard(3, initial))
    predictor = _combine(20, (initial, {0: 1}), (first, {1: 1}))
    second = _rate(30, "rhs", guard(21, predictor))
    final = _combine(40, (initial, {0: 1}), (first, {1: Fraction(1, 2)}),
                     (second, {1: Fraction(1, 2)}))
    assert explicit_update_consumers(SimpleNamespace(_commits={"state": guard(41, final)})) == \
        frozenset({20, 40})
    assert partition_stability_groups(predictor, include_transport=True) == (
        (initial, Fraction(1), ((first, Fraction(1)),)),)
    assert partition_stability_groups(final, include_transport=True) == (
        (predictor, Fraction(1, 2), ((second, Fraction(1, 2)),)),)


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


# Real public SSA: Program.value retains State-shaped storage for a rate sum.
# Consumption role, rather than a synthetic vtype rewrite, owns its CFL budget.
def _public_rate_program():
    import pops
    from pops import math
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    frame = Rectangle("role_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("independent_rate_algebra", frame=frame)
    state = model.state("material", components=("q",))
    x, y = frame.axes
    first_flux = model.flux("first_flux", state=state, frame=frame,
                            components={x: (state[0],), y: (0 * state[0],)})
    second_flux = model.flux("second_flux", state=state, frame=frame,
                             components={x: (2 * state[0],), y: (0 * state[0],)})
    first = model.rate("first_balance", equation=math.ddt(state) == -math.div(first_flux))
    second = model.rate("second_balance", equation=math.ddt(state) == -math.div(second_flux))
    case = pops.Case("independent_consumer_case")
    block = case.block("material", model)
    program = pops.Program("independent_consumer_composition")
    temporal = program.state(block[state])
    return program, temporal, first, second


@pytest.mark.parametrize("weights", [(Fraction(1, 2), Fraction(3, 4)),
                                      (Fraction(2, 3), Fraction(1, 5))])
def test_real_public_rate_sum_consumes_one_state_budget(weights):
    p, t, first, second = _public_rate_program()
    a, b = first(t.n), second(t.n)
    observed = p.value("observed_rates", weights[0] * a + weights[1] * b, at=t.n.point)
    update = p.value("accepted", t.n + p.dt * observed, at=t.next.point)
    p.commit(t.next, update)
    before = p._serialize()
    assert observed.vtype == "state"  # Actual storage type, never hand-retagged rhs.
    assert explicit_update_consumers(p) == frozenset({update.id})
    assert partition_stability_groups(update, include_transport=True) == (
        (t.n, Fraction(1), ((a, weights[0]), (b, weights[1]))),)
    lines = []
    emit_user_face_stability(update,
        {("user_face_frequency", a.id): "fa", ("user_face_frequency", b.id): "fb",
         ("partition_frequency", a.id): "fa", ("partition_frequency", b.id): "fb"},
        lines, block_index=0)
    assert any("numerical_face_courant()" in line and "503" in line for line in lines)
    assert p._serialize() == before


@pytest.mark.parametrize("damage", ["negative_rate", "quadratic_dt", "state_weight"])
def test_real_observation_coefficients_never_relax_consuming_budget(damage):
    p, t, first, second = _public_rate_program()
    a, b = first(t.n), second(t.n)
    observed = p.value("observed", a - b if damage == "negative_rate" else a + b, at=t.n.point)
    if damage == "quadratic_dt":
        expression = t.n + p.dt * p.dt * observed
    elif damage == "state_weight":
        expression = 2 * t.n + p.dt * observed
    else:
        expression = t.n + p.dt * observed
    update = p.value("accepted", expression, at=t.next.point)
    p.commit(t.next, update)
    assert explicit_update_consumers(p) == frozenset({update.id})
    with pytest.raises(ValueError, match="convex affine stability certificate"):
        partition_stability_groups(update, include_transport=True)


def test_real_observed_history_does_not_add_a_diagnostic_state_budget():
    p, t, first, second = _public_rate_program()
    a, b = first(t.n), second(t.n)
    diagnostic = p.value("diagnostic_rates", a + 7 * b, at=t.n.point)
    p.store_history("actual_rate_observation", diagnostic, depth=1)
    observed = p.value("accepted_rates", a + b, at=t.n.point)
    update = p.value("accepted", t.n + p.dt * observed, at=t.next.point)
    p.commit(t.next, update)
    assert diagnostic.vtype == observed.vtype == "state"
    assert explicit_update_consumers(p) == frozenset({update.id})
    assert partition_stability_groups(update, include_transport=True)[0][2] == ((a, 1), (b, 1))


@pytest.mark.parametrize("scaled", [False, True])
def test_real_committed_rate_only_state_still_refuses_no_base(scaled):
    p, t, first, second = _public_rate_program()
    a, b = first(t.n), second(t.n)
    expression = p.dt * (a + b) if scaled else a + b
    invalid = p.value("unanchored_state", expression, at=t.next.point)
    p.commit(t.next, invalid)
    assert invalid.id in explicit_update_consumers(p)
    with pytest.raises(ValueError, match="explicit state consumer has no state weight"):
        partition_stability_groups(invalid, include_transport=True)


def test_real_shared_rate_observation_later_sampled_as_state_requires_budget():
    p, t, first, second = _public_rate_program()
    a, b = first(t.n), second(t.n)
    observed = p.value("observed", a + b, at=t.n.point)
    # Traversal reaches observed first as a contribution, then as this rate's State input.
    sampled = first(observed)
    update = p.value("accepted", t.n + p.dt * observed + p.dt * sampled, at=t.next.point)
    p.commit(t.next, update)
    assert explicit_update_consumers(p) == frozenset({observed.id, update.id})
    with pytest.raises(ValueError, match="explicit state consumer has no state weight"):
        partition_stability_groups(observed, include_transport=True)


def test_real_nested_stage_sum_keeps_each_sampled_predictor_budget():
    p, t, first, second = _public_rate_program()
    a, b = first(t.n), second(t.n)
    observed = p.value("first_rates", a + b, at=t.n.point)
    point = p.stage("sampled_predictor", c=1)
    predictor = p.value("predictor", t.n + p.dt * observed, at=point)
    c, d = first(predictor), second(predictor)
    last = p.value("last_rates", c + d, at=point)
    accepted = p.value("accepted", (t.n + predictor + p.dt * last) / 2, at=t.next.point)
    p.commit(t.next, accepted)
    assert explicit_update_consumers(p) == frozenset({predictor.id, accepted.id})
    assert partition_stability_groups(predictor, include_transport=True)[0][2] == ((a, 1), (b, 1))
    assert partition_stability_groups(accepted, include_transport=True) == (
        (predictor, Fraction(1, 2), ((c, Fraction(1, 2)), (d, Fraction(1, 2)))),)


def test_actual_atomic_forward_euler_rhs_alias_keeps_equations_and_guard():
    import pops
    from pops.codegen import Production
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_codegen import emit_cpp_program
    from tests.python.support.atomic_cubature_path_case import make_case
    from tests.python.support.atomic_cubature_fv_oracle import DT
    case, layout, _ = make_case(nonconservative=True, amr=True, fixed_dt=DT, scale_parameter=True)
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
    program = resolved.time
    before = program._serialize()
    (accepted,) = program._commits.values()
    observed = next(value for value in accepted.inputs if value.op == "linear_combine")
    assert observed.vtype == "state" and all(value.op == "rhs" for value in observed.inputs)
    assert explicit_update_consumers(program) == frozenset({accepted.id})
    code = emit_cpp_program(program, model_graph=build_program_model_graph(resolved), target="amr_system")
    assert code.count('"user_face_numerical_stability",503') == 1
    assert 'user_face_update_frequency_%d_0' % accepted.id in code
    assert 'user_face_update_frequency_%d_0' % observed.id not in code
    assert "ctx.numerical_face_courant()" in code and "32 * std::numeric_limits<pops::Real>::epsilon()" in code
    assert "ctx.axpy(" in code and "ctx.commit_many(" in code
    assert program._serialize() == before
