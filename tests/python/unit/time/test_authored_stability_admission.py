"""Admission is an explicit executed predicate, never an inferred stability claim."""
import pytest

from pops.time import FixedDt, GuardRole, RejectAttempt
from pops.codegen.program_partition_stability import explicit_update_consumption
from pops.time._program.stability_admission import is_stability_admission_guard
from test_elementary_algebra_v02 import _spatial_rate_fixture
from test_program_expressions import emit


def signed_candidate(p, q, rate, *, dynamic=False):
    h = p.float64(-.001) if dynamic else -p.dt
    return p.value("signed", q.n+h*rate, at=q.next.point)


@pytest.mark.parametrize("dynamic", (False, True))
def test_authored_guard_replaces_only_the_default_sufficient_certificate(dynamic):
    model, p, q, rate = _spatial_rate_fixture()
    candidate = signed_candidate(p, q, rate, dynamic=dynamic)
    guarded = p.guard("energy_admission", candidate,
        p.norm_inf(candidate) <= 2*p.norm_inf(q.n),
        action=RejectAttempt(), role=GuardRole.NUMERICAL_STABILITY)
    p.commit(q.next, guarded)
    p.step_strategy(FixedDt(.001))
    updates, covered, rates = explicit_update_consumption(p)
    assert candidate.id in updates and candidate.id in covered and rate.id in rates
    cpp = emit(model, p)
    assert "energy_admission" in cpp and "StepAttemptRejected" in cpp
    assert '"runtime_affine_spatial_stability"' not in cpp
    assert '"user_face_numerical_stability"' not in cpp
    assert p.transaction_plan().guards[0].role is GuardRole.NUMERICAL_STABILITY
    assert "stencil_faces_" in cpp


@pytest.mark.parametrize("kind", ("constant", "other_state", "template_only"))
def test_disconnected_admission_refuses_atomically(kind):
    _, p, q, rate = _spatial_rate_fixture()
    candidate = signed_candidate(p, q, rate)
    condition = p.boolean(True) if kind == "constant" else p.norm_inf(q.n) < 10
    # A field used only as output authority must not satisfy the data dependency.
    if kind == "template_only":
        from pops.time.expressions import ProgramExpression
        empty = p.value("constant_field", ProgramExpression((1., 1.), candidate))
        condition = p.norm_inf(empty) < 10
    before = p._ir_hash()
    with pytest.raises(ValueError, match="exact candidate"):
        p.guard("disconnected", candidate, condition,
                action=RejectAttempt(), role=GuardRole.NUMERICAL_STABILITY)
    assert before == p._ir_hash()


def test_an_unused_guard_cannot_admit_an_unguarded_commit():
    model, p, q, rate = _spatial_rate_fixture()
    candidate = signed_candidate(p, q, rate)
    p.guard("unused", candidate, p.norm_inf(candidate) < 10,
            action=RejectAttempt(), role=GuardRole.NUMERICAL_STABILITY)
    p.commit(q.next, candidate)
    _, covered, rates = explicit_update_consumption(p)
    assert not covered and not rates
    with pytest.raises(ValueError, match="convex affine"):
        emit(model, p)


def test_a_path_without_the_guard_retains_its_default_obligation():
    _, p, q, rate = _spatial_rate_fixture()
    candidate = signed_candidate(p, q, rate)
    guarded = p.guard("one_path", candidate, p.norm_inf(candidate) < 10,
        action=RejectAttempt(), role=GuardRole.NUMERICAL_STABILITY)
    selected = p.branch(p.norm_inf(q.n) > 0, lambda b: guarded, lambda b: candidate)
    p.commit(q.next, selected)
    _, covered, rates = explicit_update_consumption(p)
    assert candidate.id not in covered and rate.id not in rates


def test_an_invariant_guard_does_not_silently_change_the_numerical_contract():
    model, p, q, rate = _spatial_rate_fixture()
    candidate = signed_candidate(p, q, rate)
    guarded = p.guard("ordinary_invariant", candidate, p.norm_inf(candidate) < 10,
                      action=RejectAttempt())
    assert not is_stability_admission_guard(p, guarded)
    p.commit(q.next, guarded)
    with pytest.raises(ValueError, match="convex affine"):
        emit(model, p)


@pytest.mark.parametrize("retry", (False, True))
def test_native_signed_midpoint_fixture_is_fully_lowered_before_reception(retry):
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from examples.api_v02.method_authored_stability import build_authored_stability
    spec = build_authored_stability(retry=retry)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "authored_midpoint_energy" in cpp
    assert '"user_face_numerical_stability"' not in cpp
    assert "ctx.dot(" in cpp


def test_declared_diffusion_uses_the_same_authored_admission_authority():
    import pops
    from pops import math
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.numerics import Diffusion, DiscretizationPlan
    from examples.api_v02.models_M01_M14 import _domain
    frame, layout = _domain("signed_diffusion", (8, 8))
    model = pops.Model("diffusion", frame=frame)
    state = model.state("u", components=("u",))
    flux = model.diffusive_flux("heat", state=state, value=.01*math.grad(state))
    rate = model.rate("rate", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("signed_diffusion")
    block = case.block("field", model)
    numerics = DiscretizationPlan(); numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    p = pops.Program("authored_diffusion"); q = p.state(block[state])
    candidate = p.value("negative_rate", q.n-p.dt*rate(q.n), at=q.next.point)
    guarded = p.guard("authored_heat_admission", candidate, p.norm_inf(candidate) < 10,
                     action=RejectAttempt(), role=GuardRole.NUMERICAL_STABILITY)
    p.commit(q.next, guarded); p.step_strategy(FixedDt(.001)); case.program(p)
    plan = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "authored_heat_admission" in cpp
    assert '"combined_transport_diffusion_stability"' not in cpp
    assert "diffusion_prepared_" in cpp
