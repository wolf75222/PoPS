"""Elementary expressions retain their semantics across the scientific/Program frontier."""
import math

import pytest

from pops._ir.expr import Var, UnaryMath
from pops._ir.lowering import diff
from pops._ir.visitors import _key
from pops.math import sin, cos, log, expm1


@pytest.mark.parametrize("function, derivative", [
    (sin, math.cos), (cos, lambda x: -math.sin(x)),
    (log, lambda x: 1 / x), (expm1, math.exp),
])
def test_composed_derivative_and_distinct_identity(function, derivative):
    x = Var("x", "cons")
    expression = function(x * x + 0.3)
    tangent = diff(expression, x)
    for value in (0.05, 0.7, 1.4):
        assert tangent.eval({"x": value}) == pytest.approx(
            2 * value * derivative(value * value + 0.3), rel=2e-14)
    assert _key(expression)[0] == function.__name__


def test_variable_exponent_and_negative_integer_base_domains():
    x = Var("x", "cons")
    assert diff(x ** x, x).eval({"x": 1.3}) == pytest.approx(
        1.3 ** 1.3 * (math.log(1.3) + 1))
    assert diff(x ** 3, x).eval({"x": -2}) == 12


def test_elementary_node_cannot_inject_arbitrary_backend_function():
    class External(UnaryMath):
        operation = "model_specific_closure"
    with pytest.raises(TypeError, match="unsupported elementary"):
        External(1)


def test_program_body_uses_the_common_lazy_native_emitter():
    from test_program_expressions import fixture, emit
    from pops.math import where
    model, _, program, state = fixture()
    x, y = state.n[0], state.n[1]
    result = program.value("elementary", (
        sin(x) + cos(y), where(x > 0, lambda: log(x), lambda: expm1(y))),
        at=state.next.point)
    program.commit(state.next, result)
    cpp = emit(model, program)
    for name in ("sin", "cos", "log", "expm1"):
        assert "Kokkos::" + name in cpp
    # Both branches must remain guarded; eagerly evaluating log(-1) is invalid.
    assert "?" in cpp
    assert program.validate()


def test_numerical_pointwise_rate_retains_exact_state_space_and_native_body():
    from test_program_expressions import fixture, emit
    from pops.time._program.expressions import validate_pointwise_rate
    from pops.codegen.program_partition_stability import partition_stability_groups
    model,_,p,state=fixture()
    rate=p.rate("authored_rhs",(-state.n[0],2*state.n[1]),of=state.n)
    assert rate.vtype=="rhs" and validate_pointwise_rate(rate) is state.n
    candidate=p.value("advance",state.n+p.dt*rate,at=state.next.point)
    groups=partition_stability_groups(candidate,include_transport=True)
    assert groups[0][0] is state.n and groups[0][2][0][0] is rate
    p.commit(state.next,candidate)
    assert "pops::for_each_cell" in emit(model,p)


def test_numerical_rate_cannot_relabel_a_foreign_template_or_point():
    from test_program_expressions import fixture
    from pops.time.expressions import ProgramExpression
    _,_,p,state=fixture()
    provisional=p.value("other",state.n,at=state.next.point)
    with pytest.raises(ValueError,match="template"):
        p.rate("wrong",ProgramExpression((1.,2.),provisional),of=state.n)
    with pytest.raises(ValueError,match="evaluation point"):
        p.rate("wrong",(state.n[0],state.n[1]),of=state.n,at=state.next.point)


def test_numerical_rate_revalidates_exact_output_provenance():
    from types import SimpleNamespace
    from test_program_expressions import fixture
    from pops.time._program.expressions import validate_pointwise_rate
    _,_,p,state=fixture()
    rate=p.rate("constant",(1.,2.),of=state.n)
    assert validate_pointwise_rate(rate) is state.n
    with pytest.raises(ValueError,match="authority"):
        validate_pointwise_rate(SimpleNamespace(op=rate.op,vtype="state",attrs=rate.attrs,
                                               inputs=rate.inputs))


def _spatial_rate_fixture():
    from test_program_expressions import fixture
    from pops.numerics.stencil_flux import StencilFlux
    model, _, p, q = fixture()
    flux = StencilFlux(dimension=2, components=2, radius=1,
        body=lambda sample, spacing, axis: (
            ((sample((-1, 0), 0), sample((-1, 0), 1)), 1/spacing[0])
            if axis == 0 else ((0., 0.), 0.)))
    return model, p, q, p.stencil_flux_divergence("spatial", q.n, flux=flux)


@pytest.mark.parametrize("alias", ("direct", "affine", "pointwise"))
def test_numerical_rate_cannot_erase_spatial_bounds_or_receipts(alias):
    from pops.time.expressions import ProgramExpression
    _, p, q, spatial = _spatial_rate_fixture()
    captured = spatial
    if alias == "affine":
        captured = p.value("rate_sum", 2*spatial, at=q.n.point)
    elif alias == "pointwise":
        captured = p.value("rate_components",
            ProgramExpression((spatial[0], spatial[1]), q.n), at=q.n.point)
    before = p._ir_hash()
    with pytest.raises(ValueError, match="cannot absorb a spatial"):
        p.rate("concealed_spatial", (captured[0], captured[1]), of=q.n)
    assert p._ir_hash() == before


def test_numerical_rate_rechecks_spatial_dependencies_at_lowering_frontier():
    from types import SimpleNamespace
    from pops.time._program.expressions import validate_pointwise_rate
    _, p, q, spatial = _spatial_rate_fixture()
    local = p.rate("local", (-q.n[0], -q.n[1]), of=q.n)
    forged = SimpleNamespace(id=-1, op=local.op, vtype=local.vtype,
        attrs=dict(local.attrs, target_input=1), inputs=(spatial, q.n),
        block=local.block, point=local.point, state_ref=local.state_ref, space=local.space)
    with pytest.raises(ValueError, match="cannot absorb a spatial"):
        validate_pointwise_rate(forged)


def test_local_rate_may_read_its_exact_spatially_advanced_stage():
    from pops.time._program.expressions import validate_pointwise_rate
    _, p, q, spatial = _spatial_rate_fixture()
    stage = p.value("stage", q.n+p.dt*spatial, at=q.next.point)
    local = p.rate("stage_local", (-stage[0], -stage[1]), of=stage)
    assert validate_pointwise_rate(local) is stage


@pytest.mark.parametrize("alias", ("affine", "pointwise", "nested_pointwise"))
def test_rate_target_cannot_relabel_a_spatial_derivative_as_a_state_stage(alias):
    from pops.time.expressions import ProgramExpression
    _, p, q, spatial = _spatial_rate_fixture()
    if alias == "affine":
        fake_state = p.value("rhs_as_state", spatial, at=q.n.point)
    else:
        fake_state = p.value("rhs_as_state",
            ProgramExpression((spatial[0], spatial[1]), q.n), at=q.n.point)
        if alias == "nested_pointwise":
            fake_state = p.value("hidden_rhs", q.n+p.dt*fake_state, at=q.next.point)
    before = p._ir_hash()
    with pytest.raises(ValueError, match="retagged rate|cannot absorb a spatial"):
        p.rate("concealed_target", (fake_state[0], fake_state[1]), of=fake_state)
    assert p._ir_hash() == before


def test_interval_collision_remains_local_and_keeps_separate_spatial_proofs():
    import pops
    from examples.api_v02.method_interval_transport import build_interval_transport
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_emit_stencil_flux import accepted_stencil_quadrature
    spec = build_interval_transport(cells=(8, 4), steps=1)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert len(accepted_stencil_quadrature(plan.time)) == 1
    assert "user_face_numerical_stability" in cpp
    assert "collision_interval_average" in cpp
