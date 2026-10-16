"""Source-only adversarial proofs for affine SSA equations, not native receipts."""
import pytest

from pops.time._program.affine_expressions import affine_field_terms
from pops.time._program.expressions import validate_pointwise_rate
from pops.codegen.program_emit_stencil_flux import accepted_stencil_quadrature
from pops.codegen.program_partition_stability import explicit_update_consumers
from test_elementary_algebra_v02 import _spatial_rate_fixture
from test_program_expressions import emit


def test_uniform_ssa_coefficient_preserves_equation_and_accepted_face_weight():
    model, p, q, rate = _spatial_rate_fixture()
    h = p.scalar_expression("duration", .001)
    update = p.value("ssa_equation", q.n+h*rate, at=q.next.point)
    assert update.op == "pointwise_expression"
    terms = affine_field_terms(update)
    assert terms is not None and [node.id for node, _ in terms] == [q.n.id, rate.id]
    p.commit(q.next, update)
    assert update.id in explicit_update_consumers(p)
    rows = accepted_stencil_quadrature(p)
    assert len(rows) == 1 and rows[0][0] is rate and not isinstance(rows[0][1], dict)
    cpp = emit(model, p)
    assert "ctx.attach_affine_flux_expression" in cpp
    assert "runtime_affine_spatial_stability" in cpp
    assert "ctx.capture_stencil_exchanges" in cpp
    assert "ctx.publish_stencil_exchanges" in cpp
    assert "std::bit_cast<std::int64_t>" in cpp
    assert "pops::for_each_cell" in cpp  # the original equation still executes


def test_local_source_consumes_the_exact_ssa_affine_stage():
    model, p, q, rate = _spatial_rate_fixture()
    h = p.scalar_expression("duration", .001)
    stage = p.value("stage", q.n+h*rate, at=q.n.point)
    local = p.rate("local", (-stage[0], -stage[1]), of=stage)
    assert validate_pointwise_rate(local) is stage
    accepted = p.value("accepted", q.n+p.dt*rate+p.dt*local, at=q.next.point)
    p.commit(q.next, accepted)
    assert stage.id in explicit_update_consumers(p)
    cpp = emit(model, p)
    assert "runtime_affine_spatial_stability" in cpp
    assert "user_face_numerical_stability" in cpp


@pytest.mark.parametrize("kind", ("nonlinear", "different_component", "different_coefficient"))
def test_non_affine_component_algebra_does_not_manufacture_provenance(kind):
    _, p, q, rate = _spatial_rate_fixture()
    h = p.scalar_expression("duration", .001)
    if kind == "nonlinear":
        expression = q.n+h*rate*rate
    elif kind == "different_component":
        expression = (q.n[0]+h*rate[1], q.n[1]+h*rate[0])
    else:
        expression = (q.n[0]+h*rate[0], q.n[1]+2*h*rate[1])
    stage = p.value("not_a_full_field_affine_map", expression, at=q.n.point)
    assert affine_field_terms(stage) is None
    with pytest.raises(ValueError, match="cannot absorb a spatial"):
        p.rate("concealed", (-stage[0], -stage[1]), of=stage)


def test_signed_coefficients_are_algebraically_legal_and_default_certificate_is_explicit():
    model, p, q, rate = _spatial_rate_fixture()
    h = p.scalar_expression("negative_duration", -.001)
    stage = p.value("signed_equation", q.n+h*rate, at=q.next.point)
    assert affine_field_terms(stage) is not None
    p.commit(q.next, stage)
    cpp = emit(model, p)
    assert "runtime_affine_spatial_stability" in cpp
    assert "affine_coefficients_%d[1] < 0" % stage.id in cpp
