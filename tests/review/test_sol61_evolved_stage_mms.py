"""Source/math admission for evolved-stage installed witnesses, no native DSO."""
from fractions import Fraction

import numpy as np
import pytest

from tests.python.support.evolved_stage_mms import (
    ACCEPTANCE, accumulation, check_original, diffusion_action, manufactured_data,
)


@pytest.mark.parametrize("cells", (8, 16))
@pytest.mark.parametrize("width,candidate", ((1, False), (2, True)))
@pytest.mark.parametrize("dt", (.01, .02))
def test_independent_discrete_load_closes_same_accumulation(cells, width, candidate, dt):
    initial, t0, target, _, volumes = manufactured_data(
        cells, width, dt, candidate_diffusion=candidate)
    q0 = np.concatenate(tuple(initial["Q%d" % i] for i in range(width)))
    np.testing.assert_array_equal(q0, accumulation(t0))
    metrics = check_original(target, accumulation(target), q0, initial["forcing"], dt,
                             candidate_diffusion=candidate, target=target)
    assert metrics["original_relative_l2"] < 1e-14
    assert np.sum(volumes) == 1
    assert np.max(np.abs(initial["forcing"])) > .1
    if width == 2:
        z = .25*target[0]+.5*target[1]
        assert np.max(np.abs(z-.25*target[0]-.5*target[1])) <= np.finfo(float).eps


def test_fraction_periodic_face_balance_and_signed_cross_flux():
    # Independent exact two-cell quotient, distinct from NumPy FV implementation.
    t = ((Fraction(1, 5), Fraction(3, 10)), (Fraction(2, 5), Fraction(1, 2)))
    d = ((Fraction(3, 250), Fraction(1, 500)),
         (Fraction(-1, 1000), Fraction(7, 500)))
    face = tuple(sum(d[i][j]*2*(t[j][1]-t[j][0]) for j in range(2)) for i in range(2))
    opposite = tuple(-value for value in face)
    volume = Fraction(1, 2)
    action = tuple(((face[i]-opposite[i])/volume,
                    (opposite[i]-face[i])/volume) for i in range(2))
    assert all(sum(volume*cell for cell in row) == 0 for row in action)
    assert d[1][0] < 0 and face[0] != face[1]


def test_captured_load_is_not_a_reaction_times_temperature():
    initial, _, t, _, _ = manufactured_data(8, 1, .01)
    q0 = initial["Q0"]
    wrong_q = q0+.01*(diffusion_action(t)[0]+initial["forcing"]*t)
    with pytest.raises(AssertionError):
        check_original(t, wrong_q, q0, initial["forcing"], .01)
    assert ACCEPTANCE == 3e-8


def test_scalar_additive_original_case_uses_actual_RHS_and_duration():
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.support.evolved_stage_mms import build

    case, layout, _ = build(8, 1, .01)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert resolved.time._serialize()["version"] == 14
    solve = next(row for row in resolved.time._values if row.op == "solve_spatial_field")
    assert solve.attrs["source_contract"]["evolved_stage"]["schema_version"] == 2
    assert "ctx.step_dt()" in cpp and "original_field_residual_recheck_failed" in cpp
    assert "field.evolution" not in cpp


def test_additive_source_refuses_unknown_dependency_before_solver():
    from pops.model import Handle, OwnerPath
    from pops.math import ValueExpr
    from pops.time import EvolvedOriginalFieldRate
    from pops._ir.elliptic import DivCoeffGrad

    t = Handle("T", kind="field", owner=OwnerPath.model("material"))
    with pytest.raises(NotImplementedError, match="per-candidate RHS"):
        EvolvedOriginalFieldRate(DivCoeffGrad(t, .012), ValueExpr(t))
    assert EvolvedOriginalFieldRate(DivCoeffGrad(t, .012), .5).to_data()["contract"].endswith("@1")


@pytest.mark.parametrize("candidate", (False, True))
def test_two_separate_Q_carriers_three_unknowns_resolve_emit_same_original_stage(candidate):
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.support.evolved_stage_mms import build

    case, layout, program = build(8, 2, .02, candidate_diffusion=candidate)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert resolved.time._serialize()["version"] == 15
    projections = [row for row in program._values if row.op == "field_evolved_state"]
    assert len(projections) == 2 and len({row.state_ref for row in projections}) == 2
    assert {row.attrs["component_indices"] for row in projections} == {(0,), (1,)}
    assert all(row.attrs["projection_contract"].endswith("@2") for row in projections)
    assert "nonfinite_original_accumulation" in cpp and "original_field_residual_recheck_failed" in cpp
    assert "ctx.step_dt()" in cpp


def test_partition_requires_both_actual_commits_and_unmodified_component_selection():
    from tests.python.support.evolved_stage_mms import build
    from pops.fields._evolved_stage_contract import validate_evolved_state
    _, _, program = build(8, 2, .01)
    projections = [row for row in program._values if row.op == "field_evolved_state"]
    old = program._commits.pop(projections[1].state_ref)
    with pytest.raises(ValueError, match="committed exactly once"):
        program.validate()
    program._commits[old.state_ref] = old
    program.validate()
    # An equal-width foreign Q selection must fail even with the same solve.
    forged = program._replace_value(projections[0], attrs={**projections[0].attrs, "component_indices": (1,)})
    with pytest.raises(ValueError, match="exact State partition"):
        validate_evolved_state(forged)


def test_pure_additive_rhs_and_reversed_carrier_order_are_source_admissible():
    import pops
    from tests.python.support.evolved_stage_mms import build
    for width, spatial, reverse in ((1, False, False), (2, True, True)):
        case, layout, program = build(8, width, .01, spatial=spatial, reverse=reverse)
        resolved = pops.resolve(pops.validate(case), layout=layout)
        assert resolved.time._serialize()["version"] == (14 if width == 1 else 15)
        if reverse:
            projections = [row for row in program._values if row.op == "field_evolved_state"]
            assert [row.attrs["component_indices"] for row in projections] == [(1,), (0,)]


def test_additive_capture_is_required_and_mutated_unknown_source_is_refused():
    from tests.python.support.evolved_stage_mms import build
    from pops.model import Handle, OwnerPath
    from pops.math import ValueExpr
    from pops.time import EvolvedOriginalFieldRate
    with pytest.raises(ValueError, match="missing exact qualified equation capture"):
        build(8, 1, .01, omit_forcing=True)
    descriptor = EvolvedOriginalFieldRate(None, .5)
    t = Handle("foreign", kind="field", owner=OwnerPath.model("other"))
    object.__setattr__(descriptor, "additive", ValueExpr(t))
    with pytest.raises(NotImplementedError, match="per-candidate RHS"):
        descriptor.to_data()


def test_current_source_read_cannot_impersonate_future_but_real_candidate_is_allowed():
    from pops.fields._evolved_stage_contract import validate_additive_capture_reads
    from tests.python.support.evolved_stage_mms import build
    from pops.time.points import TimePoint
    _, _, program = build(8, 1, .01)
    state = next(row for row in program._values if row.op == "state" and row.state_ref.name == "forcing")
    candidate = program.value("explicit-load-candidate", 1*state, at=TimePoint(program.clock, 1))
    validate_additive_capture_reads((candidate,))
    forged = program._replace_value(state, point=TimePoint(program.clock, 1))
    with pytest.raises(ValueError, match="accepted endpoint"):
        validate_additive_capture_reads((forged,))
    candidate_with_forged_read = program.value("foreign-read-candidate", 1*forged, at=TimePoint(program.clock, 1))
    with pytest.raises(ValueError, match="accepted endpoint"):
        validate_additive_capture_reads((candidate_with_forged_read,))
