"""A consumed native solve issues and later discharges its real Python equation."""
import pytest
import pops
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.unit.time.test_implicit_stage_request import make_source_stage


@pytest.mark.parametrize("nonlinear", [False, True])
def test_source_only_implicit_result_uses_same_equation_for_issue_and_discharge(nonlinear):
    case, layout = make_source_stage(nonlinear=nonlinear)
    plan = pops.resolve(pops.validate(case), layout=layout)
    emitter, _ = lower_and_validate(plan.blocks[0].model)
    source = emit_cpp_program(plan.time, model=emitter)
    if not nonlinear:
        assert "ctx.own_field_representation(" not in source
        return
    assert source.count("ctx.own_field_representation(") == 1
    assert "ctx.own_discharged_field_representation(" in source
    candidate = source.index("->candidate()")
    issue = source.index("ctx.own_field_representation(")
    discharge = source.index("ctx.own_discharged_field_representation(", issue)
    assert candidate < issue < discharge < source.index("ctx.commit_many(", discharge)
    assert "Field transformation has no retained production origin" not in source
    assert "transform_capture_.inputs.journal" in source
    assert "transform_capture_.coordinates.journal" in source


def test_representation_workspace_is_explicit_and_survives_detach():
    from pops.time._program.detach import detach_compiled_program
    p = pops.Program("representation_budget", representation_workspace_byte_capacity=(1 << 27))
    assert detach_compiled_program(p).representation_workspace_byte_capacity == 1 << 27
    from pops.identity.semantic import program_semantic_data
    assert p._serialize()["representation_workspace_byte_capacity"] == 1 << 27
    assert program_semantic_data(p)["representation_workspace_byte_capacity"] == 1 << 27
    old_hash = p._ir_hash()
    p.representation_workspace_byte_capacity += 1
    assert p._ir_hash() != old_hash
    with pytest.raises(ValueError, match="representation_workspace_byte_capacity"):
        pops.Program("invalid_representation_budget", representation_workspace_byte_capacity=0)


def test_native_receiver_declares_both_admission_and_atomic_refusal_profiles():
    from tests.python.support.nonlinear_representation import build, physical_step
    import numpy as np
    for capacity in (64 << 20, 1):
        case, layout, initial, dt = build(workspace_byte_capacity=capacity)
        plan = pops.resolve(pops.validate(case), layout=layout)
        emitter, _ = lower_and_validate(plan.blocks[0].model)
        source = emit_cpp_program(plan.time, model=emitter)
        assert "UINT64_C(%d)" % capacity in source
        assert "ctx.own_field_representation(" in source
        expected = physical_step(initial, dt)
        # Independent oracle directly satisfies the coupled physical residual.
        residual = expected - initial - dt*np.stack((-expected[0]+expected[1], expected[0]-2*expected[1]))
        np.testing.assert_allclose(residual, 0, rtol=0, atol=2.e-16)
