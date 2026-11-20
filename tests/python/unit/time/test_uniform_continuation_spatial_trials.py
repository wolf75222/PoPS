"""The complete Uniform carrier retains provisional results of spatial trials."""
import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time import ComputedDt
from pops.time._program.control import AdvanceControl
from pops.time._program.detach import detach_compiled_program
from tests.python.integration.runtime import test_implicit_diffusion_lifecycle as diffusion
from tests.python.integration.runtime import test_native_retry_transactions as source
from tests.python.support.uniform_continuation_trial import publish_trial


def test_compiled_advance_and_computed_duration_cannot_own_the_same_trial():
    def publish_computed_trial(program, temporal, trial):
        program.reached_duration(0.5 * program.requested_dt())
        publish_trial(program, temporal, trial)

    case, layout = source.make_case(direct=True, publish=publish_computed_trial)
    case._time.step_strategy(ComputedDt(1/8))
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    cpp = emit_cpp_program(detach_compiled_program(plan.time), model_graph=graph)
    assert "ctx.reached_duration(" in cpp and cpp.count("ctx.continuation_publish(") == 1

    case, layout = source.make_case(direct=True, publish=publish_computed_trial)
    case._time.step_strategy(ComputedDt(1/8))
    case._time._set_advance_control(AdvanceControl(
        assessments=(("assessment", "continuation"),), first_dt=1/8,
        max_attempts=2, max_transforms=0,
    ))
    with pytest.raises(ValueError, match="Compiled Advance owns the step duration"):
        pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="Compiled Advance owns the step duration"):
        emit_cpp_program(detach_compiled_program(case._time), model_graph=graph)


@pytest.mark.parametrize("op", ["source", "diffusive_rhs"])
def test_uniform_spatial_trial_uses_one_complete_candidate_publication(op):
    if op == "source":
        case, layout = source.make_case(direct=True, publish=publish_trial)
    else:
        case, layout = diffusion.make_case(8, method="forward_euler", dt=1/32,
                                           publish=publish_trial)
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert sum(value.op == op for value in plan.time._values) == 1
    model_graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    cpp = emit_cpp_program(detach_compiled_program(plan.time), model_graph=model_graph)
    assert cpp.count("ctx.continuation_publish(") == 1
    assert "ctx.continuation_initialize(" in cpp
    if op == "diffusive_rhs":
        assert cpp.count(".stage_accepted_exchanges(ctx,") == 1
        assert cpp.index("ctx.continuation_publish(") < cpp.index(".stage_accepted_exchanges(ctx,")


def test_uniform_continuation_refuses_a_changed_source_under_a_valid_handle():
    def publish_with_changed_source(program, temporal, trial):
        rate = next(value for value in program._values if value.op == "source")
        program._replace_value(rate, attrs={**rate.attrs, "source": "different_source"})
        publish_trial(program, temporal, trial)

    case, layout = source.make_case(direct=True, publish=publish_with_changed_source)
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="changes its declared source"):
        emit_cpp_program(detach_compiled_program(plan.time),
                         model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))


def test_uniform_continuation_refuses_changed_diffusion_under_a_valid_handle():
    from pops._ir.balance import Balance, BalanceView

    def publish_with_changed_diffusion(program, temporal, trial):
        rate = next(value for value in program._values if value.op == "diffusive_rhs")
        view = rate.attrs["physical_balance"]
        balance = view.balance
        changed = Balance.capture(
            balance.handle, balance.target,
            tuple((row.kind, row.payload, 2 * row.coefficient) for row in balance.occurrences),
            balance.accumulation,
        )
        program._replace_value(rate, attrs={**rate.attrs, "physical_balance": BalanceView(changed, view.ordinals)})
        publish_trial(program, temporal, trial)

    case, layout = diffusion.make_case(8, method="forward_euler", dt=1/32,
                                       publish=publish_with_changed_diffusion)
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="changes its declared diffusive balance"):
        emit_cpp_program(detach_compiled_program(plan.time),
                         model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))


def test_uniform_continuation_rejects_a_trial_outside_its_candidate_lineage():
    def publish_unchanged(program, temporal, trial):
        unchanged = program.value("unchanged", temporal.n, at=temporal.next.point)
        publish_trial(program, temporal, unchanged)

    case, layout = diffusion.make_case(8, method="forward_euler", dt=1/32,
                                       publish=publish_unchanged)
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="does not contribute to its root candidate"):
        plan.time.validate()


def test_uniform_continuation_rejects_attestation_of_the_same_typed_sibling():
    def publish_with_sibling(program, temporal, trial):
        publish_trial(program, temporal, trial, attest_sibling=True)

    case, layout = diffusion.make_case(8, method="forward_euler", dt=1/32,
                                       publish=publish_with_sibling)
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="attestation reads a different frozen candidate"):
        emit_cpp_program(detach_compiled_program(plan.time),
                         model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))


def test_uniform_continuation_rejects_a_branch_even_with_one_possible_field():
    from pops.codegen.program_structural_flux import field_projection_sources

    def publish_branch(program, temporal, trial):
        selected = program.branch(program.requested_dt() > 0.02,
                                  lambda _: trial, lambda _: trial)
        publish_trial(program, temporal, selected)

    case, layout = diffusion.make_case(8, method="forward_euler", dt=1/32,
                                       publish=publish_branch)
    plan = pops.resolve(pops.validate(case), layout=layout)
    commit, = plan.time._commits.values()
    assert len(field_projection_sources(commit)) == 1
    with pytest.raises(ValueError, match="requires one direct candidate State field"):
        emit_cpp_program(detach_compiled_program(plan.time),
                         model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
