"""Structured original equations own typed continuation frames across real barriers."""
import pytest
import pops

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_public_amr_original_field import build


@pytest.mark.parametrize("mode", ("branch_true", "branch_false", "fold", "fold_overflow"))
def test_original_region_reaches_public_pipeline_and_native_continuations(mode):
    case, layout = build(16, (0,), control_region=mode,
        right_preconditioner="SpatialBasisJacobi@1")
    plan = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(plan.time,
        model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target="amr_system")
    assert "structured_original_observation" in code
    assert "HierarchyBarrierKind::nonlinear_solve" in code
    assert "hierarchy region continuation allocation" in code
    if mode.startswith("branch"):
        assert "hierarchy_branch_" in code
        assert code.count("_core->solve(") == 3
        assert "hierarchy branch differs across execution lane" in code
    else:
        assert "HierarchyBarrierKind::control" in code
        assert "std::weak_ptr<hierarchy_fold_" in code
        assert "retained_weak.lock()" in code
        assert "hierarchy fold count outside declared capacity" in code
        assert code.count("_core->solve(") == 2


def test_region_effect_check_does_not_qualify_unreviewed_operation():
    from types import SimpleNamespace
    from pops.codegen.program_emit_hierarchy_control import _check_control_effects

    with pytest.raises(NotImplementedError, match="checked lane-uniform failure contract"):
        _check_control_effects((SimpleNamespace(op="unqualified_mutation", attrs={}),))


def test_original_capture_cannot_escape_its_authoring_region():
    from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case
    case, field, program, current, request, block, forcing, frame = mixed_case()
    del case, request, frame
    escaped = []

    def branch(body):
        captured = body.value("regional capture", current.n, at=current.next.point)
        escaped.append(captured)
        return body.float64(1)

    program.branch(program.boolean(True), branch, lambda body: body.float64(0))
    with pytest.raises((ValueError, TypeError), match="region|scope|available"):
        field.bind_program_inputs(program=program, values={block[forcing]: escaped[0]},
            at=current.next.point, solver=field.default_program_solver())
