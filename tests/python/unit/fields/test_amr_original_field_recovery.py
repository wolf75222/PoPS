"""A recoverable Result retains its control frontier across hierarchy yields."""
import pytest
import pops

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_public_amr_original_field import build

MODES = ("capture_success", "capture_runtime", "capture_terminal", "capture_user",
         "iterate_success", "iterate_budget", "iterate_runtime")


@pytest.mark.parametrize("mode", MODES)
def test_original_recovery_uses_common_entry_and_exit_continuations(mode):
    case, layout = build(16, (0,), control_region=mode,
        right_preconditioner="SpatialBasisJacobi@1")
    plan = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(plan.time,
        model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target="amr_system")
    assert "ctx.push_hierarchy_failure_handler(" in code
    assert "ctx.close_hierarchy_failure_handlers(" in code
    assert "hierarchy_control_failure_words(rejected)" in code
    assert "HierarchyBarrierKind::nonlinear_solve" in code
    assert "HierarchyBarrierKind::control" in code
    if mode == "capture_user":
        assert "throw pops::runtime::program::hierarchy_control_failure(" in code
    if mode.startswith("iterate"):
        assert "std::weak_ptr<hierarchy_iterate_" in code
        assert "hierarchy iterate stop differs across execution lane" in code
