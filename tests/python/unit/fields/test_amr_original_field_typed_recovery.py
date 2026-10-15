"""Closed error payloads survive hierarchy suspension without scalar erasure."""
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_source_map import source_map
from tests.python.integration.runtime.test_public_amr_original_field import build

MODES = ("capture_typed", "iterate_typed_success", "iterate_typed_error", "iterate_typed_budget")


@pytest.mark.parametrize("mode", MODES)
def test_typed_recovery_preserves_array_payload_and_authored_iteration(mode):
    case, layout = build(16, (0,), control_region=mode,
        right_preconditioner="SpatialBasisJacobi@1")
    plan = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(plan.time,
        model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target="amr_system")
    assert "ctx.raise_hierarchy_typed_failure(" in code
    assert "ctx.hierarchy_failure_value(" in code
    assert "hierarchy_typed_failure recovered;" in code
    assert "std::shared_ptr<const void> recovered_storage;" in code
    assert "continuation materialization lifetime retention failed" in code
    assert "Program numerical Field input authority" in code
    assert "pops.hierarchy.typed-error@1" in code
    assert "HierarchyBarrierKind::nonlinear_solve" in code
    assert "hierarchy_control_failure_words(rejected)" in code
    assert source_map(code)["complete"]
    if mode.startswith("iterate"):
        assert "std::weak_ptr<hierarchy_typed_iterate_" in code
        assert "hierarchy iterate stop differs across execution lane" in code
        assert "hierarchy iteration Result differs across execution lane" in code
