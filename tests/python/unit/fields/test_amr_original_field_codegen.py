"""Admission of the original physical body through the real composite Program route."""
import pytest

from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy)
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.math import ValueExpr
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import every
from pops.params import RuntimeParam
from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case, finish


@pytest.mark.parametrize("width,order", [(2, (0, 1)), (2, (1, 0)), (3, (2, 0, 1))])
def test_original_body_amr_uses_one_composite_carrier(width, order, tmp_path):
    case, field, program, current, request, block, forcing, frame = mixed_case(order, width=width)
    transfer = AMRTransfer()
    transfer.state(block[forcing], StateTransfer())
    threshold = case.param(RuntimeParam("refinement threshold", default=.5))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(16, 12), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[forcing])["f0"] > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    code = finish(case, field, program, current, request, block, forcing, frame, layout=layout, target="amr_system")
    assert "PreparedAmrFieldResidual" in code
    assert "original_hierarchy_field_authority" in code
    assert "stage_original_field_candidate_collectively" in code
    assert "with_program_attempt_level" in code
    assert "nonfinite_original_field_residual" in code
    assert "original_amr_field_residual" in code
    assert "block_inverse" not in code and "condensed" not in code
    (tmp_path / "original_amr_program.cpp").write_text(code)


@pytest.mark.parametrize("cells,order,seed,guarded", [
    (16, (0,), False, False), (16, (0, 1, 2), False, False),
    (32, (2, 0, 1), True, False), (16, (2, 0, 1), True, True)])
def test_closed_public_original_amr_witness_is_source_admissible(cells, order, seed, guarded, tmp_path):
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_emit_kernels import _prepared_native_components
    from tests.python.integration.runtime.test_public_amr_original_field import build

    case, layout = build(cells, order, seed=seed, guarded=guarded)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert "stage_original_field_candidate_collectively" in code
    assert "publish_staged_field_components" in code
    assert "original_hierarchy_field_authority" in code
    assert code.count("_core->solve(") == 1
    components = _prepared_native_components(resolved.time, target="amr_system")
    assert any("pops/runtime/program/prepared_amr_field_residual.hpp" in component.entry_headers for component in components)
    assert not any("pops/runtime/program/prepared_amr_field_residual.hpp" in component.entry_headers
                   for component in _prepared_native_components(resolved.time))
    if seed:
        assert "auto& seed = ctx.hierarchy_field_scratch(" in code
        assert "0, false)" in code
    if guarded:
        assert "response_limit" in code and "StepAttemptRejected" in code
    (tmp_path / "public_original_amr_program.cpp").write_text(code)


def test_original_amr_public_witness_refuses_subcycling_before_codegen():
    import pops
    from tests.python.integration.runtime.test_public_amr_original_field import build

    case, layout = build(16, (0,))
    with pytest.raises(ValueError, match="explicit synchronous AMR execution"):
        pops.resolve(pops.validate(case), layout=AMR(grid=layout.grid, hierarchy=layout.hierarchy, tagging=layout.tagging,
            regrid=layout.regrid, transfer=layout.transfer, execution=AMRExecution.subcycled()))
