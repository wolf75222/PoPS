"""Original equations cross real gather/solve/publication continuations."""
import pytest

from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy)
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.math import ValueExpr
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.params import RuntimeParam
from pops.time import FailRun, every
from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case, finish


@pytest.mark.parametrize("reuse_request", (True, False))
def test_multiple_original_solves_have_distinct_barriers_and_workspaces(reuse_request):
    case, field, program, current, request, block, forcing, frame = mixed_case()
    first = program.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
    observed = field.observe(first)
    handle = field._field_registry.resolved_registration(field).operator.unknowns[0]
    program.record_scalar("first-observation", program.sum(observed[handle]))
    if not reuse_request:
        request = field.bind_program_inputs(program=program, values={block[forcing]: current.n},
            at=current.next.point, solver=field.default_program_solver())
    transfer = AMRTransfer()
    transfer.state(block[forcing], StateTransfer())
    threshold = case.param(RuntimeParam("refinement threshold", default=.5))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(16, 12), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[forcing])["f0"] > case.value(threshold)), Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    code = finish(case, field, program, current, request, block, forcing, frame,
                  layout=layout, target="amr_system")
    solves = [value for value in program._values if value.op == "solve_spatial_field"]
    assert len(solves) == 2
    assert code.count("HierarchyBarrierKind::nonlinear_solve") == 2
    assert code.count("_core->solve(") == 2
    assert all("ctx.retain_hierarchy_field_solver(%d)" % value.id in code for value in solves)
    if reuse_request:
        assert "auto& amr_original_field_%d_coefficients = ctx.hierarchy_field_assembly(%d" % (solves[1].id, solves[1].id) in code
    else:
        assert "auto& amr_original_field_%d_coefficients =" % solves[1].id not in code


@pytest.mark.parametrize("reuse_request", (True, False))
def test_closed_sequence_receiver_reaches_the_public_pipeline(reuse_request):
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.integration.runtime.test_public_amr_original_field import build

    case, layout = build(16, (0,), second_solve=True, reuse_request=reuse_request,
                         right_preconditioner="SpatialBasisJacobi@1")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(resolved.time,
        model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert code.count("HierarchyBarrierKind::nonlinear_solve") == 2
    assert code.count("_core->solve(") == 2
    assert "HierarchyBarrierKind::field_publication" in code
    assert "with_program_attempt_level_or_current" in code
