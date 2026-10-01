"""Actual stage emission regression, without JIT/native execution."""
import re

import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.evolved_stage_mms import build


def emit(cells=8, width=2, candidate=True):
    case, layout, _ = build(cells, width, .01, candidate_diffusion=candidate)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    return resolved.time, cpp


@pytest.mark.parametrize("width,candidate", ((1, False), (2, True)))
def test_issued_duration_uses_actual_point_dt_and_operation_authority(width, candidate):
    program, cpp = emit(width=width, candidate=candidate)
    assert "ctx.step_dt()" not in cpp
    for node in program._values:
        if node.op in ("solve_spatial_field", "field_problem_coefficients"):
            assert "ctx.boundary_evaluation_point(%d).dt" % node.id in cpp
    assert "original stage frame/point/attempt authority changed" in cpp
    assert "original_field_residual_recheck_failed" in cpp


def test_each_partition_has_private_capture_views_and_visible_profile_timer():
    program, cpp = emit()
    publications = [node for node in program._values if node.op == "field_evolved_state"]
    assert len(publications) == 2
    for node in publications:
        start = cpp.index("const auto _pt%d =" % node.id)
        end = cpp.index('ctx.profile_record("node:evolved_accumulation", _pt%d);' % node.id)
        body = cpp[start:end]
        assert body.count("const auto output = evolved_output.fab(patch).view();") == 1
        for index in range(len(node.inputs)-1):
            assert len(re.findall(r"const auto capture%d =" % index, body)) == 1
        assert body.index("const auto capture0 =") < body.index("[=] POPS_HD")
        assert body.count("const auto _pt") == 1
