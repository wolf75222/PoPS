"""Authored two-trace equations and both AMR towers must reach native codegen."""
import pops
import pytest
from examples.api_v02.method_relaxed_multiblock import build_t10_multiblock, transport_trace_flux
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize("ratio", (1, 2))
def test_two_trace_affine_program_covers_both_owned_towers(ratio):
    flux = transport_trace_flux()
    spec = build_t10_multiblock(flux, temporal_ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout, components=(flux,))
    cpp = emit_cpp_program(plan.time, target="amr_system",
                          model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "ctx.begin_affine_finalization(dt, {0, 1})" in cpp
    assert "ctx.accept_increment" in cpp
    assert len(plan.blocks) == 2
    assert "PyObject" not in cpp
