"""Source correspondence for the AMR face adapter, not a native receipt."""
import pops
from examples.api_v02.method_flux_correction_amr import build_t12_amr
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_amr import _flux_expression_budgets
from pops.codegen.program_models import ProgramModelGraph


def test_authored_stencil_has_exact_amr_storage_and_two_distinct_bases():
    spec = build_t12_amr()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert _flux_expression_budgets(resolved.time) == ((2, 1),)
    source = emit_cpp_program(resolved.time, target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.attach_generated_face_basis(") == 2
    assert source.count("ctx.prepare_generated_state(") == 2
    assert "ctx.stage_exchange_batch(" not in source
    assert "ctx.install_flux_temporal_families(" in source
    assert "pops::Real cse" in source
