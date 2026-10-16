"""A prepared scientific field persists without borrowing a State owner."""
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_source_map import source_map
from examples.api_v02.continuation_capture_amr import build_continuation_capture_amr


def test_field_problem_observation_has_a_retained_provider_and_full_capture_path():
    spec = build_continuation_capture_amr(field_payload=True, independent_field=True)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    model = build_program_model_graph(plan)
    source = emit_cpp_program(plan.time, model=model, target="amr_system")
    assert "configure_hierarchy_continuation_field_provider(" in source
    assert "capture_hierarchy_field_observation(" in source
    assert "ctx.continuation_preflight_field(" in source
    assert "field_problem" in source and "ctx.raise_hierarchy_typed_failure(" in source
    assert "ctx.prepare_level_continuation_sync(" in source
    assert source_map(source)["complete"]
