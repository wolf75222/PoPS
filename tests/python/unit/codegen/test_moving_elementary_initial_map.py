"""Initial geometry proof must recognize exact elementary identities."""
import pops
from examples.api_v02.models_M22_M25_M28 import build_m28
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


def test_oscillating_map_keeps_reference_initial_geometry():
    spec=build_m28()
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "physical_time" in cpp and "Kokkos::sin" in cpp
    assert "face.swept_volume/time.duration" in cpp
    assert "ctx.commit_moving_interval" in cpp
