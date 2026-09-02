"""Source-level guard for the AMR scalar boundary/stencil cold-binding contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OPERATIONS = (
    ROOT
    / "include/pops/runtime/program/detail/program_execution_services_amr_spatial_operations.hpp"
)
SERVICES = ROOT / "include/pops/runtime/program/program_execution_services.hpp"
CONDENSED_EMITTER = ROOT / "python/pops/codegen/program_emit_condensed.py"


def test_amr_scalar_boundary_stencils_fail_closed_without_a_cold_bound_session() -> None:
    operations = OPERATIONS.read_text(encoding="utf-8")
    services = SERVICES.read_text(encoding="utf-8")

    assert "bind_mesh_boundary_session(" in operations
    assert "AMR scalar boundary binding is cold-only" in operations
    assert "AMR boundary fill requires a cold-bound PreparedScalarBoundarySession" in operations
    assert "AMR Program Laplacian requires a cold-bound PreparedScalarBoundarySession" in operations
    assert "AMR Program gradient requires a cold-bound PreparedScalarBoundarySession" in operations
    assert "AMR Program divergence requires a cold-bound PreparedScalarBoundarySession" in operations
    assert "scalar_boundary_session_type session(geometry()" not in operations
    assert "auto boundary = prepare_mesh_boundary_session(flux" not in operations
    assert "bind_mesh_boundary_session(" in services
    assert "AMR cold-bound scalar boundary session authority" in services


def test_condensed_amr_emission_captures_cold_bound_scalar_sessions() -> None:
    emitter = CONDENSED_EMITTER.read_text(encoding="utf-8")

    assert '"auto %s = ctx.bind_mesh_boundary_session(*%s, ctx.prepared_execution_lane());"' in emitter
    assert '"%s->fill(*%s);" % (tensor_boundary, tensor)' in emitter
    assert '"  %s->fill(%s);" % (scalar_boundary, dest)' in emitter
    assert '"ctx.laplacian(%s, %s, *%s);"' in emitter
    assert "subslot=3" in emitter
