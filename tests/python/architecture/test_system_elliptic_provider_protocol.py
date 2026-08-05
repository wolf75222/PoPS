"""The exact-ranked System field protocol has one open extension seam."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LEGACY_FIELD_SOLVER = ROOT / "include/pops/runtime/system/system_field_solver.hpp"
LEGACY_BACKEND_METRICS = ROOT / "include/pops/runtime/system/system_elliptic_backend.hpp"
HEADER_MANIFEST = ROOT / "include/pops_headers.manifest"
EXACT_BACKEND = ROOT / "include/pops/runtime/system/exact_field_solver_backend.hpp"
PREPARED_COMPONENT = ROOT / "include/pops/runtime/system/prepared_field_solver_component.hpp"
FIELD_PLANS = ROOT / "src/runtime/system/system_field_plans.cpp"
SYSTEM_INSTALL = ROOT / "src/runtime/system/system_install.cpp"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_parallel_2d_system_field_engine_is_retired() -> None:
    assert not LEGACY_FIELD_SOLVER.exists()
    assert not LEGACY_BACKEND_METRICS.exists()
    manifest = _read(HEADER_MANIFEST)
    assert "system_field_solver.hpp" not in manifest
    assert "system_elliptic_backend.hpp" not in manifest


def test_builtin_and_component_backends_share_one_ranked_protocol() -> None:
    source = _read(EXACT_BACKEND)
    assert "template <int Dim>" in source
    assert "class ExactFieldSolverBackend" in source
    assert "class CartesianFieldSolverBackend final" in source
    assert "class ComponentFieldSolverBackend final" in source
    for operation in (
        "field_type& rhs()",
        "field_type& candidate()",
        "SolveReport solve(const field_type& warm_start)",
        "std::string_view provider_identity()",
        "topology_report()",
    ):
        assert operation in source
    for legacy in ("Box2D", "Fab2D", "DistributionMapping", "Array4"):
        assert legacy not in source


def test_external_component_is_prepared_for_the_exact_dimension() -> None:
    source = _read(PREPARED_COMPONENT)
    assert "template <int Dim>" in source
    assert "class PreparedFieldSolverComponent" in source
    assert "Geometry<Dim>" in source
    assert "MultiFab<Dim>" in source
    assert "std::array<bool, Dim>" in source
    for legacy in ("Box2D", "Fab2D", "DistributionMapping"):
        assert legacy not in source


def test_install_resolves_configured_and_component_providers_by_qualified_slot() -> None:
    plans = _read(FIELD_PLANS)
    install = _read(SYSTEM_INSTALL)
    assert "register_configured_field_solver_provider" in plans
    assert "register_field_solver_provider" in plans
    assert "configured_field_solver_providers_" in install
    assert "component_field_solver_providers_" in install
    assert "backend_provider_route" in install
    assert "SystemFieldSolver" not in plans
    assert "SystemFieldSolver" not in install
