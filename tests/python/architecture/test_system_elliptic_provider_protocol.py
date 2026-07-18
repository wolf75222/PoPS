"""The System elliptic extension seam stays request-driven and open-ended."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_FIELD_SOLVER = ROOT / "include/pops/runtime/system/system_field_solver.hpp"
ELLIPTIC_BACKEND = ROOT / "include/pops/runtime/system/system_elliptic_backend.hpp"
SYSTEM_INSTALL = ROOT / "src/runtime/system/system_install.cpp"


def test_system_elliptic_core_has_no_closed_capability_vocabulary() -> None:
    source = SYSTEM_FIELD_SOLVER.read_text(encoding="utf-8")
    support = ELLIPTIC_BACKEND.read_text(encoding="utf-8")
    forbidden = (
        "EllipticBackendCapability",
        "EllipticBackendCapabilities",
        "required_capabilities_",
        "require_capabilities",
        "capability_name",
    )
    assert not {token for token in forbidden if token in source or token in support}


def test_provider_support_decides_from_the_complete_typed_request() -> None:
    source = SYSTEM_FIELD_SOLVER.read_text(encoding="utf-8")
    provider_protocol = source.split("class EllipticBackendProvider", 1)[1].split(
        "class GeometricMgBackendProvider", 1
    )[0]
    assert "capability_contracts()" in provider_protocol
    assert "supports(" in provider_protocol
    assert "const EllipticBackendBuildRequest& request" in provider_protocol
    assert "supports(request) is the sole compatibility authority" in provider_protocol


def test_generic_backend_protocol_has_no_feature_specific_configuration_setters() -> None:
    source = SYSTEM_FIELD_SOLVER.read_text(encoding="utf-8")
    backend_protocol = source.split("class NamedFieldBackend", 1)[1].split(
        "/// Canonical component", 1
    )[0]
    forbidden = (
        "capabilities()",
        "set_scalar_coefficient",
        "set_diagonal_tensor_coefficient",
        "set_reaction_coefficient",
        "set_dynamic_boundary",
        "set_nonlinear_boundary",
    )
    assert not {token for token in forbidden if token in backend_protocol}


def test_cartesian_runtime_dispatches_only_through_the_provider_registry() -> None:
    source = SYSTEM_FIELD_SOLVER.read_text(encoding="utf-8")
    ensure_elliptic = source.split("void ensure_elliptic()", 1)[1].split(
        "MultiFab& ell_rhs()", 1
    )[0]
    assert "elliptic_registry_.prepare(p_solver" in ensure_elliptic
    assert "p_solver ==" not in ensure_elliptic
    assert "p_solver !=" not in ensure_elliptic


def test_set_poisson_does_not_enumerate_registered_backend_routes() -> None:
    source = SYSTEM_INSTALL.read_text(encoding="utf-8")
    set_poisson = source.split("void System::set_poisson", 1)[1].split(
        "System::register_configured_field_solver_provider", 1
    )[0]
    assert 'solver != "fft"' not in set_poisson
    assert 'solver != "fft_spectral"' not in set_poisson
    assert "has_elliptic_provider(solver)" in set_poisson
