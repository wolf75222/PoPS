"""ADC-749/757: one exact-ranked prepared transport-boundary authority remains."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOTS = (ROOT / "include/pops", ROOT / "src/runtime")

# These names denoted executable authorities parallel to PreparedHyperbolicBoundary<Dim>.
# Closure is a zero-occurrence invariant across production, not a count ledger.
DELETED_LEGACY_AUTHORITIES = (
    "AmrBoundaryFillAuthority",
    "make_amr_boundary_fill_authority",
    "transport_boundary_fill",
    "transport_bc",
    "wall_radial",
    "fill_ghosts_polar",
)


def _production_sources() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in PRODUCTION_ROOTS
            for path in root.rglob("*")
            if path.suffix in {".cpp", ".hpp"}
        )
    )


def _occurrences() -> dict[str, dict[str, int]]:
    patterns = {
        identifier: re.compile(r"\b%s\b" % re.escape(identifier))
        for identifier in DELETED_LEGACY_AUTHORITIES
    }
    counts = {identifier: {} for identifier in patterns}
    for path in _production_sources():
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        for identifier, pattern in patterns.items():
            count = len(pattern.findall(source))
            if count:
                counts[identifier][relative] = count
    return counts


def test_prepared_hyperbolic_boundary_is_the_only_native_transport_authority() -> None:
    authority = ROOT / "include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp"
    source = authority.read_text(encoding="utf-8")
    assert "template <int Dim>" in source
    assert "class PreparedHyperbolicBoundary" in source
    assert "MultiFab<Dim" in source
    assert "FaceField<Dim" in source
    assert "PreparedBoundaryPlan" not in source


def test_legacy_transport_boundary_authorities_are_deleted() -> None:
    occurrences = _occurrences()
    violations = [
        "%s: %s has %d occurrence(s)" % (identifier, path, count)
        for identifier, paths in occurrences.items()
        for path, count in paths.items()
    ]
    assert not violations, (
        "a deleted transport-boundary authority returned; lower the route to "
        "PreparedHyperbolicBoundary<Dim> instead:\n  " + "\n  ".join(violations)
    )


def test_native_transport_boundary_is_separate_from_the_polar_metric_specialization() -> None:
    polar_operator = (
        ROOT / "include/pops/numerics/spatial/operators/polar_operator.hpp"
    ).read_text(encoding="utf-8")
    prepared_boundary = (
        ROOT / "include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp"
    ).read_text(encoding="utf-8")
    system_registry = (ROOT / "include/pops/runtime/system/system_boundary_registry.hpp").read_text(
        encoding="utf-8"
    )
    amr_block = (
        ROOT / "include/pops/runtime/builders/compiled/generated_amr_system_block.hpp"
    ).read_text(encoding="utf-8")

    assert "PlanarPolarCoordinateMap" in polar_operator
    assert "prepare_cartesian_operator<2" in polar_operator
    assert "boundary_plan" not in polar_operator
    assert "template <int Dim>" in prepared_boundary
    assert "class PreparedHyperbolicBoundary" in prepared_boundary
    assert "apply_physical_flux_conditions" in prepared_boundary
    assert "HyperbolicBoundaryLaw::NoFlux" in prepared_boundary
    assert "ZeroBoundaryFaceFlux<Dim>" in prepared_boundary
    assert "apply_flux_axes_<Axis + 1>" in prepared_boundary
    assert "PreparedHyperbolicBoundary<Dim>" in system_registry
    assert "PreparedHyperbolicBoundary<Dim>" in amr_block
    for retired in (
        ROOT / "include/pops/mesh/boundary/prepared_boundary_plan.hpp",
        ROOT / "include/pops/mesh/boundary/boundary_component_executor.hpp",
        ROOT / "include/pops/runtime/builders/block/prepared_boundary_defaults.hpp",
    ):
        assert not retired.exists()


def test_resolved_transport_authority_accepts_only_executable_descriptors() -> None:
    """Numerical resolution, rather than a later compile/bind phase, owns acceptance."""
    source = ROOT / "python/pops/boundary/transport.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    authority = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResolvedTransportBoundarySet"
    )
    post_init = next(
        node for node in authority.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__post_init__"
    )
    calls = {
        node.func.attr
        for node in ast.walk(post_init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    }
    assert "_native_contract" in calls, (
        "ResolvedTransportBoundarySet must authenticate the complete executable boundary "
        "contract during numerical resolution; do not defer unsupported descriptors to compile"
    )


def test_boundary_provider_identity_cannot_erase_its_selected_law() -> None:
    """Every provider identity must retain the law selected by its public factory."""
    source = ROOT / "python/pops/mesh/boundaries/providers.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    provider = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BoundaryProvider"
    )
    annotations = {
        node.target.id
        for node in provider.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "kind" in annotations, "BoundaryProvider must retain one immutable typed law"
    canonical = next(
        node for node in provider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "canonical_identity"
    )
    keys = {
        node.value for node in ast.walk(canonical)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "provider_kind" in keys, (
        "BoundaryProvider canonical identity must authenticate its selected law"
    )


def test_post_riemann_flux_is_one_typed_outward_oriented_pipeline_stage() -> None:
    catalog = json.loads(
        (ROOT / "schemas/component_catalog.v2.json").read_text(encoding="utf-8")
    )
    interface = next(
        row for row in catalog["native_interface_abis"]
        if row["name"] == "boundary_flux"
    )
    assert interface["cpp_table"] == "PopsBoundaryFluxApiV1"
    assert interface["operations"] == ["transform_faces"]
    route = catalog["boundary_handle_native_routes"]["boundary_flux_provider"]
    assert route["interface"] == "boundary_flux"
    assert route["operation"] == "transform_faces"

    uniform = (ROOT / "include/pops/runtime/builders/compiled/generated_system_block.hpp").read_text(
        encoding="utf-8"
    )
    first_flux = uniform.index("spatial.materialize_face_fluxes")
    first_boundary = uniform.index("boundary->apply_physical_flux_conditions", first_flux)
    first_divergence = uniform.index("spatial.assemble_residual_from_face_fluxes", first_boundary)
    assert first_flux < first_boundary < first_divergence

    amr = (ROOT / "include/pops/runtime/builders/compiled/generated_amr_system_block.hpp").read_text(
        encoding="utf-8"
    )
    first_flux = amr.index("spatial.materialize_face_fluxes")
    first_boundary = amr.index("physical_boundary->apply_physical_flux_conditions", first_flux)
    first_divergence = amr.index("spatial.assemble_residual_from_face_fluxes", first_boundary)
    assert first_flux < first_boundary < first_divergence


def test_no_flux_is_a_builtin_face_law_of_the_same_prepared_pipeline() -> None:
    transport = (ROOT / "python/pops/boundary/transport.py").read_text(encoding="utf-8")
    hyperbolic = (
        ROOT / "include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp"
    ).read_text(encoding="utf-8")
    operator = (ROOT / "include/pops/runtime/builders/compiled/generated_system_block.hpp").read_text(
        encoding="utf-8"
    )

    assert 'condition_type: ClassVar[str] = "no_flux"' in transport
    assert 'if condition_type == "no_flux":' in transport
    assert "provider = LowLevelNoFlux(" in transport
    assert 'token == "no_flux"' in hyperbolic
    assert "HyperbolicBoundaryLaw::NoFlux" in hyperbolic
    assert "boundary->apply_physical_flux_conditions" in operator
    assert "spatial.assemble_residual_from_face_fluxes" in operator
