"""ADC-740: the production hyperbolic pipeline has one compile-time ND authority."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORE = (
    ROOT / "include/pops/numerics/spatial/nd/conservation_laws.hpp",
    ROOT / "include/pops/numerics/spatial/nd/reconstruction.hpp",
    ROOT / "include/pops/numerics/spatial/nd/face_field.hpp",
    ROOT / "include/pops/numerics/spatial/nd/finite_volume.hpp",
    ROOT / "include/pops/numerics/spatial/operators/cartesian_operator.hpp",
    ROOT / "include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp",
    ROOT / "include/pops/numerics/spatial_operator.hpp",
    ROOT / "include/pops/mesh/execution/for_each.hpp",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_canonical_hyperbolic_core_has_no_two_dimensional_storage_adapter() -> None:
    forbidden = (
        r"\bBox2D\b",
        r"\bFab2D\b",
        r"\b(?:Const)?Array4\b",
        r"\bxface_box\b",
        r"\byface_box\b",
        r"\bPreparedHyperbolicBoundary\s*<\s*2\s*>",
        r"static_assert\s*\(\s*Dim\s*==\s*2",
        r"\b(?:Fx|Fy)\b",
    )
    violations: list[str] = []
    for path in CORE:
        source = _source(path)
        for pattern in forbidden:
            if re.search(pattern, source):
                violations.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert violations == []


def test_dimension_and_axis_are_static_kernel_properties() -> None:
    reconstruction = _source(CORE[1])
    operator = _source(CORE[4])
    boundary = _source(CORE[5])

    assert "template <int Axis, int Orientation" in reconstruction
    assert "ReconstructionVariables Variables" in reconstruction
    assert "reconstruct_face_pair<Axis, Variables>" in operator
    assert "for_each_face<Axis>" in operator
    assert "materialize_axes<Axis + 1, Variables>" in operator
    assert "fill_axes_<Axis + 1>" in boundary
    assert "FieldView<Real, Dim>" in boundary
    assert "MultiFab<Dim, MemorySpace>" in boundary

    iteration = _source(CORE[7])
    assert "for_each_cell(const Box<Dim>&" in iteration
    assert "for_each_face(const Box<Dim>&" in iteration

    runtime_dimension_dispatch = re.compile(
        r"(?:if|switch)\s*\(\s*(?:Dim|dimension)\b"
    )
    assert all(not runtime_dimension_dispatch.search(_source(path)) for path in CORE)


def test_face_flux_and_residual_share_one_axis_indexed_field() -> None:
    face_field = _source(CORE[2])
    finite_volume = _source(CORE[3])
    operator = _source(CORE[4])

    assert "FieldView<T, Dim> axes[Dim]" in face_field
    assert "FaceField<Dim, MemorySpace>& output" in operator
    assert "FaceFieldView<const Real, Dim> integrated_fluxes" in operator
    assert "conservative_residual<N>" in operator
    assert "assemble_residual_from_face_fluxes" in operator
    assert "accumulate_divergence<0>" in finite_volume

    boundary = _source(CORE[5])
    assert "apply_physical_flux_conditions" in boundary
    assert "ZeroBoundaryFaceFlux" in boundary
    assert "FillAnalyticFace" in boundary
    assert "geometry.cell_center(ghost)" in boundary
    assert "geometry_from_box_origin_spacing" in boundary
    assert "analytic hyperbolic tables require a requalified ND coordinate provider" not in boundary
    assert "analytic hyperbolic boundary requires a requalified ND coordinate provider" in boundary
    assert "analytic ghost depth may not exceed the normal domain extent" in boundary


def test_hllc_and_roe_are_axis_generic_euler_capabilities() -> None:
    laws = _source(CORE[0])
    finite_volume = _source(CORE[3])
    assert "contact_speed" in laws
    assert "star_state" in laws
    assert "roe_dissipation" in laws
    assert "model.template contact_speed<Axis>" in finite_volume
    assert "model.template star_state<Axis>" in finite_volume
    assert "model.template roe_dissipation<Axis>" in finite_volume


def test_umbrella_and_manifest_do_not_promote_specialized_2d_fallbacks() -> None:
    umbrella = _source(CORE[6])
    manifest = _source(ROOT / "include/pops_headers.manifest")

    for fallback in (
        "masked_operator.hpp",
        "polar_operator.hpp",
        "embedded_boundary/operator.hpp",
        "primitives/face_flux.hpp",
        "prepared_cartesian_nd.hpp",
    ):
        assert fallback not in umbrella
    assert "api pops/numerics/spatial/nd/reconstruction.hpp" in manifest
    assert "api pops/numerics/spatial/operators/cartesian_operator.hpp" in manifest
    assert "prepared_cartesian_nd.hpp" not in manifest
    assert "sdk-support pops/numerics/spatial/operators/masked_operator.hpp" in manifest
    assert "sdk-support pops/numerics/spatial/operators/polar_operator.hpp" in manifest
    assert "sdk-support pops/numerics/spatial/embedded_boundary/operator.hpp" in manifest
