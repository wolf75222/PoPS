"""Final state consumers bind compile-time rank and the canonical FieldView authority."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HEADERS = {
    "face_flux": ROOT / "include/pops/numerics/spatial/primitives/face_flux.hpp",
    "wave_speed": ROOT / "include/pops/numerics/spatial/primitives/wave_speed.hpp",
    "embedded_boundary": ROOT / "include/pops/numerics/spatial/embedded_boundary/operator.hpp",
    "masked": ROOT / "include/pops/numerics/spatial/operators/masked_operator.hpp",
    "polar": ROOT / "include/pops/numerics/spatial/operators/polar_operator.hpp",
    "amr": ROOT / "include/pops/numerics/time/amr/reflux/amr_flux_helpers.hpp",
    "implicit": ROOT / "include/pops/numerics/time/integrators/implicit_stepper.hpp",
}


def _source(name: str) -> str:
    return HEADERS[name].read_text(encoding="utf-8")


def test_migrated_consumers_do_not_reintroduce_the_2d_storage_authority() -> None:
    forbidden = (
        r"\bBox2D\b",
        r"\bFab2D\b",
        r"\bConstArray4\b",
        r"\bArray4\b",
        r"\bxface_box\b",
        r"\byface_box\b",
        r"\.const_array\s*\(",
        r"\.array\s*\(",
    )
    violations: list[str] = []
    for name in HEADERS:
        source = _source(name)
        for pattern in forbidden:
            if re.search(pattern, source):
                violations.append(f"{name}: {pattern}")
    assert violations == []


def test_cartesian_consumers_keep_dimension_and_axis_static() -> None:
    face_flux = _source("face_flux")
    wave_speed = _source("wave_speed")
    masked = _source("masked")
    implicit = _source("implicit")

    assert "PreparedCartesianOperator<Dim" in face_flux
    assert "FaceField<Dim, MemorySpace>" in face_flux
    assert "FieldView<const Real, Dim>" in wave_speed
    assert "maximum_axis_wave_speed<Axis + 1, Dim>" in wave_speed
    assert "FieldView<const Real, Dim> state" in masked
    assert "for_each_face<Axis>" in masked
    assert "materialize_axes<Axis + 1, Variables>" in masked
    assert "PreparedImplicitSourceKernel<Dim, Model>" in implicit
    assert "operator()(const Index<Dim>& index)" in implicit
    assert "MultiFab<Dim, MemorySpace>" in implicit

    runtime_dimension_dispatch = re.compile(r"(?:if|switch)\s*\(\s*(?:Dim|dimension)\b")
    for name in ("face_flux", "wave_speed", "masked", "amr", "implicit"):
        assert runtime_dimension_dispatch.search(_source(name)) is None


def test_polar_is_explicitly_planar_but_embedded_boundary_uses_exact_rank() -> None:
    polar = _source("polar")
    embedded_boundary = _source("embedded_boundary")

    assert "PlanarPolarCoordinateMap" in polar
    assert "prepare_cartesian_operator<2" in polar
    assert "PreparedEmbeddedBoundaryOperator<" in embedded_boundary
    assert "PreparedEmbeddedBoundaryMetric<" in embedded_boundary
    assert "FieldView<const Real, Dim>" in embedded_boundary
    assert "PreparedMaskedCartesianOperator<Dim" in embedded_boundary
    assert "static_assert(Dim >= 1 && Dim <= 3)" in embedded_boundary
    assert "template <int Dim" not in polar
    assert "template <int Dim" in embedded_boundary


def test_retired_disc_and_fixed_face_authorities_cannot_return() -> None:
    assert not (ROOT / "include/pops/numerics/spatial/embedded_boundary/domain.hpp").exists()
    assert not (ROOT / "include/pops/numerics/elliptic/eb/cut_fraction.hpp").exists()
    for root in (ROOT / "include", ROOT / "src", ROOT / "python"):
        for path in root.rglob("*"):
            if path.suffix not in {".hpp", ".cpp", ".py", ".pyi"}:
                continue
            source = path.read_text(encoding="utf-8")
            assert "set_disc_domain" not in source, path
            assert "disc_mask" not in source, path


def test_amr_helpers_execute_only_authenticated_ranked_transfers() -> None:
    amr = _source("amr")
    assert "PreparedTransfer<Dim>" in amr
    assert "FieldView<const Real, Dim>" in amr
    assert "const Index<Dim>& index" in amr
    assert "PreparedTransferKernel<Dim>" in amr
    assert "prepare_linear_prolongation" in amr
    assert "prepare_average_down" in amr
    assert "prepare_fill_patch" in amr
    for kind in (
        "ConservativeRestriction",
        "LinearProlongation",
        "ConstantInjection",
        "CoarseFineGhostInterpolation",
    ):
        assert kind in amr
