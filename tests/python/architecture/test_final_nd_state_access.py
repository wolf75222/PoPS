"""One ranked field-access authority feeds reconstruction and elliptic RHS assembly."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STATE_ACCESS = ROOT / "include/pops/numerics/spatial/primitives/state_access.hpp"
RECONSTRUCTION = ROOT / "include/pops/numerics/spatial/nd/reconstruction.hpp"
ELLIPTIC_RHS = ROOT / "include/pops/coupling/base/elliptic_rhs.hpp"
CPP_PROOF = ROOT / "tests/cpp/unit/numerics/test_prepared_cartesian_nd.cpp"
AUX_PROOF = ROOT / "tests/cpp/unit/physics/test_aux_single_source.cpp"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_state_and_aux_loaders_accept_only_ranked_field_views() -> None:
    source = _source(STATE_ACCESS)
    assert "#include <pops/mesh/storage/field_view.hpp>" in source
    assert re.search(
        r"load_state\(const FieldView<const Real, Dim>& field,\s*"
        r"const Index<Dim>& index\)",
        source,
    )
    assert re.search(
        r"load_aux\(const FieldView<const Real, Dim>& field,\s*"
        r"const Index<Dim>& index\)",
        source,
    )
    assert "field(index, component)" in source
    assert "field(index, kAuxNamedBase + k)" in source
    for forbidden in ("ConstArray4", "Array4", "Fab2D", "Box2D"):
        assert forbidden not in source


def test_reconstruction_consumes_the_shared_loader_without_a_private_copy() -> None:
    source = _source(RECONSTRUCTION)
    assert "#include <pops/numerics/spatial/primitives/state_access.hpp>" in source
    assert source.count("pops::load_state<Model>") >= 5
    assert not re.search(r"typename Model::State\s+load_state\s*\(", source)
    for forbidden in ("ConstArray4", "Array4", "Fab2D", "Box2D"):
        assert forbidden not in source


def test_every_elliptic_rhs_binds_dimension_memory_and_exact_field_identity() -> None:
    source = _source(ELLIPTIC_RHS)
    for declaration in (
        "struct SingleModelEllipticRhs",
        "struct TwoFieldChargeDensityRhs",
        "struct TwoBlockChargeDensityRhs",
        "struct ChargeDensityRhs",
    ):
        match = re.search(rf"{re.escape(declaration)}\s*\{{", source)
        assert match is not None
        position = match.start()
        template_start = source.rfind("template <int Dim", 0, position)
        assert position - 400 < template_start < position
        template_prefix = source[template_start:position]
        assert "MemorySpace" in template_prefix
    assert source.count("MultiFab<Dim, MemorySpace>") >= 12
    assert "FieldView<const Real, Dim>" in source
    assert "operator()(const Index<Dim>& index)" in source
    assert "source.layout() == destination.layout()" in source
    assert "source.distribution() == destination.distribution()" in source
    assert "source.local_rank() == destination.local_rank()" in source
    assert "source.shares_storage_with(destination)" in source
    for forbidden in ("ConstArray4", "Array4", "Fab2D", "Box2D"):
        assert forbidden not in source


def test_n_species_preflight_precedes_clear_and_all_accumulation() -> None:
    source = _source(ELLIPTIC_RHS)
    charge = source[source.index("struct ChargeDensityRhs") :]
    preflight = charge.index("detail::require_component_rhs_target")
    clear = charge.index("rhs.set_val(Real(0))")
    accumulation = charge.index("detail::add_scaled_component_unchecked")
    assert preflight < clear < accumulation


def test_cpp_proof_instantiates_ranked_access_and_rhs_paths() -> None:
    source = _source(CPP_PROOF)
    for rank in (1, 2, 3):
        assert f"check_ranked_state_access<{rank}>()" in source
    assert "SingleModelEllipticRhs<1" in source
    assert "TwoFieldChargeDensityRhs<2>" in source
    assert "ChargeDensityRhs<3>" in source
    assert "wrong_distribution" in source
    assert "wrong_layout" in source

    auxiliary = _source(AUX_PROOF)
    for rank in (1, 2, 3):
        assert f"check_device_and_host_marshaling<{rank}>()" in auxiliary
        assert f"check_base_width_ignores_extra_fields<{rank}>()" in auxiliary
    for forbidden in ("ConstArray4", "Array4", "Fab2D", "Box2D"):
        assert forbidden not in auxiliary
