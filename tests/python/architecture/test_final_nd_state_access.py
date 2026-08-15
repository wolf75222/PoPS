"""One ranked field-access authority feeds state, provider, and elliptic consumers."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STATE_ACCESS = ROOT / "include/pops/numerics/spatial/primitives/state_access.hpp"
RECONSTRUCTION = ROOT / "include/pops/numerics/spatial/nd/reconstruction.hpp"
ELLIPTIC_RHS = ROOT / "include/pops/coupling/base/elliptic_rhs.hpp"
CPP_PROOF = ROOT / "tests/cpp/unit/numerics/test_prepared_cartesian_nd.cpp"
AUX_PROOF = ROOT / "tests/cpp/unit/physics/test_aux_single_source.cpp"
ELLIPTIC_PROOF = ROOT / "tests/cpp/integration/runtime/test_system_abstraction.cpp"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_state_and_provider_loaders_accept_only_ranked_field_views() -> None:
    source = _source(STATE_ACCESS)
    assert "#include <pops/mesh/storage/field_view.hpp>" in source
    assert re.search(
        r"load_state\(const FieldView<const Real, Dim>& field,\s*"
        r"const Index<Dim>& index\)",
        source,
    )
    assert re.search(
        r"load_provider_values\(const Storage& storage,\s*"
        r"const Index<Dim>& index\)",
        source,
    )
    assert "struct ProviderStorageView" in source
    assert "std::array<FieldView<const Real, Dim>" in source
    assert "std::array<int" in source
    assert "field(index, component)" in source
    assert "storage(index, slot)" in source
    assert "kAuxNamedBase" not in source
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


def test_cpp_proof_instantiates_ranked_state_and_provider_access() -> None:
    source = _source(CPP_PROOF)
    for rank in (1, 2, 3):
        assert f"check_ranked_state_access<{rank}>()" in source
    assert "ProviderStorageView<Dim, Dim>" in source
    assert "mapped.storage_components[slot] = ncomp - 1 - slot" in source
    assert "load_provider_values<Dim>(mapped, sample)" in source

    auxiliary = _source(AUX_PROOF)
    for rank in (1, 2, 3):
        assert f"check_magnetic_factory_dispatch<{rank}>()" in auxiliary
    assert "ProviderValues, QualifiedConsumersDoNotAliasBySpelling" in auxiliary
    assert "ProviderValues, GradientSourcesUseExplicitPermutedSlotsInEveryRank" in auxiliary
    for forbidden in ("ConstArray4", "Array4", "Fab2D", "Box2D"):
        assert forbidden not in auxiliary


def test_native_elliptic_rhs_proof_uses_the_ranked_assembler() -> None:
    source = _source(ELLIPTIC_PROOF)
    for rank in (1, 2, 3):
        assert f"check_ranked_elliptic_rhs<{rank}>()" in source
    assert "TwoFieldChargeDensityRhs<Dim> charge" in source
    assert "ProviderValues<0>" in source
    assert "charge(Ue, Ui, rhs)" in source
    assert "charge_density_rhs" in source
