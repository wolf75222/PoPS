"""ADC-682 fences for the final PhysicalFlux/NumericalFlux/SpatialOperator split."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]


def _behavior(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return re.sub(r"//.*?$|/\*.*?\*/", "", source,
                  flags=re.MULTILINE | re.DOTALL)


def test_numerical_flux_has_only_the_two_trace_narrow_interface():
    header = _behavior(ROOT / "include/pops/numerics/fv/numerical_flux.hpp")
    assert "const Model&" not in header
    assert "const Aux&" not in header
    assert "physical, left, right, face" in header
    assert "FluxEvaluation<typename Physical::State>" in header


def test_spatial_operators_own_geometric_measure_exactly_once():
    paths = (
        ROOT / "include/pops/numerics/spatial/operators/cartesian_operator.hpp",
        ROOT / "include/pops/numerics/spatial/operators/polar_operator.hpp",
        ROOT / "include/pops/numerics/spatial/operators/masked_operator.hpp",
        ROOT / "include/pops/numerics/spatial/embedded_boundary/operator.hpp",
        ROOT / "include/pops/numerics/spatial/primitives/face_flux.hpp",
    )
    combined = "\n".join(_behavior(path) for path in paths)
    assert "nflux(model" not in combined
    assert "apply_face_measure" in combined
    assert ".density" not in combined
    assert "checked_density()" in combined
    assert "evaluate_numerical_flux_at" in combined
    assert "rf * F[" not in combined
    assert "alpha * F[" not in combined


def test_bound_native_flux_pack_is_exact_and_does_not_store_global_aux():
    header = _behavior(ROOT / "include/pops/numerics/fv/flux_interfaces.hpp")
    bound = header.split("class BoundFluxProviders", 1)[1].split("};", 1)[0]
    assert "FluxProviderValues<Model> values_" in bound
    assert "Aux values_" not in bound
    assert "bind_flux_providers(const Aux" not in header
    assert "bind_flux_providers_at" in header
    assert "FluxDensity<State> checked_density() const" in header


def test_generated_flux_pack_metadata_controls_native_storage_reads():
    header = _behavior(ROOT / "include/pops/numerics/fv/flux_interfaces.hpp")
    assert "qualified_flux_provider_requirements_valid" in header
    assert "qualified_flux_provider_storage_slot<Model, Indices>" in header
    assert "std::make_index_sequence<count>" in header
    assert "generated physical flux provider requirements are invalid" in header


def test_provider_selection_is_qualified_and_never_returns_a_neutral_value():
    source = (ROOT / "python/pops/model/provider_pack.py").read_text(encoding="utf-8")
    assert "def select(" in source
    assert "def select_spaces(" in source
    assert "owner_qid" in source
    assert "return 0" not in source
    assert "return 0.0" not in source


def test_hllc_rejects_nonfinite_provider_stages_before_publication():
    policy = _behavior(ROOT / "include/pops/numerics/fv/numerical_flux.hpp")
    interface = _behavior(ROOT / "include/pops/numerics/fv/flux_interfaces.hpp")
    hllc = policy.split("struct HLLCFlux", 1)[1].split(
        "concept RoePhysicalFlux", 1
    )[0]
    causes = (
        "kHllcNonFinitePhysicalFlux",
        "kHllcNonFinitePressure",
        "kHllcNonFiniteContact",
        "kHllcNonFiniteStarState",
        "kHllcNonFiniteFlux",
    )

    for cause in causes:
        assert cause in interface
        assert cause in hllc
    assert hllc.count("detail::finite_state") >= 6
    assert hllc.count("Kokkos::isfinite") >= 2


def test_capability_driven_riemann_has_no_euler_specific_production_authority():
    production_roots = (ROOT / "include/pops", ROOT / "src", ROOT / "python/pops")
    sources = (
        path
        for root in production_roots
        for path in root.rglob("*")
        if path.suffix in {".hpp", ".cpp", ".py"}
    )
    production = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    for retired_authority in (
        "EulerHLLCFlux2D",
        "EulerRoeFlux2D",
        "euler_hllc",
        "euler_roe",
    ):
        assert retired_authority not in production
