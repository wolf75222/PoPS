"""ADC-682 fences for the final PhysicalFlux/NumericalFlux/SpatialOperator split."""
import json
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
    assert "evaluate_axis_flux<Axis>" in combined
    assert "evaluate_numerical_flux_at" not in combined
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


def test_physical_flux_consumes_the_exact_pack_without_reconstructing_aux():
    header = _behavior(ROOT / "include/pops/numerics/fv/flux_interfaces.hpp")
    physical = header.split("struct PhysicalFluxView", 2)[2].split("template <class T>", 1)[0]
    assert "physical_providers" not in physical
    assert "Aux result" not in physical
    assert "const Aux" not in physical
    assert "trace.providers" in physical
    assert "left.providers" in physical
    assert "right.providers" in physical

    emitter = (ROOT / "python/pops/codegen/module_emit_brick.py").read_text(encoding="utf-8")
    assert 'aux_param = "const auto& a"' in emitter
    assert "_flux_provider_locals_lines" in emitter


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


def test_polar_riemann_dispatch_uses_model_capabilities_not_a_coordinate_allowlist():
    builder = _behavior(
        ROOT / "include/pops/runtime/builders/block/block_builder_polar.hpp"
    )
    catalog = json.loads(
        (ROOT / "schemas/component_catalog.v2.json").read_text(encoding="utf-8")
    )

    assert "case RiemannRouteId::kHllc" in builder
    assert "if constexpr (HasHLLCStructure<Model>)" in builder
    assert "case RiemannRouteId::kRoe" in builder
    assert "if constexpr (HasRoeDissipation<Model>)" in builder
    assert "no fallback" in builder
    riemann = next(
        family for family in catalog["route_families"] if family["name"] == "riemann"
    )
    routes = {route["token"]: route for route in riemann["routes"]}
    assert routes["hllc"]["metadata"]["polar_ok"] is True
    assert routes["roe"]["metadata"]["polar_ok"] is True


def test_fixed_riemann_recovery_route_is_wired_for_cartesian_uniform_and_amr_only():
    catalog = json.loads(
        (ROOT / "schemas/component_catalog.v2.json").read_text(encoding="utf-8")
    )
    riemann = next(
        family for family in catalog["route_families"] if family["name"] == "riemann"
    )
    route = next(
        row for row in riemann["routes"]
        if row["token"] == "roe_hll_rusanov_recovery"
    )
    uniform = _behavior(
        ROOT / "include/pops/runtime/builders/compiled/generated_system_block.hpp"
    )
    amr = _behavior(
        ROOT / "include/pops/runtime/builders/compiled/generated_amr_system_block.hpp"
    )
    policy = _behavior(ROOT / "include/pops/numerics/fv/numerical_flux.hpp")

    assert route["native_entry"] == (
        "pops::PreparedRiemannRecoveryPolicy<pops::RoeFlux,pops::HLLFlux,"
        "pops::RusanovFlux,pops::RejectRiemannRecovery>"
    )
    assert route["metadata"]["polar_ok"] is False
    assert "using RoeHllRusanovRecoveryPolicy" in policy
    assert "case RiemannRouteId::kRoeHllRusanovRecovery" in uniform
    assert "materialize_block<Dim, Model, Reconstruction, RoeHllRusanovRecoveryPolicy" in uniform
    assert "case RiemannRouteId::kRoeHllRusanovRecovery" in amr
    assert "materialize_system<Dim, Model, Reconstruction, RoeHllRusanovRecoveryPolicy" in amr
    assert "if (routes.wave_speed_cache)" in amr
