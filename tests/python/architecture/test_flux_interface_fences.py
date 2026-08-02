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
