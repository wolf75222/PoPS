"""ADC-681: a prepared Reflux component executes without owning AMR authority."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PATCH_RANGE = (
    ROOT / "include" / "pops" / "numerics" / "time" / "amr" / "levels"
    / "amr_patch_range.hpp"
)
SUBCYCLING = (
    ROOT / "include" / "pops" / "numerics" / "time" / "amr" / "levels"
    / "amr_subcycling.hpp"
)
PROVIDERS = (
    ROOT / "include" / "pops" / "runtime" / "amr"
    / "prepared_component_providers.hpp"
)
AMR_RUNTIME = ROOT / "include" / "pops" / "runtime" / "amr" / "amr_runtime.hpp"
PROGRAM_REFLUX = (
    ROOT / "include" / "pops" / "runtime" / "amr" / "amr_program_reflux.hpp"
)
PROGRAM_CONTEXT = (
    ROOT / "include" / "pops" / "runtime" / "program" / "amr_program_context.hpp"
)
AMR_SYSTEM = ROOT / "src" / "runtime" / "amr" / "amr_system.cpp"
AMR_BINDING = ROOT / "python" / "bindings" / "core" / "init" / "init_amr.cpp"
RUNTIME_AUTHORITIES = ROOT / "python" / "pops" / "runtime" / "_runtime_authorities.py"
AMR_PROVIDER_PROTOCOLS = ROOT / "python" / "pops" / "amr" / "providers.py"


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_transition_executes_local_kernel_before_pops_collective_publication() -> None:
    source = SUBCYCLING.read_text()
    transition = _between(
        source,
        "class PreparedAmrProgramRefluxTransition",
        "class PreparedAmrProgramRefluxPlan",
    )
    assert "PreparedAmrRefluxLocalKernel local_kernel_" in transition
    assert "workspace.poison();" in transition
    assert "local_kernel_(PreparedAmrRefluxLocalRequest{" in transition
    assert "workspace.all_finite()" in transition
    assert "all_reduce_or_inplace(&preflight_consensus" in transition
    assert "prepared Reflux provider differs between communicator ranks" in transition
    assert transition.index("local_kernel_(PreparedAmrRefluxLocalRequest{") < (
        transition.index("route_prepared_reflux_correction_")
    )
    assert transition.index("all_reduce_max(local_failure") < transition.index(
        "correction_.gather(communicator);"
    )
    assert "route_reflux_integrated_pair_prevalidated_" in transition
    assert "apply_reflux_interface_batch" not in transition


def test_component_adapter_is_host_local_noncollective_and_has_no_topology() -> None:
    source = PROVIDERS.read_text()
    adapter = _between(
        source,
        "class PreparedRefluxComponent final",
        "/// External Clustering ABI contract",
    )
    assert "without_collective_authority()" in adapter
    assert "collective_contract() const noexcept" in adapter
    assert "POPS_MEMORY_SPACE_HOST_V1" in adapter
    assert "apply_reflux_interface_batch" in adapter
    assert "POPS_NATIVE_INTERFACE_REFLUX_V1" in adapter
    assert "FluxRegister" not in adapter
    assert "CoverageMask" not in adapter
    assert "all_reduce" not in adapter


def test_pops_maps_validated_faces_through_coverage_and_periodicity() -> None:
    source = PATCH_RANGE.read_text()
    kernel = _between(
        source,
        "struct RoutePreparedRefluxCorrectionKernel",
        "}  // namespace detail",
    )
    assert "canonicalize" in kernel
    assert "coverage.covered" in kernel
    assert "correction.add" in kernel
    assert "faces.x_low[index]" in kernel
    assert "faces.x_high[index]" in kernel
    assert "faces.y_low[index]" in kernel
    assert "faces.y_high[index]" in kernel


def test_runtime_installation_reprepares_transitions_and_routes_logical_time() -> None:
    runtime = AMR_RUNTIME.read_text()
    install = _between(
        runtime,
        "void install_external_reflux(",
        "/// Inject the current Program evaluation coordinate",
    )
    assert "external_reflux_ = std::move(provider);" in install
    assert "require_prepared_provider_collective_consensus" in install
    assert "rematerialize_persistent_topology_resources_" in install
    rematerialize = _between(
        runtime,
        "void rematerialize_persistent_topology_resources_(",
        "void record_topology_replacement_()",
    )
    assert "provider->apply(request);" in rematerialize
    assert "external_reflux_kernel, block.state_identity" in rematerialize

    route = PROGRAM_REFLUX.read_text()
    assert "const amr::ClockStamp& logical_time" in route
    assert "&logical_time" in route
    context = PROGRAM_CONTEXT.read_text()
    assert (
        "route_reflux_program(*eng_, sb, child, coarse_role, fine_role, sync_clock)"
        in context
    )


def test_reflux_uses_the_public_normalized_amr_provider_resolution() -> None:
    system = AMR_SYSTEM.read_text()
    binding = AMR_BINDING.read_text()
    authorities = RUNTIME_AUTHORITIES.read_text()
    protocols = AMR_PROVIDER_PROTOCOLS.read_text()
    assert "install_amr_reflux_component(" in system
    assert "runtime->install_external_reflux(amr_reflux_component_);" in system
    assert "if (amr_reflux_component_)" not in _between(
        system,
        "runtime->install_external_tagger(amr_tagger_component_);",
        "if (!boundary_plans_.empty())",
    )
    assert '"_install_amr_reflux_component"' in binding
    assert 'component_installer="_install_amr_reflux_component"' in protocols
    assert '"_install_amr_reflux_component"' not in authorities
    assert 'tuple(providers) != ("clustering", "tagger", "reflux")' in authorities
