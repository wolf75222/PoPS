"""ADC-738: final AMR providers retain one prepared compile-time rank."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ROOT / "include"
BINDINGS = ROOT / "python" / "bindings" / "core" / "init" / "init_amr.cpp"

RANKED_PROVIDERS = (
    "pops/numerics/time/amr/reflux/amr_reflux_mf.hpp",
    "pops/runtime/amr/bootstrap_transfer_builtins.hpp",
    "pops/runtime/amr/composite_reduction.hpp",
    "pops/runtime/amr/prepared_component_providers.hpp",
    "pops/runtime/builders/block/block_builder.hpp",
    "pops/runtime/builders/compiled/amr_dsl_block.hpp",
)
SPECIALIZED_FAC = "pops/numerics/elliptic/mg/composite_fac_poisson.hpp"
RANKED_TAGGING_EXECUTION = (
    "pops/runtime/amr/prepared_tagging_execution.hpp",
    "pops/runtime/amr/persistent_tagging_state.hpp",
)
PREPARED_COMPONENT_UMBRELLA = "pops/runtime/amr/prepared_component_providers.hpp"
SEMANTIC_AUTHORITIES = {
    PREPARED_COMPONENT_UMBRELLA: "compatibility umbrella",
    "pops/runtime/amr/prepared_tagger_component.hpp": "public Tagger facade and Spec",
    "pops/runtime/amr/detail/native_tagger_session.hpp": "native Tagger storage, ABI, execution",
    "pops/runtime/amr/prepared_clustering_component.hpp": "clustering facade",
    "pops/runtime/amr/prepared_reflux_component.hpp": "reflux facade",
}
AUTHORITY_LINE_BUDGETS = {
    PREPARED_COMPONENT_UMBRELLA: 12,
    "pops/runtime/amr/prepared_tagger_component.hpp": 120,
    "pops/runtime/amr/detail/native_tagger_session.hpp": 900,
    "pops/runtime/amr/prepared_clustering_component.hpp": 80,
    "pops/runtime/amr/prepared_reflux_component.hpp": 100,
}
UPSTREAM_BOUNDARIES = {
    "pops/amr/tagging/clustering_provider.hpp",
    "pops/core/foundation/allocator.hpp",
    "pops/core/identity/prepared_provider.hpp",
    "pops/parallel/comm.hpp",
    "pops/runtime/amr/amr_runtime.hpp",
    "pops/runtime/amr/prepared_tagging_execution.hpp",
    "pops/runtime/dynamic/component_loader.hpp",
    "pops/runtime/dynamic/prepared_execution_context.hpp",
    "pops/runtime/program/step_transaction.hpp",
}
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*<(?P<path>pops/[^>]+)>', re.MULTILINE)


def _source(relative: str) -> str:
    return (INCLUDE / relative).read_text(encoding="utf-8")


def _local_includes(source: str) -> set[str]:
    return {match["path"] for match in LOCAL_INCLUDE.finditer(source)}


def _prepared_component_closure() -> dict[str, str]:
    pending = [PREPARED_COMPONENT_UMBRELLA]
    closure: dict[str, str] = {}
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        assert relative in SEMANTIC_AUTHORITIES, relative
        source = _source(relative)
        closure[relative] = source
        for dependency in _local_includes(source):
            assert dependency in set(SEMANTIC_AUTHORITIES) | UPSTREAM_BOUNDARIES, (
                f"{relative} includes unclassified local dependency {dependency}"
            )
            if dependency in SEMANTIC_AUTHORITIES:
                pending.append(dependency)
    assert closure.keys() == SEMANTIC_AUTHORITIES.keys()
    return closure


def _without_get_value_indices(source: str) -> str:
    """Do not confuse tuple value selection with a fixed spatial rank."""
    return re.sub(r"std::get\s*<\s*\d+\s*>", "std::get<VALUE_INDEX>", source)


def _assert_no_fixed_2d_or_dispatch(source: str) -> None:
    source = _without_get_value_indices(source)
    assert not re.search(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b", source)
    assert not re.search(r"<\s*2\s*>", source)


def test_ranked_amr_providers_have_no_2d_storage_or_dimension_dispatch() -> None:
    components = _prepared_component_closure()
    ranked_sources = {relative: _source(relative) for relative in RANKED_PROVIDERS}
    joined = "\n".join((*ranked_sources.values(), *components.values()))

    for forbidden in (
        "Box2D",
        "Fab2D",
        "DistributionMapping",
        "TagBox",
        "Array4",
        "FluxRegister",
        "ProgramExecutionServices",
        "PreparedAmrProgramRefluxPlan",
        "pops/amr/tagging/cluster.hpp",
        "pops/amr/tagging/tag_box.hpp",
    ):
        assert forbidden not in joined, forbidden

    _assert_no_fixed_2d_or_dispatch(joined)
    assert sum(source.count("\n") + 1 for source in ranked_sources.values()) < 800
    assert sum(source.count("\n") + 1 for source in components.values()) <= 1200
    for relative, budget in AUTHORITY_LINE_BUDGETS.items():
        assert components[relative].count("\n") + 1 <= budget, relative


def test_prepared_component_decomposition_has_bounded_responsibilities() -> None:
    closure = _prepared_component_closure()
    umbrella = closure[PREPARED_COMPONENT_UMBRELLA]
    facade = closure["pops/runtime/amr/prepared_tagger_component.hpp"]
    session = closure["pops/runtime/amr/detail/native_tagger_session.hpp"]
    clustering = closure["pops/runtime/amr/prepared_clustering_component.hpp"]
    reflux = closure["pops/runtime/amr/prepared_reflux_component.hpp"]

    assert umbrella.count("#include <pops/runtime/amr/") == 3
    assert "PreparedTaggerComponentSpec" in facade
    assert "NativeTaggerSession" in facade
    assert "std::unique_ptr<Session> session_" in facade
    assert "return session_->execute" in facade
    for forbidden in ("PopsTaggerRequestV2", "ComponentState", "execute_patch_", "Storage {"):
        assert forbidden not in facade, forbidden

    assert "class NativeTaggerSession" in session
    assert "struct Storage" in session
    assert "PopsTaggerRequestV2" in session
    assert "component::tag_batch" in session
    assert "execute_patch_" in session
    assert "Storage storage{}" in session
    assert session.index("std::shared_ptr<component::LoadedComponent> component") < session.index(
        "ComponentState state"
    )
    assert "class PreparedTaggerComponent" not in session

    assert "ClusterProvider<Dim>" in clustering
    assert "provider_->cluster" in clustering
    assert "AmrRuntime" not in clustering
    assert "PreparedTagger" not in clustering
    assert "PreparedReflux" not in clustering

    assert "runtime.reconcile_reflux" in reflux
    assert "PreparedRefluxRequest" in reflux
    assert "PreparedTagger" not in reflux
    assert "ClusterProvider" not in reflux


def test_generic_providers_delegate_to_canonical_ranked_authorities() -> None:
    transfer = _source("pops/runtime/amr/bootstrap_transfer_builtins.hpp")
    reduction = _source("pops/runtime/amr/composite_reduction.hpp")
    components = "\n".join(_prepared_component_closure().values())
    block = _source("pops/runtime/builders/block/block_builder.hpp")
    amr_block = _source("pops/runtime/builders/compiled/amr_dsl_block.hpp")

    assert "PreparedTransfer<Dim>" in transfer
    assert "prepare_linear_prolongation" in transfer
    assert "prepare_average_down" in transfer
    assert "prepare_fill_patch" in transfer

    assert "class MemorySpace" in reduction
    assert "MultiFab<Dim, MemorySpace>" in reduction
    assert "active-cell mask" in reduction
    assert "ExecutionLane" in reduction

    assert "PreparedTaggerComponent" in components
    assert "TagMask<Dim>" in components
    assert "ClusterProvider<Dim>" in components
    assert "TransactionalFaceFluxLedger<Dim, Payload>" in components
    assert "runtime.reconcile_reflux" in components

    assert "class PreparedBlockOperator" in block
    assert "BlockGeometryCapability" in block
    assert "PreparedProvider<MultiFab<Dim, MemorySpace>" in block
    assert "class PreparedAmrDslBlock" in amr_block
    assert "AmrRuntime<Dim, MemorySpace>" in amr_block
    assert "runtime_->spatial_contract()" in amr_block


def test_composite_fac_poisson_is_an_exact_ranked_generic_solver() -> None:
    source = _source(SPECIALIZED_FAC)

    assert "template <int Dim, class MemorySpace" in source
    assert "class CompositeFacPoisson" in source
    assert "static_assert(Dim >= 1 && Dim <= 3" in source
    assert "using field_type = MultiFab<Dim, MemorySpace>" in source
    assert "CompositeFacBuildRequest<Dim>" in source
    assert "GeometricMG<Dim, MemorySpace>" in source
    for retired in (
        "Box2D",
        "Array4",
        "DistributionMapping",
        "FluxRegister",
        "CompositeFacPoisson2DProvider",
        "PreparedCompositeFacPoisson2DKernel",
        "supported_dimension = 2",
        "MultiFab<2>",
    ):
        assert retired not in source, retired


def test_provider_types_and_contracts_carry_the_selected_rank() -> None:
    components = "\n".join(_prepared_component_closure().values())
    block = _source("pops/runtime/builders/block/block_builder.hpp")
    amr_block = _source("pops/runtime/builders/compiled/amr_dsl_block.hpp")

    for source in (components, block, amr_block):
        assert ".scalar(std::int32_t{Dim})" in source
        assert "collective_contract" in source
    assert "BlockProviderCapabilities<Dim>" in block
    assert "bound_spatial_contract_" in amr_block
    assert "materialization_generation_" in amr_block


def test_tagging_bytecode_and_hysteresis_execute_in_one_exact_native_rank() -> None:
    sources = {relative: _source(relative) for relative in RANKED_TAGGING_EXECUTION}
    sources.update(_prepared_component_closure())
    joined = "\n".join(sources.values())

    for required in (
        "PreparedTaggingProgram<Dim>",
        "PreparedTaggingExecutionPlan",
        "FieldView<const Real, Dim>",
        "TagMask<Dim>",
        "Box<Dim>",
        "Index<Dim>",
        "PersistentTaggingState",
        "all_ranks_agree_exact_ordered_byte_pairs",
        "replicated-consensus budget",
    ):
        assert required in joined, required

    for forbidden in (
        "Box2D",
        "ConstArray4",
        "Array4",
        "TagBox",
        "kPreparedTaggingDimension",
        "requires an exact 2D",
    ):
        assert forbidden not in joined, forbidden
    _assert_no_fixed_2d_or_dispatch(joined)

    binding = BINDINGS.read_text(encoding="utf-8")
    assert "PreparedTaggingProgram<pops::kNativeDimension>" in binding
    assert 'row["dimension"]' in binding
    assert "differs from the selected native specialization" in binding
    assert "result.dimension" not in binding
