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


def _source(relative: str) -> str:
    return (INCLUDE / relative).read_text(encoding="utf-8")


def test_ranked_amr_providers_have_no_2d_storage_or_dimension_dispatch() -> None:
    sources = {relative: _source(relative) for relative in RANKED_PROVIDERS}
    joined = "\n".join(sources.values())

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

    assert not re.search(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b", joined)
    assert not re.search(r"<\s*2\s*>", joined)
    assert sum(source.count("\n") + 1 for source in sources.values()) < 800


def test_generic_providers_delegate_to_canonical_ranked_authorities() -> None:
    transfer = _source("pops/runtime/amr/bootstrap_transfer_builtins.hpp")
    reduction = _source("pops/runtime/amr/composite_reduction.hpp")
    components = _source("pops/runtime/amr/prepared_component_providers.hpp")
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


def test_historical_fac_is_an_explicit_2d_capability_not_a_generic_solver() -> None:
    source = _source(SPECIALIZED_FAC)

    assert "class CompositeFacPoisson2DProvider" in source
    assert "supported_dimension = 2" in source
    assert "CompositeFacCapabilityRequest" in source
    assert "PreparedProviderSupport" in source
    assert "PreparedCompositeFacPoisson2DKernel" in source
    assert "template <int Dim>" not in source
    assert "MultiFab" not in source
    for retired in (
        "Box2D",
        "Array4",
        "DistributionMapping",
        "GeometricMG",
        "FluxRegister",
    ):
        assert retired not in source, retired


def test_provider_types_and_contracts_carry_the_selected_rank() -> None:
    components = _source("pops/runtime/amr/prepared_component_providers.hpp")
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
    assert not re.search(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b", joined)
    assert not re.search(r"<\s*2\s*>", joined)

    binding = BINDINGS.read_text(encoding="utf-8")
    assert "PreparedTaggingProgram<pops::kNativeDimension>" in binding
    assert "row[\"dimension\"]" in binding
    assert "differs from the selected native specialization" in binding
    assert "result.dimension" not in binding
