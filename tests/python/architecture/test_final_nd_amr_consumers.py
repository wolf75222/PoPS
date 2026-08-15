"""ADC-738: AMR coupling and Program consumers retain one compile-time rank."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ROOT / "include"

CONSUMERS = (
    "pops/coupling/amr/amr_coupler_mp.hpp",
    "pops/coupling/amr/amr_regrid_coupler.hpp",
    "pops/coupling/system/amr_system_coupler.hpp",
    "pops/numerics/time/amr/levels/amr_patch_range.hpp",
    "pops/numerics/time/amr/levels/amr_subcycling.hpp",
    "pops/numerics/time/amr/reflux/amr_flux_helpers.hpp",
    "pops/runtime/program/amr_program_context.hpp",
)


def _sources() -> dict[str, str]:
    return {
        relative: (INCLUDE / relative).read_text(encoding="utf-8")
        for relative in CONSUMERS
    }


def test_amr_consumers_are_ranked_facades_without_a_2d_authority() -> None:
    sources = _sources()
    joined = "\n".join(sources.values())

    for token in (
        "CoarseHierarchyRequest",
        "class AmrCouplerMP",
        "class AmrRegridCoupler",
        "class AmrSystemCoupler",
        "class PatchRange",
        "class CoarseFineInterface",
        "class PreparedAmrSubcycleTransition",
        "class PreparedAmrSubcyclePlan",
        "struct PreparedTransferKernel",
        "class AmrProgramContext",
    ):
        assert token in joined, token

    forbidden_tokens = (
        "Box2D",
        "Fab2D",
        "DistributionMapping",
        "TagBox",
        "kAmrRefRatio",
        "Array4",
        "FluxRegister",
        "ProgramExecutionServices<AmrProgramContext>",
        "pops/amr/hierarchy/refinement_ratio.hpp",
        "pops/amr/tagging/cluster.hpp",
        "pops/amr/tagging/tag_box.hpp",
    )
    for forbidden in forbidden_tokens:
        assert forbidden not in joined, forbidden

    for relative, source in sources.items():
        assert re.search(r"template\s*<[^>]*\bint Dim\b", source), relative
    assert "kNativeDimension" not in joined
    assert not re.search(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b", joined)
    assert not re.search(r"\bDim\s*(?:==|!=)\s*2\b", joined)
    assert not re.search(r"<\s*2\s*>", joined)


def test_consumers_delegate_to_canonical_prepared_authorities() -> None:
    sources = _sources()

    transfer = sources["pops/numerics/time/amr/reflux/amr_flux_helpers.hpp"]
    assert "runtime.template prepare_transfer" in transfer
    assert "PreparedTransfer<Dim>" in transfer
    assert "for_each_cell" in transfer

    subcycling = sources["pops/numerics/time/amr/levels/amr_subcycling.hpp"]
    assert "LevelStateSpatialContract<Dim>" in subcycling
    assert "runtime.spatial_contract()" in subcycling
    assert "runtime.reconcile_reflux" in subcycling

    regrid = sources["pops/coupling/amr/amr_regrid_coupler.hpp"]
    assert "ClusterProvider<Dim>" in regrid
    assert "runtime_->prepare_regrid" in regrid
    assert "runtime_->publish_regrid" in regrid

    coupler = sources["pops/coupling/amr/amr_coupler_mp.hpp"]
    assert "load_balance.prepare(" in coupler
    assert "ownership.plan().distribution()" in coupler
    assert "AmrRuntime<Dim, MemorySpace>" in coupler

    program = sources["pops/runtime/program/amr_program_context.hpp"]
    for operation in (
        "prepare_subcycling(",
        "prepare_regrid(",
        "publish_regrid(",
        "prepare_rebalance(",
        "apply_rebalance(",
        "reconcile_reflux(",
    ):
        assert operation in program


def test_direct_native_proof_exercises_one_and_three_dimensional_consumers() -> None:
    proof = ROOT / "tests/cpp/unit/amr/test_nd_amr_consumers.cpp"
    assert proof.is_file()
    source = proof.read_text(encoding="utf-8")
    assert "AmrCouplerMP<1>::dimension == 1" in source
    assert "AmrSystemCoupler<3>::dimension == 3" in source
    assert "AmrProgramContext<3>::dimension == 3" in source
    assert "PatchRange<1>" in source
    assert "PatchRange<3>" in source
    assert "RefinementRatio<3>{2, 3, 1}" in source
    assert "level_domains_refine_every_axis_without_rank_dispatch" in source
    assert "interface_retains_anisotropic_layout_identity" in source
