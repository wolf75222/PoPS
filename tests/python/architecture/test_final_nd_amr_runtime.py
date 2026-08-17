"""ADC-738: AMR topology and conservative operations have one ranked authority.

This source-only ratchet deliberately builds nothing.  The separate hyperbolic consumer cutover is
outside this ownership lane; this gate locks the canonical AMR geometry, transfer, reflux,
load-balance, and runtime transaction surfaces that those consumers must use.
"""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[3]
INCLUDE = ROOT / "include"
MANIFEST = INCLUDE / "pops_headers.manifest"

CANONICAL_HEADERS = (
    "pops/amr/refinement_ratio.hpp",
    "pops/amr/hierarchy/level_layout.hpp",
    "pops/amr/hierarchy/hierarchy_plan.hpp",
    "pops/amr/hierarchy/amr_hierarchy.hpp",
    "pops/amr/tagging/tag_mask.hpp",
    "pops/amr/tagging/clustering_provider.hpp",
    "pops/amr/tagging/berger_rigoutsos.hpp",
    "pops/amr/regridding/regrid.hpp",
    "pops/amr/transfer/transfer_provider.hpp",
    "pops/amr/reflux/face_flux_ledger.hpp",
    "pops/amr/reflux/metric_reflux.hpp",
    "pops/parallel/ownership_plan.hpp",
    "pops/parallel/load_balance.hpp",
    "pops/parallel/prepared_load_balance.hpp",
    "pops/runtime/amr/amr_runtime.hpp",
)

REMOVED_HEADERS = (
    "pops/amr/nd/refinement_ratio.hpp",
    "pops/amr/hierarchy/refinement_ratio.hpp",
    "pops/amr/hierarchy/nd/level_layout.hpp",
    "pops/amr/hierarchy/nd/hierarchy_plan.hpp",
    "pops/amr/hierarchy/nd/tag_mask.hpp",
    "pops/amr/hierarchy/nd/cluster_provider.hpp",
    "pops/amr/hierarchy/nd/berger_rigoutsos.hpp",
    "pops/amr/tagging/tag_box.hpp",
    "pops/amr/tagging/cluster.hpp",
    "pops/amr/tagging/tagging_truth.hpp",
    "pops/amr/transfer/nd/refinement_ratio.hpp",
    "pops/amr/transfer/nd/transfer_provider.hpp",
    "pops/amr/reflux/nd/face_flux_ledger.hpp",
    "pops/amr/reflux/nd/metric_reflux.hpp",
)


def _source(relative: str) -> str:
    return (INCLUDE / relative).read_text(encoding="utf-8")


def _manifest_rows() -> set[str]:
    return {
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_canonical_amr_authorities_are_compile_time_ranked_without_2d_escape_hatches() -> None:
    sources = {relative: _source(relative) for relative in CANONICAL_HEADERS}
    joined = "\n".join(sources.values())

    for declaration in (
        "class RefinementRatio",
        "class LevelLayout",
        "class HierarchyPlan",
        "class AmrHierarchy",
        "class TagMask",
        "class ClusterProvider",
        "class PreparedRegrid",
        "class PreparedTransfer",
        "class TransactionalFaceFluxLedger",
        "class OwnershipPlan",
        "class LoadBalanceProvider",
        "class PreparedLoadBalanceAuthority",
        "struct PreparedRebalanceDecision",
        "class AmrRuntime",
    ):
        assert declaration in joined, declaration

    dimension_branch = re.compile(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b")
    offenders = [relative for relative, source in sources.items() if dimension_branch.search(source)]
    assert not offenders, "AMR algorithms must iterate axes, not branch on Dim: %r" % offenders

    literal_two_specialization = re.compile(r"<\s*2\s*>")
    specialized = [
        relative for relative, source in sources.items() if literal_two_specialization.search(source)
    ]
    assert not specialized, "canonical AMR code contains a literal rank-2 specialization: %r" % specialized

    for forbidden in (
        "Box2D",
        "Fab2D",
        "DistributionMapping",
        "namespace pops::amr::nd",
        "namespace pops::amr::hierarchy::nd",
        "namespace pops::amr::transfer::nd",
        "namespace pops::amr::reflux::nd",
        "static_assert(Dim == 2",
    ):
        assert forbidden not in joined, forbidden


def test_amr_runtime_keeps_spatial_evidence_through_regrid_rebalance_transfer_and_reflux() -> None:
    runtime = _source("pops/runtime/amr/amr_runtime.hpp")
    regrid = _source("pops/amr/regridding/regrid.hpp")
    load_balance = _source("pops/parallel/prepared_load_balance.hpp")

    for token in (
        "exact_runtime_spatial_contract",
        "prepare_regrid(",
        "publish_regrid(",
        "prepare_rebalance(",
        "apply_rebalance(",
        "prepare_transfer(",
        "reconcile_reflux(",
        "LevelStateSpatialContract<Dim>",
    ):
        assert token in runtime, token
    assert "PreparedLoadBalanceResult<Dim> proposed" in load_balance
    assert "decision.proposed.plan().distribution()" in runtime
    assert "PreparedLoadBalanceResult<Dim>> ownership_" in regrid
    assert "ownership->plan().distribution()" in regrid


def test_temporary_and_2d_amr_authorities_are_deleted_and_uninstalled() -> None:
    present = [relative for relative in REMOVED_HEADERS if (INCLUDE / relative).exists()]
    assert not present, "duplicate AMR authorities must be deleted: %r" % present

    rows = _manifest_rows()
    expected = {f"api {relative}" for relative in CANONICAL_HEADERS if "runtime/" not in relative}
    expected.add("sdk-support pops/runtime/amr/amr_runtime.hpp")
    assert expected <= rows
    stale = {row for row in rows if any(row.endswith(relative) for relative in REMOVED_HEADERS)}
    assert not stale, "removed AMR authorities remain packaged: %r" % sorted(stale)


def test_legacy_core_tests_are_replaced_by_ranked_coverage() -> None:
    removed = (
        "tests/cpp/integration/amr/test_amr_hierarchy.cpp",
        "tests/cpp/integration/amr/test_ref_ratio.cpp",
        "tests/cpp/integration/amr/test_regrid.cpp",
        "tests/cpp/unit/mesh/test_cluster.cpp",
        "tests/cpp/unit/mesh/test_load_balance.cpp",
    )
    assert not [relative for relative in removed if (ROOT / relative).exists()]

    for relative in (
        "tests/cpp/unit/amr/test_nd_amr_runtime.cpp",
        "tests/cpp/unit/amr/test_nd_flux_ledger.cpp",
        "tests/cpp/unit/amr/test_nd_transfer.cpp",
        "tests/cpp/unit/mesh/test_nd_cluster.cpp",
        "tests/cpp/unit/mesh/test_nd_hierarchy_plan.cpp",
        "tests/cpp/unit/mesh/test_nd_load_balance.cpp",
        "tests/cpp/unit/mesh/test_nd_tag_mask.cpp",
    ):
        assert (ROOT / relative).is_file(), relative


def test_python_amr_product_is_not_hardcoded_2d() -> None:
    forbidden = (
        "2D isotropic ratio-(2,2)",
        "square n x n",
        "static_assert(Dim == 2",
    )
    roots = (
        ROOT / "python" / "pops" / "amr",
        ROOT / "python" / "pops" / "mesh" / "_amr",
        ROOT / "include" / "pops" / "amr",
    )
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".hpp", ".h", ".inc"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append((str(path.relative_to(ROOT)), token))
    assert not offenders, offenders
