"""ADC-734: mesh identity has one compile-time-ranked production authority.

This is a source-only ratchet.  It deliberately does not import ``pops`` or build the native
extension, so duplicate 2D/test-only definitions cannot hide behind an unavailable toolchain.
"""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[3]
INCLUDE = ROOT / "include"
MANIFEST = INCLUDE / "pops_headers.manifest"

CANONICAL_HEADERS = (
    "pops/mesh/index/index.hpp",
    "pops/mesh/index/extent.hpp",
    "pops/mesh/index/box.hpp",
    "pops/mesh/index/box_hash.hpp",
    "pops/mesh/layout/box_array.hpp",
    "pops/mesh/layout/distribution.hpp",
    "pops/mesh/layout/rank_space.hpp",
    "pops/mesh/geometry/geometry.hpp",
)

REMOVED_HEADERS = (
    "pops/mesh/index/box2d.hpp",
    "pops/mesh/layout/distribution_mapping.hpp",
    "pops/mesh/layout/nd/box_array.hpp",
    "pops/mesh/layout/nd/distribution.hpp",
    "pops/mesh/layout/nd/rank_space.hpp",
    "pops/mesh/nd_proof/box_array.hpp",
    "pops/mesh/nd_proof/box_hash.hpp",
    "pops/mesh/nd_proof/distribution.hpp",
    "pops/mesh/nd_proof/rank_space.hpp",
)


def _source(relative: str) -> str:
    return (INCLUDE / relative).read_text(encoding="utf-8")


def _manifest_rows() -> set[str]:
    return {
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_canonical_mesh_headers_are_compile_time_ranked_without_dimension_branches() -> None:
    sources = {relative: _source(relative) for relative in CANONICAL_HEADERS}
    joined = "\n".join(sources.values())

    assert joined.count("Dim >= 1 && Dim <= 3") >= len(CANONICAL_HEADERS)
    for token in (
        "template <int Dim>\nstruct Index",
        "template <int Dim>\nstruct Extent",
        "template <int Dim>\nstruct Box",
        "template <int Dim>\nclass BoxHash",
        "template <int Dim>\nclass BoxArray",
        "template <int Dim>\nclass Distribution",
        "template <int Dim>\nclass RankSpace",
        "template <int Dim>\nclass Geometry",
    ):
        assert token in joined, token

    dimension_branch = re.compile(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b")
    offenders = [relative for relative, source in sources.items() if dimension_branch.search(source)]
    assert not offenders, "canonical mesh code must iterate axes, not branch on Dim: %r" % offenders

    for forbidden in ("Box2D", "DistributionMapping", "PolarGeometry", "namespace nd_proof"):
        assert forbidden not in joined, forbidden


def test_legacy_and_test_only_duplicate_mesh_authorities_are_deleted() -> None:
    present = [relative for relative in REMOVED_HEADERS if (INCLUDE / relative).exists()]
    assert not present, "duplicate mesh authorities must be deleted: %r" % present


def test_header_manifest_installs_only_the_canonical_mesh_authorities() -> None:
    rows = _manifest_rows()
    expected = {f"api {relative}" for relative in CANONICAL_HEADERS}
    assert expected <= rows

    stale = {
        row
        for row in rows
        if any(row.endswith(relative) for relative in REMOVED_HEADERS)
    }
    assert not stale, "removed mesh authorities remain packaged: %r" % sorted(stale)


def test_nd_proof_consumers_use_canonical_layout_identity_directly() -> None:
    local_neighbors = _source("pops/mesh/nd_proof/local_neighbors.hpp")
    translation = _source("pops/mesh/nd_proof/translation_schedule.hpp")
    combined = local_neighbors + translation

    for include in (
        "#include <pops/mesh/index/box_hash.hpp>",
        "#include <pops/mesh/layout/distribution.hpp>",
    ):
        assert include in combined
    for forbidden in (
        "nd_proof/box_array.hpp",
        "nd_proof/box_hash.hpp",
        "nd_proof/distribution.hpp",
        "nd_proof/rank_space.hpp",
        "proof_layout",
    ):
        assert forbidden not in combined, forbidden
