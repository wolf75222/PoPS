"""Final AMR builder ratchet: one exact-ranked generated package, no legacy seam product."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUILDERS = ROOT / "src" / "runtime" / "builders"
RUNTIME_CMAKE = ROOT / "src" / "CMakeLists.txt"
GENERATED_PACKAGE = (
    ROOT
    / "include"
    / "pops"
    / "runtime"
    / "builders"
    / "compiled"
    / "generated_amr_system_block.hpp"
)

RETIRED_PRODUCT = (
    ROOT / "include/pops/runtime/builders/block/amr_block_seam.hpp",
    BUILDERS / "seam_combinations.cmake",
    BUILDERS / "templates/amr_block_transport_seam.cpp.in",
    BUILDERS / "templates/amr_block_flux_seam.cpp.in",
    BUILDERS / "amr/block/compressible/amr_block_compressible.cpp",
)


def test_legacy_amr_builder_product_is_absent() -> None:
    present = [str(path.relative_to(ROOT)) for path in RETIRED_PRODUCT if path.exists()]
    assert not present, "legacy AMR builder seams returned:\n  " + "\n  ".join(present)


def test_runtime_build_owns_no_generated_seam_graph() -> None:
    source = RUNTIME_CMAKE.read_text(encoding="utf-8")
    for token in (
        "seam_combinations.cmake",
        "POPS_GENERATED_SEAMS_DIR",
        "POPS_SEAM_COMBINATIONS",
        "POPS_RUNTIME_AMR_GENERATED_SEAMS",
        "amr_block_compressible.cpp",
    ):
        assert token not in source


def test_exact_ranked_generated_package_owns_every_supported_dimension() -> None:
    source = GENERATED_PACKAGE.read_text(encoding="utf-8")
    assert "template <int Dim" in source
    assert "struct PreparedAmrSystemBlock" in source
    assert "static constexpr int dimension = Dim;" in source
    assert "PreparedAmrSystemBlock<Request::dimension>" in source
    assert "prepare_compiled_amr_system_block<Dim>" in source
    assert "for (int axis = 0; axis < Dim; ++axis)" in source
    assert "Box2D" not in source
    assert "Fab2D" not in source
    assert "if constexpr (Dim" not in source
    assert not re.search(r"\bif\s*\(\s*Dim\s*(?:==|!=|<=|>=|<|>)", source)
