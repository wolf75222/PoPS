"""Final ND uniform/AMR facade and Python-boundary architecture ratchet."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SYSTEM_HEADER = ROOT / "include/pops/runtime/system.hpp"
AMR_HEADER = ROOT / "include/pops/runtime/amr_system.hpp"
SPATIAL_DOMAIN = ROOT / "include/pops/runtime/config/spatial_domain.hpp"
SYSTEM_DOMAIN = ROOT / "include/pops/runtime/system/system_domain.hpp"
LAYOUT_TRANSFER = ROOT / "src/runtime/system/system_layout_transfer.cpp"
BINDING_DETAIL = ROOT / "python/bindings/core/bindings_detail.hpp"
CORE_BINDING = ROOT / "python/bindings/core/init/init_core.cpp"
SYSTEM_BINDING = ROOT / "python/bindings/core/init/init_system.cpp"
AMR_BINDING = ROOT / "python/bindings/core/init/init_amr.cpp"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_uniform_and_amr_facades_have_one_visible_ranked_template() -> None:
    system = _read(SYSTEM_HEADER)
    amr = _read(AMR_HEADER)

    assert re.search(r"template\s*<\s*int\s+Dim\s*>\s*struct\s+SystemConfig\s*:", system)
    assert re.search(r"template\s*<\s*int\s+Dim\s*>\s*class\s+System\s*\{", system)
    assert re.search(r"template\s*<\s*int\s+Dim\s*>\s*struct\s+AmrSystemConfig\s*:", amr)
    assert re.search(r"template\s*<\s*int\s+Dim\s*>\s*class\s+AmrSystem\s*\{", amr)
    assert "class System<2>" not in system
    assert "class AmrSystem<2>" not in amr
    assert "struct SystemConfig<2>" not in system
    assert "struct AmrSystemConfig<2>" not in amr


def test_ranked_domain_is_one_authority_from_config_through_storage() -> None:
    config = _read(SPATIAL_DOMAIN)
    domain = _read(SYSTEM_DOMAIN)

    for token in (
        "Extent<Dim> shape",
        "RealVector<Dim> lower",
        "RealVector<Dim> upper",
        "std::array<bool, Dim> periodicity",
        "std::vector<Box<Dim>> boxes",
        "Box<Dim> index_domain() const",
    ):
        assert token in config
    for token in (
        "SystemConfig<Dim> cfg",
        "Box<Dim> dom",
        "Geometry<Dim> geom",
        "mesh::BoxArray<Dim>",
        "mesh::Distribution<Dim>",
        "mesh::RankSpace<Dim>",
        "MultiFab<Dim>",
        "PreparedLoadBalanceAuthority<Dim>",
        ".plan().distribution()",
    ):
        assert token in domain


def test_python_selects_the_artifact_rank_without_shape_inference() -> None:
    core = _read(CORE_BINDING)
    system = _read(SYSTEM_BINDING)
    amr = _read(AMR_BINDING)
    detail = _read(BINDING_DETAIL)

    assert "using NativeSystemConfig = SystemConfig<kNativeDimension>;" in core
    assert "using System = pops::System<pops::kNativeDimension>;" in system
    assert "using AmrSystem = pops::AmrSystem<pops::kNativeDimension>;" in amr
    assert "using AmrSystemConfig = pops::AmrSystemConfig<pops::kNativeDimension>;" in amr
    assert "native_shape[Dim - 1 - numpy_axis]" in detail
    assert "shape.insert(shape.begin(), static_cast<py::ssize_t>(ncomp));" in detail
    assert "array.ndim() != pops::kNativeDimension" in amr


def test_generic_core_has_no_parallel_2d_mesh_authority() -> None:
    paths = (
        SPATIAL_DOMAIN,
        SYSTEM_DOMAIN,
        LAYOUT_TRANSFER,
        BINDING_DETAIL,
        CORE_BINDING,
        SYSTEM_BINDING,
        AMR_BINDING,
    )
    forbidden = (
        r"\bBox2D\b",
        r"\bFab2D\b",
        r"\bDistributionMapping\b",
        r"\bPatchBox\b",
        r"\bto_2d\b",
        r"\bto_3d\b",
        r"\.nx\s*\(",
        r"\.ny\s*\(",
        r"<\s*2\s*>",
    )
    violations: list[str] = []
    for path in paths:
        source = _read(path)
        for pattern in forbidden:
            if re.search(pattern, source):
                violations.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert not violations, "2D authority leaked back into the ranked core:\n" + "\n".join(
        violations
    )


def test_layout_transfer_is_generic_and_instantiated_only_for_the_artifact() -> None:
    source = _read(LAYOUT_TRANSFER)
    assert "template <int Dim>" in source
    assert "SystemLayoutTransferSpec<Dim>" in source
    assert "MultiFab<Dim>" in source
    assert "Box<Dim>" in source
    assert "template class PreparedSystemLayoutTransfer<kNativeDimension>;" in source
    assert "if constexpr" not in source
    assert not re.search(r"\bif\s*\(\s*Dim\s*(?:==|!=|<=|>=|<|>)", source)
