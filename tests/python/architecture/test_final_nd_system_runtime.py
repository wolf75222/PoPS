"""Final ND uniform/AMR facade and Python-boundary architecture ratchet."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SYSTEM_HEADER = ROOT / "include/pops/runtime/system.hpp"
SYSTEM_RUNTIME = ROOT / "src/runtime/system/system.cpp"
RETIRED_PROGRAM_DRIVER = (
    ROOT / "include/pops/runtime/system/system_program_driver.hpp"
)
AMR_HEADER = ROOT / "include/pops/runtime/amr_system.hpp"
SPATIAL_DOMAIN = ROOT / "include/pops/runtime/config/spatial_domain.hpp"
SYSTEM_DOMAIN = ROOT / "include/pops/runtime/system/system_domain.hpp"
BLOCK_STORE = ROOT / "include/pops/runtime/system/system_block_store.hpp"
BOUNDARY_REGISTRY = ROOT / "include/pops/runtime/system/system_boundary_registry.hpp"
LAYOUT_TRANSFER = ROOT / "src/runtime/system/system_layout_transfer.cpp"
BINDING_DETAIL = ROOT / "python/bindings/core/bindings_detail.hpp"
CORE_BINDING = ROOT / "python/bindings/core/init/init_core.cpp"
SYSTEM_BINDING = ROOT / "python/bindings/core/init/init_system.cpp"
AMR_BINDING = ROOT / "python/bindings/core/init/init_amr.cpp"
PERIODICITY = ROOT / "include/pops/mesh/boundary/periodicity.hpp"
LEGACY_BOUNDARY_PLAN = ROOT / "include/pops/mesh/boundary/prepared_boundary_plan.hpp"
RUNTIME_AUTHORITIES = ROOT / "python/pops/runtime/_runtime_authorities.py"


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
    assert "implicit_stepper.hpp" not in system
    assert "implicit_stepper.hpp" not in amr
    assert "numerics/nonlinear/newton_options.hpp" in system
    assert "numerics/nonlinear/newton_options.hpp" in amr


def test_system_step_driver_is_the_exact_ranked_facade_not_a_parallel_authority() -> None:
    runtime = _read(SYSTEM_RUNTIME)
    manifest = _read(ROOT / "include/pops_headers.manifest")

    assert not RETIRED_PROGRAM_DRIVER.exists()
    assert "system_program_driver.hpp" not in manifest
    assert "System<Dim>::step(double dt)" in runtime
    assert "System<Dim>::step_cfl(" in runtime
    assert "p_->geom.spacing(axis)" in runtime
    assert "p_->coupling_.coupled_frequencies" in runtime
    assert "p_->coupling_.dt_bounds" in runtime
    assert "dispatch_cadence_step(" in runtime
    assert "if constexpr" not in runtime
    assert not re.search(r"\bif\s*\(\s*Dim\s*(?:==|!=|<=|>=|<|>)", runtime)
    for legacy in ("SystemProgramDriver", "Box2D", "Array4"):
        assert legacy not in runtime


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
        BLOCK_STORE,
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


def test_block_store_retains_only_the_ranked_hyperbolic_boundary() -> None:
    source = _read(BLOCK_STORE)
    facade = _read(SYSTEM_HEADER)
    amr_facade = _read(AMR_HEADER)
    assert "template <int Dim>" in source
    assert "PreparedHyperbolicBoundary<Dim>" in source
    assert "std::shared_ptr<const boundary_type> boundary" in source
    assert "install_hyperbolic_boundary" in facade
    for legacy in ("PreparedBoundaryPlan", "PreparedGridBoundarySession"):
        assert legacy not in source
        assert legacy not in facade
        assert legacy not in amr_facade
    assert "interface_flux_scheduler.hpp" not in source


def test_periodicity_rows_are_ranked_and_never_restore_a_2d_core_authority() -> None:
    sources = {
        path: _read(path)
        for path in (
            PERIODICITY,
            LEGACY_BOUNDARY_PLAN,
            BINDING_DETAIL,
            SYSTEM_BINDING,
            AMR_BINDING,
            RUNTIME_AUTHORITIES,
        )
    }
    assert "template <int Dim>\nstruct PeriodicIdentification" in sources[PERIODICITY]
    assert "2 + 2 * Dim" in sources[PERIODICITY]
    assert "2 + 2 * Dim" in sources[BINDING_DETAIL]
    assert "2 * dimension" in sources[RUNTIME_AUTHORITIES]
    for path, source in sources.items():
        assert "PeriodicIdentification2D" not in source, path.relative_to(ROOT)
        assert "std::array<int, 6>" not in source, path.relative_to(ROOT)


def test_amr_hierarchy_config_carries_one_ranked_row_per_transition() -> None:
    facade = _read(AMR_HEADER)
    binding = _read(AMR_BINDING)
    detail = _read(BINDING_DETAIL)
    for name in ("transition_ratios", "transition_buffers", "transition_lookaheads"):
        assert f"std::vector<Extent<Dim>> {name}" in facade
        assert f'"{name}"' in binding
        assert f"config.{name}" in binding
    assert "regrid_grow" not in facade
    assert "regrid_margin" not in facade
    assert "regrid_grow" not in binding
    assert "regrid_margin" not in binding
    assert "ranked_extents_from_python<kNativeDimension>" in binding
    assert "2 + 2 * Dim" in detail


def test_boundary_installation_registry_is_ranked_and_transactional() -> None:
    source = _read(BOUNDARY_REGISTRY)
    assert "template <int Dim>" in source
    assert "PreparedHyperbolicBoundary<Dim>" in source
    assert "prepare_hyperbolic_boundary<Dim>" in source
    assert "void discard_transaction() noexcept" in source
    assert "with_converted_fixed_states" in source
    for legacy in (
        "PreparedBoundaryPlan",
        "PreparedGridBoundarySession",
        "PeriodicIdentification2D",
        "Box2D",
        "Fab2D",
        "kNativeDimension",
    ):
        assert legacy not in source


def test_layout_transfer_is_generic_and_instantiated_only_for_the_artifact() -> None:
    source = _read(LAYOUT_TRANSFER)
    assert "template <int Dim>" in source
    assert "SystemLayoutTransferSpec<Dim>" in source
    assert "MultiFab<Dim>" in source
    assert "Box<Dim>" in source
    assert "template class PreparedSystemLayoutTransfer<kNativeDimension>;" in source
    assert "if constexpr" not in source
    assert not re.search(r"\bif\s*\(\s*Dim\s*(?:==|!=|<=|>=|<|>)", source)
