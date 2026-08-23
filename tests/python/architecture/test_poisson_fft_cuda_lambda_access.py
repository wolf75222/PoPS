"""CUDA compilation contract for elliptic and System device-kernel launch helpers.

NVCC rejects an extended ``__host__ __device__`` lambda when its lexical enclosing
member function is private or protected.  Kokkos expands ``KOKKOS_LAMBDA`` to that
kind of lambda for CUDA, so this structural contract prevents a host-only edit from
reintroducing the CUDA-only compiler failure.  It does not execute an FFT.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FFT_HEADER = ROOT / "include/pops/numerics/elliptic/poisson/poisson_fft.hpp"
FFT_SOLVER_HEADER = ROOT / "include/pops/numerics/elliptic/poisson/poisson_fft_solver.hpp"
FFT_MULTIFAB_HEADER = ROOT / "include/pops/numerics/elliptic/poisson/poisson_fft_multifab.hpp"
GEOMETRIC_MG_HEADER = ROOT / "include/pops/numerics/elliptic/mg/geometric_mg.hpp"
COMPOSITE_FAC_HEADER = ROOT / "include/pops/numerics/elliptic/mg/composite_fac_poisson.hpp"
PARTITIONED_COMPOSITE_FAC_HEADER = (
    ROOT / "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp"
)
SYSTEM_INSTALL_SOURCE = ROOT / "src/runtime/system/system_install.cpp"

CUDA_LAUNCH_HELPERS = (
    "local_dft_axis_",
    "local_radix2_axis_",
    "distributed_last_dft_",
    "distributed_last_radix2_",
    "local_last_radix_stage_",
    "distributed_radix_stage_",
    "apply_discrete_inverse_symbol_",
)


def _access_at_member(source: str, member: str) -> str:
    member_offset = source.index(f"  void {member}(")
    access_markers = (
        (source.rfind("\n public:\n", 0, member_offset), "public"),
        (source.rfind("\n private:\n", 0, member_offset), "private"),
        (source.rfind("\n protected:\n", 0, member_offset), "protected"),
    )
    return max(access_markers)[1]


def test_cuda_lambda_launch_helpers_have_public_lexical_parents() -> None:
    source = FFT_HEADER.read_text()

    launch_region_start = source.index("\n public:\n", source.index("  bool try_local_fftw_axis_"))
    launch_region_end = source.index("\n private:\n", launch_region_start + 1)
    assert all(
        launch_region_start < offset < launch_region_end
        for offset in (
            offset
            for offset in range(len(source))
            if source.startswith("KOKKOS_LAMBDA", offset)
        )
    )
    assert {
        member: _access_at_member(source, member) for member in CUDA_LAUNCH_HELPERS
    } == {member: "public" for member in CUDA_LAUNCH_HELPERS}


def _braced_definition(source: str, start: str) -> str:
    definition_offset = source.index(start)
    opening_brace = source.index("{", definition_offset)
    depth = 0
    for offset in range(opening_brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[definition_offset : offset + 1]
    raise AssertionError(f"unterminated definition: {start}")


def _access_at_offset(source: str, offset: int) -> str:
    markers = (
        (source.rfind("\n public:\n", 0, offset), "public"),
        (source.rfind("\n private:\n", 0, offset), "private"),
        (source.rfind("\n protected:\n", 0, offset), "protected"),
    )
    nearest = max(markers, key=lambda marker: marker[0])
    assert nearest[0] >= 0, "device parent must declare its lexical access explicitly"
    return nearest[1]


def _public_device_operator(source: str, struct_name: str) -> None:
    struct = _braced_definition(source, f"struct {struct_name}")
    operator_offset = struct.index("POPS_HD void operator()")
    assert _access_at_offset(struct, operator_offset) == "public"


def _public_frequency_operator_contains_device_lambda(source: str) -> None:
    struct = _braced_definition(source, "struct CoupledSourceMaximumFrequency")
    operator = _braced_definition(struct, "Real operator()() const")
    operator_offset = struct.index("Real operator()() const")
    assert _access_at_offset(struct, operator_offset) == "public"
    assert "for_each_cell_reduce_max(reference.box(local), [=] POPS_HD" in operator


def _contains_no_device_lambda(source: str, start: str, end: str) -> None:
    start_offset = source.index(start)
    end_offset = source.index(end, start_offset)
    assert "POPS_HD" not in source[start_offset:end_offset]


def test_multifab_and_multigrid_device_functors_replace_private_parents() -> None:
    multifab = FFT_MULTIFAB_HEADER.read_text()
    geometric_mg = GEOMETRIC_MG_HEADER.read_text()
    composite_fac = COMPOSITE_FAC_HEADER.read_text()
    partitioned_composite_fac = PARTITIONED_COMPOSITE_FAC_HEADER.read_text()

    for source, struct_name in (
        (multifab, "PackSlabKernel"),
        (multifab, "UnpackSlabKernel"),
        (geometric_mg, "CopyComponentsKernel"),
    ):
        _public_device_operator(source, struct_name)

    _contains_no_device_lambda(multifab, "  void pack_slab_", "  void unpack_slab_")
    _contains_no_device_lambda(multifab, "  void unpack_slab_", "  void subtract_mean_")
    _contains_no_device_lambda(geometric_mg, "  SolveReport solve_dynamic_", "  FieldBoundary")
    _contains_no_device_lambda(geometric_mg, "  static void copy_vector_valid_", "  void smooth_")
    _contains_no_device_lambda(composite_fac, "  static void copy_vector_valid_", "  void smooth_level_")
    _contains_no_device_lambda(composite_fac, "  static void copy_grown_", "  void stage_iterate_")
    _contains_no_device_lambda(
        partitioned_composite_fac,
        "  static void copy_vector_valid_",
        "  void smooth_(",
    )
    partitioned_copy = _braced_definition(
        partitioned_composite_fac,
        "  static void copy_vector_valid_",
    )
    assert "::pops::elliptic::mg::detail::CopyComponentsKernel<" in partitioned_copy


def test_fft_solver_named_device_functors_replace_local_for_each_lambdas() -> None:
    source = FFT_SOLVER_HEADER.read_text()
    solve = _braced_definition(source, "  SolveReport solve()")

    for struct_name in ("PackSolverSlabKernel", "UnpackSolverSlabKernel"):
        _public_device_operator(source, struct_name)
        assert f"fft_solver_detail::{struct_name}<" in solve
    assert "for_each_cell(valid, [=" not in solve
    assert "[=](const CellIndex<Dim>& cell)" not in solve


def test_system_frequency_functor_removes_device_lambda_from_private_member() -> None:
    source = SYSTEM_INSTALL_SOURCE.read_text()
    struct_offset = source.index("struct CoupledSourceMaximumFrequency")
    template_offset = source.rfind("template", 0, struct_offset)
    struct = _braced_definition(source, "struct CoupledSourceMaximumFrequency")

    _public_frequency_operator_contains_device_lambda(source)
    assert source[template_offset:struct_offset] == "template <int Dim>\n"
    assert "CoupledSourceMaximumFrequency<Dim>{" in source
    assert "std::function<const MultiFab<Dim>&(int)> state_for_block;" in struct
    _contains_no_device_lambda(
        source,
        "void System<Dim>::add_coupled_source_prepared_",
        "void System<Dim>::add_coupled_source(",
    )
