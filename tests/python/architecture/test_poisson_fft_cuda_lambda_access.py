"""CUDA compilation contract for PoissonFFT device-kernel launch helpers.

NVCC rejects an extended ``__host__ __device__`` lambda when its lexical enclosing
member function is private or protected.  Kokkos expands ``KOKKOS_LAMBDA`` to that
kind of lambda for CUDA, so this structural contract prevents a host-only edit from
reintroducing the CUDA-only compiler failure.  It does not execute an FFT.
"""
from __future__ import annotations

from pathlib import Path


HEADER = (
    Path(__file__).resolve().parents[3]
    / "include/pops/numerics/elliptic/poisson/poisson_fft.hpp"
)

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
    source = HEADER.read_text()

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
