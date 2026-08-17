#pragma once

/// @file
/// @brief Base scalar types and the POPS_HD macro (host+device portability). Minimal foundation
///        with no external dependency; the switch to pde_core::Real waits for the distributed mesh.
///
/// `Real`: centralized scalar. Default binary64; a `--float32` / `-DPOPS_REAL_TYPE=float` build
/// selects binary32. All numerical computation uses it; do not write `double` directly in the
/// physics layer or the kernels.
///
/// `POPS_HD`: annotation for functions called inside Kokkos kernels on host AND device.
/// - Kokkos: KOKKOS_FUNCTION (portable Cuda/HIP/SYCL/CPU, without manual CUDA syntax).
///   KOKKOS_FUNCTION is preferred over KOKKOS_INLINE_FUNCTION so as not to add an implicit `inline`
///   on sites already marked `POPS_HD inline ...`.
/// - Direct CUDA/HIP (without Kokkos): __host__ __device__.
/// - Pure CPU: empty expansion.
/// INVARIANT: POPS_HD can only wrap device-clean code (no host object,
/// no std::vector, no vtable).

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Macros.hpp>
#define POPS_HD KOKKOS_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define POPS_HD __host__ __device__
#else
#define POPS_HD
#endif

#include <cstdint>
#include <type_traits>

#if !defined(POPS_REAL_TYPE)
#define POPS_REAL_TYPE double
#endif

namespace pops {

using Real = POPS_REAL_TYPE;
static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>,
              "pops::Real must be binary64 or binary32");
inline constexpr bool kRealIsBinary64 = std::is_same_v<Real, double>;
using RealBits = std::conditional_t<kRealIsBinary64, std::uint64_t, std::uint32_t>;
static_assert(sizeof(Real) == sizeof(RealBits));

/// Speed FLOOR for the CFL step policies (audit 2026-06, explicit constant instead of the
/// scattered literal 1e-30): w = max(reduced_speed, kCflSpeedFloor) avoids the division by zero
/// when a block has no wave (frozen transport / null field). WARNING: a system in which ALL
/// the speeds are null then receives a step ~cfl*h/1e-30, enormous -- that is the historical
/// behavior assumed (such a step transports nothing); diagnose it via last_dt_bound() ==
/// "degenerate" on the System side. Shared by System::step_cfl and low-level adaptive/AMR CFL
/// policies.
inline constexpr Real kCflSpeedFloor = kRealIsBinary64 ? Real(1e-30) : Real(1e-15);

/// Speed FLOOR for the AMR drift / wave-speed reductions (ADC-643, single source of the scattered
/// 1e-12): the seed and post-reduction clamp of amr_max_drift_speed / AmrCouplerMP::max_drift_speed /
/// AmrCouplerMP::max_wave_speed / AmrRuntime::max_speed, so a level with no wave still yields a finite
/// CFL speed. DISTINCT from kCflSpeedFloor (1e-30): this is the reduction seed/clamp, not the
/// dt = cfl*h/w division floor -- kept separate so neither value moves.
inline constexpr Real kAmrDriftSpeedFloor = Real(1e-12);

}  // namespace pops
