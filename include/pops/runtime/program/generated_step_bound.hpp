#pragma once

/// @file
/// @brief One reduction of a generated cell step bound over the valid accepted state.
///
/// The generated entry supplies this rank's summary. The core merges it with
/// dispatch status in the same rank reduction (spec 0.3 §2.6.2, PF8).

#include <pops/mesh/execution/for_each.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pops::runtime::program {

static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53,
              "cell step selection requires IEEE binary64");

inline constexpr double kUnboundedStep = std::numeric_limits<double>::infinity();
inline constexpr std::uint64_t kNoStepBoundCell = std::numeric_limits<std::uint64_t>::max();

struct StepBoundReduction {
  double bound = kUnboundedStep;
  std::uint64_t nonfinite = kNoStepBoundCell;
  std::uint64_t nonpositive = kNoStepBoundCell;

  POPS_HD void join(const StepBoundReduction& other) {
    if (other.bound < bound)
      bound = other.bound;
    if (other.nonfinite < nonfinite)
      nonfinite = other.nonfinite;
    if (other.nonpositive < nonpositive)
      nonpositive = other.nonpositive;
  }
};

namespace detail {
template <int Dim, class F>
struct StepBoundCellReduction {
  using value_type = StepBoundReduction;
  Index<Dim> lower;
  Extent<Dim> extent;
  F kernel;

  POPS_HD void init(value_type& value) const { value = value_type{}; }
  POPS_HD void join(value_type& destination, const value_type& source) const {
    destination.join(source);
  }
  POPS_HD void operator()(std::int64_t ordinal, value_type& value) const {
    kernel(pops::detail::cell_index_from_ordinal(lower, extent, ordinal), value);
  }
};
}  // namespace detail

template <int Dim, class F>
StepBoundReduction reduce_step_bound_cells(const Box<Dim>& box, F kernel) {
  if (box.empty())
    return {};
  pops::detail::require_iterable_box(box);
  pops::detail::ensure_kokkos_initialized();
  const Kokkos::DefaultExecutionSpace execution{};
  StepBoundReduction result;
  Kokkos::parallel_reduce(
      "pops_cell_step_bound",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(
          execution, 0, box.numPts()),
      detail::StepBoundCellReduction<Dim, F>{box.lo, box.extent(), kernel}, result);
  return result;
}

/// Dispatch and installed-plan status have priority over invalid cell data.
enum class StepBoundDispatch : std::uint64_t {
  kReady,
  kDefaultFieldSolve,
  kEllipticSource,
  kStabilityConsumer,
  kSourceFrequency,
  kParabolicFrequency,
  kStabilityBound,
  kBlockCadence,
  kCoupledFrequency,
  kEvolvingBlockCount,
  kGlobalBounds,
  kProgramBound,
  kPrecision,
  kEvaluationException,
  kEntryRefused,
  kIdentityMismatch,
  kMissingEntry,
};

/// Positive binary64 encodings are increasing. Complement them for a maximum of the reverse
/// order. Zero is the neutral element when a cell or rank imposes no finite bound.
inline std::uint64_t encode_step_bound(double value) {
  return value > 0.0 && std::isfinite(value) ? ~std::bit_cast<std::uint64_t>(value) : 0;
}

inline double decode_step_bound(std::uint64_t value) {
  return value == 0 ? kUnboundedStep : std::bit_cast<double>(~value);
}

/// Both integer ordinals are reversed for MAX; their absent sentinel becomes zero.
inline std::array<std::uint64_t, 4> step_bound_words(StepBoundDispatch dispatch,
                                                     const StepBoundReduction& local) {
  return {static_cast<std::uint64_t>(dispatch), kNoStepBoundCell - local.nonfinite,
          kNoStepBoundCell - local.nonpositive, encode_step_bound(local.bound)};
}

/// int entry(System<Dim>*, StepBoundReduction*): generate one summary for this rank.
inline constexpr const char* kGeneratedStepBoundEntrySymbol = "pops_program_advance_step_bound_v2";
inline constexpr const char* kLegacyStepBoundEntrySymbol = "pops_program_advance_step_bound";
/// const char* identity(): the identity of the bound named by the installed Program.
inline constexpr const char* kGeneratedStepBoundIdentitySymbol =
    "pops_program_advance_step_bound_identity";

}  // namespace pops::runtime::program
