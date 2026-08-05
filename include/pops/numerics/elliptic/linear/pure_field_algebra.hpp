#pragma once

/// @file
/// @brief Side-effect-free field algebra used by prepared linear solves.
///
/// These operations deliberately bypass ProgramContext.  In particular, an AMR ProgramContext may
/// attach time-integration and reflux-ledger semantics to its public axpy/lincomb methods; Krylov
/// recurrences are private algebra on scratch fields and must never mutate that ledger.

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>

#ifdef POPS_HAS_KOKKOS
#include <Kokkos_MathematicalFunctions.hpp>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

namespace detail {

/// Device-clean fill over valid cells. Stencil applications overwrite their ghosts through the
/// authenticated boundary/halo plan before reading them, so prepared iterations do not need a host
/// fill of allocated ghost storage.
template <int Dim>
struct FillValidKernel {
  FieldView<Real, Dim> values;
  Real value;
  int component;
  POPS_HD void operator()(const Index<Dim>& index) const { values(index, component) = value; }
};

template <int Dim>
struct DivideValidKernel {
  FieldView<Real, Dim> values;
  Real divisor;
  int component;
  POPS_HD void operator()(const Index<Dim>& index) const { values(index, component) /= divisor; }
};

template <int Dim>
struct NormalizedDifferenceKernel {
  FieldView<Real, Dim> destination;
  FieldView<const Real, Dim> left, right;
  Real scale;
  int component;
  POPS_HD void operator()(const Index<Dim>& index) const {
    const Real difference = left(index, component) - right(index, component);
    const Real finite_max = std::numeric_limits<Real>::max();
    destination(index, component) =
        difference <= finite_max && difference >= -finite_max
            ? difference / scale
            : left(index, component) / scale - right(index, component) / scale;
  }
};

template <int Dim>
struct ExactValueMismatchKernel {
  FieldView<const Real, Dim> left, right;
  int component;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const std::uint64_t left_bits = Kokkos::bit_cast<std::uint64_t>(left(index, component));
    const std::uint64_t right_bits = Kokkos::bit_cast<std::uint64_t>(right(index, component));
    return left_bits == right_bits ? Real(0) : Real(1);
  }
};

template <int Dim>
struct ScaledDotKernel {
  FieldView<const Real, Dim> left, right;
  Real left_scale, right_scale;
  int component;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    return (left(index, component) / left_scale) * (right(index, component) / right_scale);
  }
};

template <int Dim>
struct AbsDotKernel {
  FieldView<const Real, Dim> left, right;
  int component;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real product = left(index, component) * right(index, component);
    return product < Real(0) ? -product : product;
  }
};

template <int Dim>
struct MeasuredDotKernel {
  FieldView<const Real, Dim> left, right;
  Real measure;
  int component;
  bool absolute;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real product = left(index, component) * right(index, component) * measure;
    return absolute && product < Real(0) ? -product : product;
  }
};

template <int Dim>
void fill_valid(MultiFab<Dim>& field, Real value) {
  for (int local = 0; local < field.local_size(); ++local) {
    const FieldView<Real, Dim> values = field.fab(local).view();
    const Box<Dim> valid = field.box(local);
    for (int component = 0; component < field.ncomp(); ++component)
      for_each_cell(valid, FillValidKernel<Dim>{values, value, component});
  }
}

template <int Dim>
Real local_max_abs(const MultiFab<Dim>& field) {
  Real result = Real(0);
  for (int component = 0; component < field.ncomp(); ++component)
    result = std::max(result, norm_inf(field, component));
  return result;
}

template <int Dim>
bool local_exact_values_equal(const MultiFab<Dim>& left, const MultiFab<Dim>& right) {
  static_assert(sizeof(Real) == sizeof(std::uint64_t));
  if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
      left.local_rank() != right.local_rank() || left.ncomp() != right.ncomp() ||
      left.local_size() != right.local_size())
    throw std::invalid_argument("exact field comparison requires one vector space");
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < left.ncomp(); ++component) {
      if (for_each_cell_reduce_max(valid, ExactValueMismatchKernel<Dim>{left_values, right_values,
                                                                        component}) != Real(0))
        return false;
    }
  }
  return true;
}

inline Real rescale_product(Real normalized, Real left_scale, Real right_scale) {
  if (normalized == Real(0) || left_scale == Real(0) || right_scale == Real(0))
    return Real(0);
  if (!std::isfinite(static_cast<double>(normalized)) ||
      !std::isfinite(static_cast<double>(left_scale)) ||
      !std::isfinite(static_cast<double>(right_scale)))
    return std::numeric_limits<Real>::quiet_NaN();
  int normalized_exponent = 0;
  int left_exponent = 0;
  int right_exponent = 0;
  const Real normalized_mantissa = std::frexp(normalized, &normalized_exponent);
  const Real left_mantissa = std::frexp(left_scale, &left_exponent);
  const Real right_mantissa = std::frexp(right_scale, &right_exponent);
  return std::ldexp(normalized_mantissa * left_mantissa * right_mantissa,
                    normalized_exponent + left_exponent + right_exponent);
}

template <int Dim>
Real scaled_dot_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right, Real left_scale,
                      Real right_scale) {
  Real result = Real(0);
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < left.ncomp(); ++component)
      result += for_each_cell_reduce_sum(
          valid,
          ScaledDotKernel<Dim>{left_values, right_values, left_scale, right_scale, component});
  }
  return result;
}

static_assert(std::numeric_limits<Real>::is_iec559,
              "PureFieldAlgebra's robust dot fallback expects an IEEE-754 Real");

// The fallback only runs after the ordinary Kokkos dot underflows to zero or overflows.  Sixty-six
// bins cover every double product exponent in windows of 64 powers of two: 528 B of payload plus a
// non-finite witness, rather than a full per-exponent superaccumulator.  It consequently preserves
// cancellation between terms in the same window and products separated by many decades, but is not
// an exact/reproducible accumulator for arbitrary adversarial cancellation within one 64-bit window.
constexpr int kRobustDotInputLogbMin =
    std::numeric_limits<Real>::min_exponent - std::numeric_limits<Real>::digits;
constexpr int kRobustDotInputLogbMax = std::numeric_limits<Real>::max_exponent - 1;
constexpr int kRobustDotProductLogbMin = 2 * kRobustDotInputLogbMin;
constexpr int kRobustDotProductLogbMax = 2 * kRobustDotInputLogbMax;
constexpr int kRobustDotBandWidth = 64;
constexpr int kRobustDotBandCount =
    (kRobustDotProductLogbMax - kRobustDotProductLogbMin) / kRobustDotBandWidth + 1;
constexpr std::size_t kRobustDotNonfiniteIndex = static_cast<std::size_t>(kRobustDotBandCount);
constexpr std::size_t kRobustDotPayloadWidth = kRobustDotNonfiniteIndex + 1;
static_assert(kRobustDotPayloadWidth == 67,
              "the Python/native GMRES restart capacity assumes 67 robust-dot values");
using RobustDotBands = std::array<double, kRobustDotPayloadWidth>;

POPS_HD inline Real robust_dot_logb(Real value) {
#ifdef POPS_HAS_KOKKOS
  return Kokkos::logb(value);
#else
  return std::logb(value);
#endif
}

/// Multiply by 2^exponent without ever forming an overflowing reciprocal for a subnormal input.
POPS_HD inline Real robust_dot_scale_pow2(Real value, int exponent) {
  constexpr int kStageExponent = 512;
  while (exponent > kStageExponent) {
#ifdef POPS_HAS_KOKKOS
    value *= Kokkos::pow(Real(2), Real(kStageExponent));
#else
    value *= std::pow(Real(2), Real(kStageExponent));
#endif
    exponent -= kStageExponent;
  }
  while (exponent < -kStageExponent) {
#ifdef POPS_HAS_KOKKOS
    value *= Kokkos::pow(Real(2), Real(-kStageExponent));
#else
    value *= std::pow(Real(2), Real(-kStageExponent));
#endif
    exponent += kStageExponent;
  }
#ifdef POPS_HAS_KOKKOS
  return value * Kokkos::pow(Real(2), Real(exponent));
#else
  return value * std::pow(Real(2), Real(exponent));
#endif
}

template <int Dim>
struct RobustDotBandKernel {
  FieldView<const Real, Dim> left, right;
  int component;
  int band;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real x = left(index, component);
    const Real y = right(index, component);
    const Real finite_max = std::numeric_limits<Real>::max();
    if (!(x <= finite_max && x >= -finite_max && y <= finite_max && y >= -finite_max) || x == 0 ||
        y == 0)
      return Real(0);
    const Real x_abs = x < 0 ? -x : x;
    const Real y_abs = y < 0 ? -y : y;
    const int x_exponent = static_cast<int>(robust_dot_logb(x_abs));
    const int y_exponent = static_cast<int>(robust_dot_logb(y_abs));
    const int product_exponent = x_exponent + y_exponent;
    const int product_band = (product_exponent - kRobustDotProductLogbMin) / kRobustDotBandWidth;
    if (product_band != band)
      return Real(0);
    const int band_exponent = kRobustDotProductLogbMin + band * kRobustDotBandWidth;
    const Real x_mantissa = robust_dot_scale_pow2(x, -x_exponent);
    const Real y_mantissa = robust_dot_scale_pow2(y, -y_exponent);
    return robust_dot_scale_pow2(x_mantissa * y_mantissa, product_exponent - band_exponent);
  }
};

template <int Dim>
struct RobustDotNonfiniteKernel {
  FieldView<const Real, Dim> left, right;
  int component;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real x = left(index, component);
    const Real y = right(index, component);
    const Real finite_max = std::numeric_limits<Real>::max();
    return !(x <= finite_max && x >= -finite_max && y <= finite_max && y >= -finite_max) ? Real(1)
                                                                                         : Real(0);
  }
};

template <int Dim>
struct RobustDotActivityKernel {
  FieldView<const Real, Dim> left, right;
  int component;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real x = left(index, component);
    const Real y = right(index, component);
    return x != Real(0) && y != Real(0) ? Real(1) : Real(0);
  }
};

template <int Dim>
Real robust_dot_band_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right, int band,
                           bool all_components) {
  Real result = Real(0);
  const int component_count = all_components ? left.ncomp() : 1;
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < component_count; ++component)
      result += for_each_cell_reduce_sum(
          valid, RobustDotBandKernel<Dim>{left_values, right_values, component, band});
  }
  return result;
}

template <int Dim>
Real robust_dot_nonfinite_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                bool all_components) {
  Real result = Real(0);
  const int component_count = all_components ? left.ncomp() : 1;
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < component_count; ++component)
      result += for_each_cell_reduce_sum(
          valid, RobustDotNonfiniteKernel<Dim>{left_values, right_values, component});
  }
  return result;
}

template <int Dim>
Real robust_dot_activity_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                               bool all_components) {
  Real result = Real(0);
  const int component_count = all_components ? left.ncomp() : 1;
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < component_count; ++component)
      result += for_each_cell_reduce_sum(
          valid, RobustDotActivityKernel<Dim>{left_values, right_values, component});
  }
  return result;
}

inline Real robust_dot_reconstruct(const double* bands) {
  int high = kRobustDotBandCount - 1;
  while (high >= 0 && bands[static_cast<std::size_t>(high)] == 0.0)
    --high;
  if (high < 0)
    return Real(0);
  const int high_exponent = kRobustDotProductLogbMin + high * kRobustDotBandWidth;
  double sum = 0.0;
  double compensation = 0.0;
  for (int band = high; band >= 0; --band) {
    const int exponent = kRobustDotProductLogbMin + band * kRobustDotBandWidth;
    const double term = std::ldexp(bands[static_cast<std::size_t>(band)], exponent - high_exponent);
    const double corrected = term - compensation;
    const double next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
  }
  return static_cast<Real>(std::ldexp(sum, high_exponent));
}

inline Real robust_dot_reconstruct(const RobustDotBands& bands) {
  return robust_dot_reconstruct(bands.data());
}

template <int Dim>
void robust_dot_local_payload(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                              bool all_components, double* payload) {
  for (int band = 0; band < kRobustDotBandCount; ++band)
    payload[static_cast<std::size_t>(band)] =
        static_cast<double>(robust_dot_band_local(left, right, band, all_components));
  payload[kRobustDotNonfiniteIndex] =
      static_cast<double>(robust_dot_nonfinite_local(left, right, all_components));
}

template <int Dim>
Real robust_dot_global_value(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                             bool all_components) {
  RobustDotBands bands{};
  robust_dot_local_payload(left, right, all_components, bands.data());
  all_reduce_sum_inplace(bands.data(), bands.size());
  if (bands[kRobustDotNonfiniteIndex] != 0.0)
    return std::numeric_limits<Real>::quiet_NaN();
  return robust_dot_reconstruct(bands);
}

template <int Dim>
Real robust_dot_owned_value(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                            bool all_components, const PreparedVectorDistribution<Dim>& ownership,
                            std::span<double> scratch, const ExecutionLane& lane) {
  RobustDotBands bands{};
  robust_dot_local_payload(left, right, all_components, bands.data());
  ownership.reduce_sum_values(bands, scratch, "prepared robust dot product", lane);
  if (bands[kRobustDotNonfiniteIndex] != 0.0)
    return std::numeric_limits<Real>::quiet_NaN();
  return robust_dot_reconstruct(bands);
}

template <int Dim>
Real fast_dot_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right, bool all_components) {
  return all_components ? pops::dot_all_local(left, right) : pops::dot_local(left, right);
}

template <int Dim>
Real fast_dot_global(const MultiFab<Dim>& left, const MultiFab<Dim>& right, bool all_components) {
  return all_components ? pops::dot_all(left, right) : pops::dot(left, right);
}

template <int Dim>
Real fast_dot_owned(const MultiFab<Dim>& left, const MultiFab<Dim>& right, bool all_components,
                    const PreparedVectorDistribution<Dim>& ownership, std::span<double> scratch,
                    const ExecutionLane& lane) {
  double value = static_cast<double>(fast_dot_local(left, right, all_components));
  ownership.reduce_sum_values(std::span<double>(&value, 1), scratch, "prepared dot product", lane);
  return static_cast<Real>(value);
}

template <int Dim>
Real absolute_dot_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right) {
  Real result = Real(0);
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < left.ncomp(); ++component)
      result +=
          for_each_cell_reduce_sum(valid, AbsDotKernel<Dim>{left_values, right_values, component});
  }
  return result;
}

template <int Dim>
Real absolute_dot_owned(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                        const PreparedVectorDistribution<Dim>& ownership, std::span<double> scratch,
                        const ExecutionLane& lane) {
  double value = static_cast<double>(absolute_dot_local(left, right));
  ownership.reduce_sum_values(std::span<double>(&value, 1), scratch,
                              "prepared absolute inner product", lane);
  return static_cast<Real>(value);
}

template <int Dim>
Real measured_dot_local(const MultiFab<Dim>& left, const MultiFab<Dim>& right, Real measure,
                        bool absolute) {
  Real result = Real(0);
  for (int local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
    const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
    const Box<Dim> valid = left.box(local);
    for (int component = 0; component < left.ncomp(); ++component)
      result += for_each_cell_reduce_sum(
          valid, MeasuredDotKernel<Dim>{left_values, right_values, measure, component, absolute});
  }
  return result;
}

template <int Dim>
Real measured_dot_owned(const MultiFab<Dim>& left, const MultiFab<Dim>& right, Real measure,
                        bool absolute, const PreparedVectorDistribution<Dim>& ownership,
                        std::span<double> scratch, const ExecutionLane& lane) {
  double value = static_cast<double>(measured_dot_local(left, right, measure, absolute));
  ownership.reduce_sum_values(
      std::span<double>(&value, 1), scratch,
      absolute ? "prepared absolute nullspace pairing" : "prepared nullspace pairing", lane);
  return static_cast<Real>(value);
}

inline bool robust_dot_fallback_needed(Real value) {
  return value == Real(0) || !std::isfinite(static_cast<double>(value));
}

template <int Dim>
Real robust_dot_after_fast_global(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                  bool all_components, Real fast) {
  if (!robust_dot_fallback_needed(fast))
    return fast;
  if (fast == Real(0)) {
    const Real activity = static_cast<Real>(all_reduce_sum(
        static_cast<double>(robust_dot_activity_local(left, right, all_components))));
    if (activity == Real(0))
      return Real(0);
  }
  return robust_dot_global_value(left, right, all_components);
}

template <int Dim>
Real robust_dot_after_fast_owned(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                 bool all_components, Real fast,
                                 const PreparedVectorDistribution<Dim>& ownership,
                                 std::span<double> scratch, const ExecutionLane& lane) {
  if (!robust_dot_fallback_needed(fast))
    return fast;
  if (fast == Real(0)) {
    double activity = static_cast<double>(robust_dot_activity_local(left, right, all_components));
    ownership.reduce_sum_values(std::span<double>(&activity, 1), scratch,
                                "prepared dot-product activity", lane);
    if (activity == 0.0)
      return Real(0);
  }
  return robust_dot_owned_value(left, right, all_components, ownership, scratch, lane);
}

template <typename Value, typename Allocator = std::allocator<Value>>
[[nodiscard]] std::vector<Value, Allocator> materialize_collective_scratch(
    std::size_t count, Value initial_value, const char* where, const ExecutionLane& lane) {
  std::vector<Value, Allocator> scratch;
  long allocation_failure_local = 0;
  try {
    scratch.assign(count, initial_value);
  } catch (...) {
    allocation_failure_local = 1;
  }
  if (all_reduce_max(allocation_failure_local, lane) != 0)
    throw std::runtime_error(std::string(where) +
                             ": scratch allocation failed on at least one communicator rank");
  return scratch;
}

template <int Dim>
void require_public_field_distribution(const MultiFab<Dim>& value,
                                       PreparedVectorDistribution<Dim> distribution,
                                       const char* where, const ExecutionLane& lane) {
  // Public algebra has no prepared authority, so it authenticates both the exact layout and the
  // complete replica contents. Prepared hot paths bypass this boundary after their own one-time
  // solve preflight.
  require_collective_field_distribution_layout(value, distribution, where, lane);
  auto storage = materialize_collective_scratch<char, comm_allocator<char>>(
      distribution.validation_scratch_byte_count(), char{0}, where, lane);
  distribution.require_exact_values(value, storage, where, lane);
}

template <int Dim>
Real max_abs_owned_unchecked(const MultiFab<Dim>& value,
                             const PreparedVectorDistribution<Dim>& ownership,
                             std::span<double> scratch, const ExecutionLane& lane) {
  double result = static_cast<double>(local_max_abs(value));
  ownership.reduce_max_values(std::span<double>(&result, 1), scratch,
                              "prepared vector maximum norm", lane);
  return static_cast<Real>(result);
}

template <int Dim>
Real scale_safe_norm_owned_unchecked(const MultiFab<Dim>& value,
                                     const PreparedVectorDistribution<Dim>& ownership,
                                     std::span<double> scratch, const ExecutionLane& lane) {
  const Real scale = max_abs_owned_unchecked(value, ownership, scratch, lane);
  if (!std::isfinite(static_cast<double>(scale)))
    return std::numeric_limits<Real>::quiet_NaN();
  if (scale == Real(0))
    return Real(0);
  double normalized_square = static_cast<double>(scaled_dot_local(value, value, scale, scale));
  ownership.reduce_sum_values(std::span<double>(&normalized_square, 1), scratch,
                              "prepared normalized vector square", lane);
  if (!std::isfinite(normalized_square) || normalized_square < 0.0)
    return std::numeric_limits<Real>::quiet_NaN();
  return rescale_product(std::sqrt(static_cast<Real>(normalized_square)), scale, Real(1));
}

}  // namespace detail

struct PureFieldAlgebra {
  template <int Dim>
  static bool same_vector_space(const MultiFab<Dim>& left, const MultiFab<Dim>& right) {
    return left.layout() == right.layout() && left.distribution() == right.distribution() &&
           left.local_rank() == right.local_rank() && left.ncomp() == right.ncomp();
  }

  template <int Dim>
  static void require_same_vector_space(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                        const char* where) {
    if (!same_vector_space(left, right))
      throw std::invalid_argument(std::string(where) +
                                  ": fields do not share box, distribution, and component space");
  }

  template <int Dim>
  static void zero(MultiFab<Dim>& value) {
    value.set_val(Real(0));
  }

  /// Global maximum magnitude over every valid component. Non-finite input remains a collective
  /// non-finite witness so no rank can silently hide an invalid sample.
  template <int Dim>
  static Real max_abs(
      const MultiFab<Dim>& value,
      PreparedVectorDistribution<Dim> ownership = PreparedVectorDistribution<Dim>::Distributed) {
    const ExecutionLane lane = ExecutionLane::world();
    detail::require_public_field_distribution(value, ownership, "PureFieldAlgebra::max_abs", lane);
    auto scratch = detail::materialize_collective_scratch<double>(
        ownership.reduction_scratch_value_count(1), 0.0, "PureFieldAlgebra::max_abs", lane);
    return detail::max_abs_owned_unchecked(value, ownership, scratch, lane);
  }

  /// Fill valid cells on the active Kokkos execution space. Ghosts are deliberately left for the
  /// next typed halo/boundary fill; this is the initialization primitive for prepared hot paths.
  template <int Dim>
  static void fill_valid(MultiFab<Dim>& value, Real fill) {
    detail::fill_valid(value, fill);
  }
  template <int Dim>
  static void zero_valid(MultiFab<Dim>& value) {
    fill_valid(value, Real(0));
  }

  template <int Dim>
  static void copy(MultiFab<Dim>& destination, const MultiFab<Dim>& source) {
    require_same_vector_space(destination, source, "PureFieldAlgebra::copy");
    pops::lincomb(destination, Real(1), source, Real(0), source);
  }

  /// Copy valid cells and every allocated ghost cell without replacing either storage object. This
  /// is the allocation-free transaction primitive used when a prepared evaluation temporarily
  /// substitutes a live field and must restore its exact prior storage contents.
  template <int Dim>
  static void copy_allocated(MultiFab<Dim>& destination, const MultiFab<Dim>& source) {
    require_same_vector_space(destination, source, "PureFieldAlgebra::copy_allocated");
    if (destination.ghosts() != source.ghosts())
      throw std::invalid_argument(
          "PureFieldAlgebra::copy_allocated: fields have different ghost footprints");
    for (int local = 0; local < destination.local_size(); ++local) {
      FieldView<Real, Dim> out = destination.fab(local).view();
      const FieldView<const Real, Dim> in = std::as_const(source.fab(local)).view();
      const Box<Dim> allocated = destination.fab(local).grown_box();
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(allocated, mf_arith_detail::LincombKernel<Dim>{out, in, in, Real(1), Real(0),
                                                                     component});
    }
  }

  template <int Dim>
  static void axpy(MultiFab<Dim>& destination, Real coefficient, const MultiFab<Dim>& source) {
    require_same_vector_space(destination, source, "PureFieldAlgebra::axpy");
    pops::saxpy(destination, coefficient, source);
  }

  template <int Dim>
  static void lincomb(MultiFab<Dim>& destination, Real left_coefficient, const MultiFab<Dim>& left,
                      Real right_coefficient, const MultiFab<Dim>& right) {
    require_same_vector_space(destination, left, "PureFieldAlgebra::lincomb(left)");
    require_same_vector_space(destination, right, "PureFieldAlgebra::lincomb(right)");
    pops::lincomb(destination, left_coefficient, left, right_coefficient, right);
  }

  /// One collective full-vector dot product. Every rank calls the same reduction, including ranks
  /// with no local box.
  template <int Dim>
  static Real dot(
      const MultiFab<Dim>& left, const MultiFab<Dim>& right,
      PreparedVectorDistribution<Dim> ownership = PreparedVectorDistribution<Dim>::Distributed) {
    const ExecutionLane lane = ExecutionLane::world();
    detail::require_public_field_distribution(left, ownership, "PureFieldAlgebra::dot", lane);
    const long vector_space_mismatch =
        all_reduce_max(same_vector_space(left, right) ? 0L : 1L, lane);
    if (vector_space_mismatch != 0)
      throw std::invalid_argument(
          "PureFieldAlgebra::dot: fields do not share box, distribution, and component space");
    detail::require_public_field_distribution(right, ownership, "PureFieldAlgebra::dot(right)",
                                              lane);
    auto scratch = detail::materialize_collective_scratch<double>(
        ownership.reduction_scratch_value_count(detail::kRobustDotPayloadWidth), 0.0,
        "PureFieldAlgebra::dot", lane);
    const Real fast = detail::fast_dot_owned(left, right, true, ownership, scratch, lane);
    return detail::robust_dot_after_fast_owned(left, right, true, fast, ownership, scratch, lane);
  }

  template <int Dim>
  static Real absolute_dot(
      const MultiFab<Dim>& left, const MultiFab<Dim>& right,
      PreparedVectorDistribution<Dim> ownership = PreparedVectorDistribution<Dim>::Distributed) {
    const ExecutionLane lane = ExecutionLane::world();
    detail::require_public_field_distribution(left, ownership, "PureFieldAlgebra::absolute_dot",
                                              lane);
    const long vector_space_mismatch =
        all_reduce_max(same_vector_space(left, right) ? 0L : 1L, lane);
    if (vector_space_mismatch != 0)
      throw std::invalid_argument(
          "PureFieldAlgebra::absolute_dot: fields do not share vector space");
    detail::require_public_field_distribution(right, ownership,
                                              "PureFieldAlgebra::absolute_dot(right)", lane);
    auto scratch = detail::materialize_collective_scratch<double>(
        ownership.reduction_scratch_value_count(1), 0.0, "PureFieldAlgebra::absolute_dot", lane);
    return detail::absolute_dot_owned(left, right, ownership, scratch, lane);
  }

  template <int Dim>
  static Real norm(const MultiFab<Dim>& value, PreparedVectorDistribution<Dim> ownership =
                                                   PreparedVectorDistribution<Dim>::Distributed) {
    const ExecutionLane lane = ExecutionLane::world();
    detail::require_public_field_distribution(value, ownership, "PureFieldAlgebra::norm", lane);
    auto scratch = detail::materialize_collective_scratch<double>(
        ownership.reduction_scratch_value_count(detail::kRobustDotPayloadWidth), 0.0,
        "PureFieldAlgebra::norm", lane);
    return detail::scale_safe_norm_owned_unchecked(value, ownership, scratch, lane);
  }
};

namespace detail {

/// Unchecked algebra for an already authenticated prepared solve.  The public helpers above remain
/// defensive for transaction and extension call sites; the Krylov hot path validates its complete
/// vector space once when the problem/workspace are bound and must not rescan every box/rank vector
/// for each recurrence primitive.
struct PreparedFieldAlgebra {
  static constexpr std::size_t kRobustDotPayloadWidth = detail::kRobustDotPayloadWidth;

  template <int Dim>
  static void zero(MultiFab<Dim>& value) {
    fill_valid(value, Real(0));
  }

  template <int Dim>
  static void copy(MultiFab<Dim>& destination, const MultiFab<Dim>& source) {
    pops::lincomb(destination, Real(1), source, Real(0), source);
  }

  template <int Dim>
  static void axpy(MultiFab<Dim>& destination, Real coefficient, const MultiFab<Dim>& source) {
    pops::saxpy(destination, coefficient, source);
  }

  template <int Dim>
  static void lincomb(MultiFab<Dim>& destination, Real left_coefficient, const MultiFab<Dim>& left,
                      Real right_coefficient, const MultiFab<Dim>& right) {
    pops::lincomb(destination, left_coefficient, left, right_coefficient, right);
  }

  template <int Dim>
  static bool local_exact_values_equal(const MultiFab<Dim>& left, const MultiFab<Dim>& right) {
    return detail::local_exact_values_equal(left, right);
  }

  /// Divide valid cells directly instead of materializing 1/divisor. This remains defined when a
  /// finite subnormal equation scale has a non-representable reciprocal.
  template <int Dim>
  static void divide(MultiFab<Dim>& value, Real divisor) {
    for (int local = 0; local < value.local_size(); ++local) {
      FieldView<Real, Dim> values = value.fab(local).view();
      const Box<Dim> valid = value.box(local);
      for (int component = 0; component < value.ncomp(); ++component)
        for_each_cell(valid, DivideValidKernel<Dim>{values, divisor, component});
    }
  }

  /// destination = (left-right)/scale. Subtract first when the physical difference is finite so a
  /// large common affine response cancels before division; only an overflowing difference falls
  /// back to separately normalized operands. Pointwise aliasing is safe.
  template <int Dim>
  static void normalized_difference(MultiFab<Dim>& destination, const MultiFab<Dim>& left,
                                    const MultiFab<Dim>& right, Real scale) {
    for (int local = 0; local < destination.local_size(); ++local) {
      FieldView<Real, Dim> output = destination.fab(local).view();
      const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
      const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
      const Box<Dim> valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(valid, NormalizedDifferenceKernel<Dim>{output, left_values, right_values,
                                                             scale, component});
    }
  }

  template <int Dim>
  static Real max_abs(const MultiFab<Dim>& value, const PreparedVectorDistribution<Dim>& ownership,
                      std::span<double> scratch, const ExecutionLane& lane) {
    return detail::max_abs_owned_unchecked(value, ownership, scratch, lane);
  }

  template <int Dim>
  static Real dot(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                  const PreparedVectorDistribution<Dim>& ownership, std::span<double> scratch,
                  const ExecutionLane& lane) {
    const bool all_components = left.ncomp() != 1;
    const Real fast = detail::fast_dot_owned(left, right, all_components, ownership, scratch, lane);
    return detail::robust_dot_after_fast_owned(left, right, all_components, fast, ownership,
                                               scratch, lane);
  }

  template <int Dim>
  static Real dot(
      const MultiFab<Dim>& left, const MultiFab<Dim>& right,
      PreparedVectorDistribution<Dim> ownership = PreparedVectorDistribution<Dim>::Distributed) {
    const ExecutionLane lane = ExecutionLane::world();
    auto scratch = detail::materialize_collective_scratch<double>(
        ownership.reduction_scratch_value_count(kRobustDotPayloadWidth), 0.0,
        "PreparedFieldAlgebra::dot", lane);
    return dot(left, right, ownership, scratch, lane);
  }

  template <int Dim>
  static Real absolute_dot(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                           const PreparedVectorDistribution<Dim>& ownership,
                           std::span<double> scratch, const ExecutionLane& lane) {
    return detail::absolute_dot_owned(left, right, ownership, scratch, lane);
  }

  template <int Dim>
  static Real nullspace_pairing(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                Real cell_measure, bool absolute,
                                const PreparedVectorDistribution<Dim>& ownership,
                                std::span<double> scratch, const ExecutionLane& lane) {
    return detail::measured_dot_owned(left, right, cell_measure, absolute, ownership, scratch,
                                      lane);
  }

  template <int Dim>
  static Real local_dot(const MultiFab<Dim>& left, const MultiFab<Dim>& right) {
    const bool all_components = left.ncomp() != 1;
    return detail::fast_dot_local(left, right, all_components);
  }

  template <int Dim>
  static void local_robust_dot_payload(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                       double* payload) {
    const bool all_components = left.ncomp() != 1;
    detail::robust_dot_local_payload(left, right, all_components, payload);
  }

  static Real dot_from_global_robust_payload(const double* payload) {
    if (payload[detail::kRobustDotNonfiniteIndex] != 0.0)
      return std::numeric_limits<Real>::quiet_NaN();
    return detail::robust_dot_reconstruct(payload);
  }

  template <int Dim>
  static Real norm(const MultiFab<Dim>& value, const PreparedVectorDistribution<Dim>& ownership,
                   std::span<double> scratch, const ExecutionLane& lane) {
    const bool all_components = value.ncomp() != 1;
    // A zero square is handled by the existing max-scaled norm below.  Do not route an exact zero
    // through every exponent band merely to rediscover that a zero field has zero norm.
    const Real square =
        detail::fast_dot_owned(value, value, all_components, ownership, scratch, lane);
    if (std::isfinite(static_cast<double>(square)) && square > Real(0))
      return std::sqrt(square);
    if (square < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    // The ordinary one-collective dot is the fast path for normalized recurrences.  Fall back to
    // the max-scaled norm only when squaring a finite subnormal vector underflowed to zero, or when
    // an intermediate square overflowed.  This keeps the common hot path unchanged while allowing
    // GMRES to distinguish a true lucky breakdown from a representable subnormal Arnoldi column.
    return detail::scale_safe_norm_owned_unchecked(value, ownership, scratch, lane);
  }

  template <int Dim>
  static Real norm(const MultiFab<Dim>& value, PreparedVectorDistribution<Dim> ownership =
                                                   PreparedVectorDistribution<Dim>::Distributed) {
    const ExecutionLane lane = ExecutionLane::world();
    auto scratch = detail::materialize_collective_scratch<double>(
        ownership.reduction_scratch_value_count(kRobustDotPayloadWidth), 0.0,
        "PreparedFieldAlgebra::norm", lane);
    return norm(value, ownership, scratch, lane);
  }
};

}  // namespace detail

}  // namespace pops
