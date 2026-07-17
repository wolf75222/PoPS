#pragma once

/// @file
/// @brief Binary-scaled finite scalar arithmetic for Krylov recurrences.
///
/// `ScaledScalar` retains a finite value as `mantissa * 2^exponent`, so recurrence products,
/// quotients, and signed cancellations do not first have to fit in `Real`.

#include <pops/core/foundation/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_MathematicalFunctions.hpp>
#endif

namespace pops {
namespace detail {

// Kokkos supplies a device-safe `isfinite`, but deliberately does not wrap frexp/ldexp.  SYCL
// exposes the latter in `sycl`, while CUDA/HIP and the host expose them in `std`; keep that
// backend distinction in one narrow layer rather than leaking it into the recurrence arithmetic.
namespace scaled_scalar_math {

POPS_HD [[nodiscard]] inline bool isfinite(Real value) {
#if defined(POPS_HAS_KOKKOS)
  return Kokkos::isfinite(value);
#else
  return std::isfinite(value);
#endif
}

POPS_HD [[nodiscard]] inline Real frexp(Real value, int* exponent) {
#if defined(KOKKOS_ENABLE_SYCL)
  return sycl::frexp(value, exponent);
#else
  return std::frexp(value, exponent);
#endif
}

POPS_HD [[nodiscard]] inline Real ldexp(Real value, int exponent) {
#if defined(KOKKOS_ENABLE_SYCL)
  return sycl::ldexp(value, exponent);
#else
  return std::ldexp(value, exponent);
#endif
}

}  // namespace scaled_scalar_math

/// Internal classification retained separately from the mantissa/exponent pair.
///
/// A non-finite input or an invalid product/quotient is never silently converted to zero.  It is
/// represented by `kNonFinite`, which propagates through subsequent multiplicative operations.
enum class ScaledScalarState : std::uint8_t { kZero, kFinite, kNonFinite };

/// Allocation-free binary-scaled coefficient for host and device Krylov recurrence arithmetic.
class ScaledScalar {
 public:
  /// Construct a normalized representation of a finite `Real`, or preserve its exceptional state.
  POPS_HD [[nodiscard]] static ScaledScalar from(Real value) {
    if (value == Real(0))
      return zero();
    if (!finite(value))
      return nonfinite();

    int exponent = 0;
    return ScaledScalar{ScaledScalarState::kFinite, scaled_scalar_math::frexp(value, &exponent),
                        exponent};
  }

  /// Exact zero, represented independently of a finite mantissa/exponent pair.
  POPS_HD [[nodiscard]] static constexpr ScaledScalar zero() {
    return ScaledScalar{ScaledScalarState::kZero, Real(0), 0};
  }

  /// Invalid or non-finite coefficient. It must be handled before field application.
  POPS_HD [[nodiscard]] static constexpr ScaledScalar nonfinite() {
    return ScaledScalar{ScaledScalarState::kNonFinite, Real(0), 0};
  }

  /// Multiply without first materializing the potentially overflowing coefficient.
  POPS_HD [[nodiscard]] static ScaledScalar product(const ScaledScalar& left,
                                                    const ScaledScalar& right) {
    if (!left.is_finite() || !right.is_finite())
      return nonfinite();
    if (left.is_zero() || right.is_zero())
      return zero();

    std::int64_t exponent = 0;
    if (!combine_exponents(left.exponent_, right.exponent_, exponent))
      return nonfinite();
    return normalize(left.mantissa_ * right.mantissa_, exponent);
  }

  /// Divide without first materializing the potentially overflowing quotient.
  POPS_HD [[nodiscard]] static ScaledScalar quotient(const ScaledScalar& numerator,
                                                     const ScaledScalar& denominator) {
    if (!numerator.is_finite() || !denominator.is_finite() || denominator.is_zero())
      return nonfinite();
    if (numerator.is_zero())
      return zero();

    std::int64_t negated_denominator_exponent = 0;
    if (!negate_exponent(denominator.exponent_, negated_denominator_exponent))
      return nonfinite();
    std::int64_t exponent = 0;
    if (!combine_exponents(numerator.exponent_, negated_denominator_exponent, exponent))
      return nonfinite();
    return normalize(numerator.mantissa_ / denominator.mantissa_, exponent);
  }

  /// Return `left + right` without materializing either summand.  The smaller operand is rounded
  /// away only after it is below the precision of the larger normalized mantissa; this is the same
  /// binary rounding decision a correctly staged `Real` addition would make, without an exponent
  /// range limit.  Exact cancellation remains an exact scaled zero.
  POPS_HD [[nodiscard]] static ScaledScalar sum(const ScaledScalar& left,
                                                const ScaledScalar& right) {
    if (!left.is_finite() || !right.is_finite())
      return nonfinite();
    if (left.is_zero())
      return right;
    if (right.is_zero())
      return left;

    const ScaledScalar* larger = &left;
    const ScaledScalar* smaller = &right;
    if (right.exponent_ > left.exponent_) {
      larger = &right;
      smaller = &left;
    }
    // `frexp` normalizes every finite mantissa to [0.5, 1).  Beyond this gap the smaller term
    // cannot alter a `Real` mantissa, and avoiding an out-of-range `int` shift keeps this kernel
    // device-clean even for coefficients produced by many recurrence operations.
    constexpr std::int64_t kRoundingGap =
        static_cast<std::int64_t>(std::numeric_limits<Real>::digits) + 2;
    if (larger->exponent_ > std::numeric_limits<std::int64_t>::min() + kRoundingGap &&
        smaller->exponent_ < larger->exponent_ - kRoundingGap)
      return *larger;

    const std::int64_t exponent_gap = larger->exponent_ - smaller->exponent_;

    const Real aligned_smaller =
        scaled_scalar_math::ldexp(smaller->mantissa_, -static_cast<int>(exponent_gap));
    return normalize(larger->mantissa_ + aligned_smaller, larger->exponent_);
  }

  POPS_HD [[nodiscard]] static ScaledScalar difference(const ScaledScalar& left,
                                                       const ScaledScalar& right) {
    return sum(left, negated(right));
  }

  POPS_HD [[nodiscard]] static ScaledScalar negated(const ScaledScalar& value) {
    if (!value.is_finite())
      return nonfinite();
    if (value.is_zero())
      return zero();
    return ScaledScalar{ScaledScalarState::kFinite, -value.mantissa_, value.exponent_};
  }

  /// Form `left_coefficient * left_value + right_coefficient * right_value` in one scaled
  /// expression.  In particular, a pair of individually non-materializable products may cancel to
  /// a finite `Real`; only an unrepresentable final result is rejected.
  POPS_HD [[nodiscard]] static bool try_sum_products(const ScaledScalar& left_coefficient,
                                                     Real left_value,
                                                     const ScaledScalar& right_coefficient,
                                                     Real right_value, Real& result) {
    const ScaledScalar left = product(left_coefficient, from(left_value));
    const ScaledScalar right = product(right_coefficient, from(right_value));
    return sum(left, right).try_materialize(result);
  }

  /// Three-term variant used by the full BiCGStab iterate update.  Keeping the cancellation in one
  /// kernel prevents an overflowing intermediate field value from poisoning a finite final cell.
  POPS_HD [[nodiscard]] static bool try_sum_products(const ScaledScalar& first_coefficient,
                                                     Real first_value,
                                                     const ScaledScalar& second_coefficient,
                                                     Real second_value,
                                                     const ScaledScalar& third_coefficient,
                                                     Real third_value, Real& result) {
    const ScaledScalar first = product(first_coefficient, from(first_value));
    const ScaledScalar second = product(second_coefficient, from(second_value));
    const ScaledScalar third = product(third_coefficient, from(third_value));
    return sum(sum(first, second), third).try_materialize(result);
  }

  /// Compare magnitudes without converting an extreme coefficient to `Real`.
  POPS_HD [[nodiscard]] static bool abs_less_equal(const ScaledScalar& left,
                                                   const ScaledScalar& right) {
    if (!left.is_finite() || !right.is_finite())
      return false;
    if (left.is_zero())
      return true;
    if (right.is_zero())
      return false;
    if (left.exponent_ != right.exponent_)
      return left.exponent_ < right.exponent_;
    const Real left_magnitude = left.mantissa_ < Real(0) ? -left.mantissa_ : left.mantissa_;
    const Real right_magnitude = right.mantissa_ < Real(0) ? -right.mantissa_ : right.mantissa_;
    return left_magnitude <= right_magnitude;
  }

  /// Apply this coefficient to one finite field value without materializing the coefficient.
  ///
  /// Returns false and assigns NaN if the input is non-finite, this coefficient is non-finite, or
  /// the mathematical result cannot be represented as a non-zero finite `Real`.  Finite results
  /// are reconstructed in bounded base-two stages, preserving cellwise values such as
  /// `(3.2e199 * 1e109) * 0.03125` even though the parenthesized coefficient overflows `Real`.
  POPS_HD [[nodiscard]] bool try_apply(Real value, Real& result) const {
    result = quiet_nan();
    if (!is_finite() || !finite(value))
      return false;
    if (is_zero() || value == Real(0)) {
      result = Real(0);
      return true;
    }

    int value_exponent = 0;
    const Real value_mantissa = scaled_scalar_math::frexp(value, &value_exponent);
    std::int64_t exponent = 0;
    if (!combine_exponents(exponent_, value_exponent, exponent))
      return false;
    return materialize(mantissa_ * value_mantissa, exponent, result);
  }

  /// Checked conversion for callers that genuinely need a scalar `Real`.
  POPS_HD [[nodiscard]] bool try_materialize(Real& result) const {
    result = quiet_nan();
    if (!is_finite())
      return false;
    if (is_zero()) {
      result = Real(0);
      return true;
    }
    return materialize(mantissa_, exponent_, result);
  }

  POPS_HD [[nodiscard]] constexpr ScaledScalarState state() const { return state_; }
  POPS_HD [[nodiscard]] constexpr bool is_zero() const {
    return state_ == ScaledScalarState::kZero;
  }
  POPS_HD [[nodiscard]] constexpr bool is_finite() const {
    return state_ == ScaledScalarState::kZero || state_ == ScaledScalarState::kFinite;
  }
  POPS_HD [[nodiscard]] constexpr bool is_nonfinite() const {
    return state_ == ScaledScalarState::kNonFinite;
  }
  POPS_HD [[nodiscard]] constexpr Real mantissa() const { return mantissa_; }
  POPS_HD [[nodiscard]] constexpr std::int64_t exponent() const { return exponent_; }

 private:
  static constexpr int kExponentStage = 512;
  static constexpr std::int64_t kMinMaterializableExponent =
      static_cast<std::int64_t>(std::numeric_limits<Real>::min_exponent) -
      static_cast<std::int64_t>(std::numeric_limits<Real>::digits) + 1;
  static constexpr std::int64_t kMaxMaterializableExponent =
      static_cast<std::int64_t>(std::numeric_limits<Real>::max_exponent);

  constexpr ScaledScalar(ScaledScalarState state, Real mantissa, std::int64_t exponent)
      : state_(state), mantissa_(mantissa), exponent_(exponent) {}

  POPS_HD [[nodiscard]] static bool finite(Real value) {
    return scaled_scalar_math::isfinite(value);
  }

  POPS_HD [[nodiscard]] static Real quiet_nan() { return std::numeric_limits<Real>::quiet_NaN(); }

  POPS_HD [[nodiscard]] static bool combine_exponents(std::int64_t left, std::int64_t right,
                                                      std::int64_t& result) {
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
      return false;
    result = left + right;
    return true;
  }

  POPS_HD [[nodiscard]] static bool negate_exponent(std::int64_t value, std::int64_t& result) {
    if (value == std::numeric_limits<std::int64_t>::min())
      return false;
    result = -value;
    return true;
  }

  POPS_HD [[nodiscard]] static ScaledScalar normalize(Real mantissa, std::int64_t exponent) {
    if (!finite(mantissa))
      return nonfinite();
    if (mantissa == Real(0))
      return zero();

    int adjustment = 0;
    mantissa = scaled_scalar_math::frexp(mantissa, &adjustment);
    if (!combine_exponents(exponent, adjustment, exponent))
      return nonfinite();
    return ScaledScalar{ScaledScalarState::kFinite, mantissa, exponent};
  }

  POPS_HD [[nodiscard]] static bool materialize(Real mantissa, std::int64_t exponent,
                                                Real& result) {
    result = quiet_nan();
    if (!finite(mantissa) || mantissa == Real(0))
      return false;

    int adjustment = 0;
    mantissa = scaled_scalar_math::frexp(mantissa, &adjustment);
    if (!combine_exponents(exponent, adjustment, exponent) ||
        exponent < kMinMaterializableExponent || exponent > kMaxMaterializableExponent)
      return false;

    Real staged = mantissa;
    while (exponent > kExponentStage) {
      staged = scaled_scalar_math::ldexp(staged, kExponentStage);
      if (!finite(staged))
        return false;
      exponent -= kExponentStage;
    }
    while (exponent < -kExponentStage) {
      staged = scaled_scalar_math::ldexp(staged, -kExponentStage);
      if (!finite(staged) || staged == Real(0))
        return false;
      exponent += kExponentStage;
    }
    staged = scaled_scalar_math::ldexp(staged, static_cast<int>(exponent));
    if (!finite(staged) || staged == Real(0))
      return false;
    result = staged;
    return true;
  }

  ScaledScalarState state_ = ScaledScalarState::kZero;
  Real mantissa_ = Real(0);
  std::int64_t exponent_ = 0;
};

}  // namespace detail
}  // namespace pops
